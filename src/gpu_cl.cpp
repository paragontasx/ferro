// gpu_cl.cpp — Intel iGPU (Xe-LP) backend via OpenCL, dlopen'd at runtime.
// Whole transformer layer loop runs on-device; GEMMs use int8 DP4A (cl_khr_integer_dot_product).
// Measured on N100 (ADL-N, Mesa rusticl): DP4A GEMM ~225-240 GF-equiv vs 56 GF CPU AVX2 ceiling.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include "gpu_cl.h"
#include "kernels.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>

namespace ferro { namespace gpu {

// ---------------------------------------------------------------- dlopen'd API table
#define CL_FUNCS(X) \
    X(clGetPlatformIDs) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext) \
    X(clCreateCommandQueueWithProperties) X(clCreateProgramWithSource) X(clBuildProgram) \
    X(clGetProgramBuildInfo) X(clCreateKernel) X(clSetKernelArg) X(clCreateBuffer) \
    X(clEnqueueWriteBuffer) X(clEnqueueReadBuffer) X(clEnqueueCopyBuffer) X(clEnqueueNDRangeKernel) X(clFinish) \
    X(clFlush) X(clReleaseMemObject) X(clReleaseKernel) X(clReleaseProgram) \
    X(clReleaseCommandQueue) X(clReleaseContext)

struct Api {
#define DECL(f) decltype(&::f) f = nullptr;
    CL_FUNCS(DECL)
#undef DECL
};
static Api cl;
static void* g_lib = nullptr;

#ifndef FERRO_GPU_TT
#define FERRO_GPU_TT 10
#endif
#ifndef FERRO_GPU_KB
#define FERRO_GPU_KB 16
#endif
#define ATTN_MAXS 1024

static const char* kSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

// ============================ int8 DP4A GEMM ============================
// activations -> int8 per (token, 32-block), symmetric (llama.cpp q8_0 activation scheme)
__kernel void quant_x(__global const float* restrict X, __global uint* restrict Xp,
                      __global float* restrict Xs, int cols) {
  const int t = get_global_id(1), b = get_global_id(0), nb = cols / 32;
  const __global float* x = X + (size_t)t * cols + b * 32;
  float8 v0 = vload8(0, x), v1 = vload8(1, x), v2 = vload8(2, x), v3 = vload8(3, x);
  float8 m = fmax(fmax(fabs(v0), fabs(v1)), fmax(fabs(v2), fabs(v3)));
  float4 m4 = fmax(m.lo, m.hi); float2 m2 = fmax(m4.lo, m4.hi);
  float amax = fmax(m2.x, m2.y);
  float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
  Xs[(size_t)t * nb + b] = d;
  __global uint* o = Xp + (size_t)t * (cols / 4) + b * 8;
  o[0] = as_uint(convert_char4_sat_rte(v0.lo * id)); o[1] = as_uint(convert_char4_sat_rte(v0.hi * id));
  o[2] = as_uint(convert_char4_sat_rte(v1.lo * id)); o[3] = as_uint(convert_char4_sat_rte(v1.hi * id));
  o[4] = as_uint(convert_char4_sat_rte(v2.lo * id)); o[5] = as_uint(convert_char4_sat_rte(v2.hi * id));
  o[6] = as_uint(convert_char4_sat_rte(v3.lo * id)); o[7] = as_uint(convert_char4_sat_rte(v3.hi * id));
}

// WI = 4 consecutive rows x TT tokens; WG stages X (TT tokens x KB blocks) in SLM.
__kernel void q8gemm(__global const uint* restrict Wp, __global const float* restrict Ws,
                     __global const uint* restrict Xp, __global const float* restrict Xs,
                     __global float* restrict Y, int rows, int cols, int T) {
  __local uint xq[TT * KB * 8];
  __local float xsc[TT * KB];
  const int lid = get_local_id(0), LW = get_local_size(0);
  const int r4 = get_global_id(0) * 4;
  const int t0 = get_group_id(1) * TT;
  const int nb = cols / 32, cw = cols / 4;
  float4 acc[TT];
  for (int j = 0; j < TT; ++j) acc[j] = (float4)(0.f);
  const int R4 = rows >> 2, gx = r4 >> 2;
  for (int b0 = 0; b0 < nb; b0 += KB) {
    const __global uint4* Wb = (const __global uint4*)Wp + (size_t)(b0 * 8) * R4 + gx;
    const __global float4* Sb = (const __global float4*)Ws + (size_t)b0 * R4 + gx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < TT * KB * 8; i += LW) {
      int j = i / (KB * 8), rem = i - j * (KB * 8); int t = min(t0 + j, T - 1);
      xq[i] = Xp[(size_t)t * cw + b0 * 8 + rem];
    }
    for (int i = lid; i < TT * KB; i += LW) {
      int j = i / KB, bb = i - j * KB; int t = min(t0 + j, T - 1);
      xsc[i] = Xs[(size_t)t * nb + b0 + bb];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int bb = 0; bb < KB; ++bb) {
      const int b = b0 + bb;
      // 32-bit typed indexing: 64-bit address multiplies are emulated on Xe-LP and cost ~1/3 of GEMM throughput
      uint4 w[8];
      for (int k = 0; k < 8; ++k) w[k] = Wb[(bb * 8 + k) * R4];
      const float4 ws = Sb[bb * R4];
      for (int j = 0; j < TT; ++j) {
        const __local uint* xp = xq + (j * KB + bb) * 8;
        uint4 xa = vload4(0, xp), xb = vload4(1, xp);
        int4 s = (int4)(0);
#define STEP(A, K, XV) \
        A.s0 = dot_acc_sat_4x8packed_ss_int(w[K].s0, XV, A.s0); \
        A.s1 = dot_acc_sat_4x8packed_ss_int(w[K].s1, XV, A.s1); \
        A.s2 = dot_acc_sat_4x8packed_ss_int(w[K].s2, XV, A.s2); \
        A.s3 = dot_acc_sat_4x8packed_ss_int(w[K].s3, XV, A.s3);
        STEP(s, 0, xa.s0) STEP(s, 1, xa.s1) STEP(s, 2, xa.s2) STEP(s, 3, xa.s3)
        STEP(s, 4, xb.s0) STEP(s, 5, xb.s1) STEP(s, 6, xb.s2) STEP(s, 7, xb.s3)
        acc[j] = mad(convert_float4(s), ws * xsc[j * KB + bb], acc[j]);
      }
    }
  }
  for (int j = 0; j < TT; ++j) if (t0 + j < T) vstore4(acc[j], 0, Y + (size_t)(t0 + j) * rows + r4);
}

// Swizzled q8gemm: consecutive work-groups share a weight row-slice and walk the token tiles, so a weight
// matrix larger than the LLC is streamed from DRAM ~once instead of once per token tile.
__kernel void q8gemm_sw(__global const uint* restrict Wp, __global const float* restrict Ws,
                     __global const uint* restrict Xp, __global const float* restrict Xs,
                     __global float* restrict Y, int rows, int cols, int T) {
  __local uint xq[TT * KB * 8];
  __local float xsc[TT * KB];
  const int lid = get_local_id(0), LW = get_local_size(0);
  const int ngx = get_num_groups(0), nty = get_num_groups(1);
  const int lin = get_group_id(1) * ngx + get_group_id(0);
  const int r4 = ((lin / nty) * LW + lid) * 4;
  const int t0 = (lin % nty) * TT;
  const int nb = cols / 32, cw = cols / 4;
  float4 acc[TT];
  for (int j = 0; j < TT; ++j) acc[j] = (float4)(0.f);
  const int R4 = rows >> 2, gx = r4 >> 2;
  for (int b0 = 0; b0 < nb; b0 += KB) {
    const __global uint4* Wb = (const __global uint4*)Wp + (size_t)(b0 * 8) * R4 + gx;
    const __global float4* Sb = (const __global float4*)Ws + (size_t)b0 * R4 + gx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < TT * KB * 8; i += LW) {
      int j = i / (KB * 8), rem = i - j * (KB * 8); int t = min(t0 + j, T - 1);
      xq[i] = Xp[(size_t)t * cw + b0 * 8 + rem];
    }
    for (int i = lid; i < TT * KB; i += LW) {
      int j = i / KB, bb = i - j * KB; int t = min(t0 + j, T - 1);
      xsc[i] = Xs[(size_t)t * nb + b0 + bb];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int bb = 0; bb < KB; ++bb) {
      const int b = b0 + bb;
      // 32-bit typed indexing: 64-bit address multiplies are emulated on Xe-LP and cost ~1/3 of GEMM throughput
      uint4 w[8];
      for (int k = 0; k < 8; ++k) w[k] = Wb[(bb * 8 + k) * R4];
      const float4 ws = Sb[bb * R4];
      for (int j = 0; j < TT; ++j) {
        const __local uint* xp = xq + (j * KB + bb) * 8;
        uint4 xa = vload4(0, xp), xb = vload4(1, xp);
        int4 s = (int4)(0);
#define STEP(A, K, XV) \
        A.s0 = dot_acc_sat_4x8packed_ss_int(w[K].s0, XV, A.s0); \
        A.s1 = dot_acc_sat_4x8packed_ss_int(w[K].s1, XV, A.s1); \
        A.s2 = dot_acc_sat_4x8packed_ss_int(w[K].s2, XV, A.s2); \
        A.s3 = dot_acc_sat_4x8packed_ss_int(w[K].s3, XV, A.s3);
        STEP(s, 0, xa.s0) STEP(s, 1, xa.s1) STEP(s, 2, xa.s2) STEP(s, 3, xa.s3)
        STEP(s, 4, xb.s0) STEP(s, 5, xb.s1) STEP(s, 6, xb.s2) STEP(s, 7, xb.s3)
        acc[j] = mad(convert_float4(s), ws * xsc[j * KB + bb], acc[j]);
      }
    }
  }
  for (int j = 0; j < TT; ++j) if (t0 + j < T) vstore4(acc[j], 0, Y + (size_t)(t0 + j) * rows + r4);
}

// ============================ layer ops ============================
// one work-group per (head h, token t); LS lanes reduce sum of squares
__kernel void rmsnorm(__global const float* x, int xoff, int ldx, __global float* y, int yoff, int ldy,
                      int cols, __global const float* w, int has_w, float eps) {
  __local float red[256];
  const int h = get_group_id(0), t = get_group_id(1);
  const int lid = get_local_id(0), LS = get_local_size(0);
  const __global float* xr = x + xoff + (size_t)t * ldx + h * cols;
  __global float* yr = y + yoff + (size_t)t * ldy + h * cols;
  float ss = 0.f;
  for (int c = lid; c < cols; c += LS) { float v = xr[c]; ss = mad(v, v, ss); }
  red[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = LS >> 1; s > 0; s >>= 1) { if (lid < s) red[lid] += red[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  const float inv = 1.0f / sqrt(red[0] / (float)cols + eps);
  for (int c = lid; c < cols; c += LS) yr[c] = xr[c] * inv * (has_w ? w[c] : 1.0f);
}

__kernel void rope(__global float* x, int xoff, int ld, int hd, int hlf,
                   __global const float* cosb, __global const float* sinb, __global const int* pos) {
  const int i = get_global_id(0), h = get_global_id(1), t = get_global_id(2);
  if (i >= hlf) return;
  __global float* v = x + xoff + (size_t)t * ld + h * hd;
  const int p = pos[t];
  const float c = cosb[(size_t)p * hlf + i], s = sinb[(size_t)p * hlf + i];
  const float x0 = v[i], x1 = v[i + hlf];
  v[i] = x0 * c - x1 * s;
  v[i + hlf] = x1 * c + x0 * s;
}

// one work-group per (q-head h, token t). Scores in SLM (segment length <= MAXS).
__kernel void attention(__global const float* C, int ldc, int qoff, int koff, int voff,
                        int nq, int nkv, int hd,
                        __global const int* seg_lo, __global const int* seg_hi, int window, float scale,
                        __global float* O, int ldo) {
  __local float p[MAXS];
  __local float qv[512];
  __local float red[256];
  const int h = get_group_id(0), t = get_group_id(1);
  const int lid = get_local_id(0), LS = get_local_size(0);
  const int kvh = h / (nq / nkv);
  const int lo = seg_lo[t], hi = seg_hi[t], n = hi - lo + 1;
  const __global float* q = C + (size_t)t * ldc + qoff + h * hd;
  for (int i = lid; i < hd; i += LS) qv[i] = q[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -1e30f;
  for (int k = lid; k < n; k += LS) {
    const int s = lo + k;
    float d;
    if (window > 0 && abs(t - s) >= window) d = -1e30f;
    else {
      const __global float* ks = C + (size_t)s * ldc + koff + kvh * hd;
      float4 a = (float4)(0.f);
      for (int i = 0; i < hd; i += 4) a = mad(vload4(0, qv + i), vload4(0, ks + i), a);
      d = (a.x + a.y + a.z + a.w) * scale;
    }
    p[k] = d; mx = fmax(mx, d);
  }
  red[lid] = mx; barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = LS >> 1; s > 0; s >>= 1) { if (lid < s) red[lid] = fmax(red[lid], red[lid + s]); barrier(CLK_LOCAL_MEM_FENCE); }
  mx = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
  float sm = 0.f;
  for (int k = lid; k < n; k += LS) { float e = p[k] < -1e29f ? 0.f : exp(p[k] - mx); p[k] = e; sm += e; }
  red[lid] = sm; barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = LS >> 1; s > 0; s >>= 1) { if (lid < s) red[lid] += red[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  const float inv = red[0] > 0.f ? 1.0f / red[0] : 0.f;
  __global float* o = O + (size_t)t * ldo + h * hd;
  for (int i = lid; i < hd; i += LS) {
    float a = 0.f;
    for (int k = 0; k < n; ++k) a = mad(p[k], C[(size_t)(lo + k) * ldc + voff + kvh * hd + i], a);
    o[i] = a * inv;
  }
}

inline float gelu1(float x) { return 0.5f * x * (1.0f + erf(x * 0.70710678118654752440f)); }
// Abramowitz-Stegun 7.1.26 erf (|err| < 1.5e-7) -- same approximation as the CPU AVX2 gelu
inline float8 gelu8(float8 x) {
  const float8 z = fabs(x) * 0.70710678118654752440f;
  const float8 t = 1.0f / (1.0f + 0.3275911f * z);
  float8 y = mad(t, (float8)(1.061405429f), (float8)(-1.453152027f));
  y = mad(y, t, (float8)(1.421413741f));
  y = mad(y, t, (float8)(-0.284496736f));
  y = mad(y, t, (float8)(0.254829592f));
  y = y * t;
  const float8 e = 1.0f - y * exp(-z * z);
  return 0.5f * x * (1.0f + copysign(e, x));
}

__kernel void gelu_mul(__global const float* G, int FF, __global float* F) {
  const int i = get_global_id(0), t = get_global_id(1);
  const __global float* g = G + (size_t)t * 2 * FF;
  F[(size_t)t * FF + i] = gelu1(g[i]) * g[FF + i];
}

__kernel void gelu_mulvec(__global float* I, __global const float* H, int hoff) {
  const int i = get_global_id(0);
  I[i] = gelu1(I[i]) * H[hoff + i];
}


// ============================ fused layer kernels (emit int8 GEMM inputs directly) ============================
inline void quant_store32(const float* v, __global uint* Xp, __global float* Xs, int t, int b, int cols) {
  float amax = 0.f;
  for (int i = 0; i < 32; ++i) amax = fmax(amax, fabs(v[i]));
  const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
  Xs[(size_t)t * (cols / 32) + b] = d;
  __global uint* o = Xp + (size_t)t * (cols / 4) + b * 8;
  for (int k = 0; k < 8; ++k)
    o[k] = as_uint(convert_char4_sat_rte((float4)(v[4*k], v[4*k+1], v[4*k+2], v[4*k+3]) * id));
}
inline void quant_store32_l(__local const float* v, __global uint* Xp, __global float* Xs, int t, int b, int cols) {
  float amax = 0.f;
  for (int i = 0; i < 32; ++i) amax = fmax(amax, fabs(v[i]));
  const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
  Xs[(size_t)t * (cols / 32) + b] = d;
  __global uint* o = Xp + (size_t)t * (cols / 4) + b * 8;
  for (int k = 0; k < 8; ++k)
    o[k] = as_uint(convert_char4_sat_rte(vload4(k, v) * id));
}
inline float wg_sum(float v, __local float* red) {
  const int lid = get_local_id(0), LS = get_local_size(0);
  barrier(CLK_LOCAL_MEM_FENCE);
  red[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = LS >> 1; s > 0; s >>= 1) { if (lid < s) red[lid] += red[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  return red[0];
}

// One WG per token row (cols <= 2048):
//   y   = ((has_r ? R : 0) + (has_d ? rms(D)*wd : 0)) * scale      (stored to OUT if has_out)
//   q   = norm_q ? rms(y)*wq : y      -> int8 (Xp, Xs) if do_q
__kernel void row_fused(__global const float* R, int has_r, __global const float* D, __global const float* wd, int has_d,
                        float scale, __global float* OUT, int has_out,
                        __global const float* wq, int norm_q, int do_q,
                        __global uint* Xp, __global float* Xs, int cols, float eps) {
  __local float y[2048];
  __local float red[256];
  const int t = get_group_id(0), lid = get_local_id(0), LS = get_local_size(0);
  const __global float* r = R + (size_t)t * cols;
  const __global float* d = D + (size_t)t * cols;
  float invd = 0.f;
  if (has_d) {
    float ss = 0.f;
    for (int c = lid; c < cols; c += LS) { float v = d[c]; ss = mad(v, v, ss); }
    invd = 1.0f / sqrt(wg_sum(ss, red) / (float)cols + eps);
  }
  float ssy = 0.f;
  for (int c = lid; c < cols; c += LS) {
    float v = (has_r ? r[c] : 0.f) + (has_d ? d[c] * invd * wd[c] : 0.f);
    v *= scale;
    y[c] = v; ssy = mad(v, v, ssy);
    if (has_out) OUT[(size_t)t * cols + c] = v;
  }
  if (!do_q) return;
  if (norm_q) {
    const float invq = 1.0f / sqrt(wg_sum(ssy, red) / (float)cols + eps);
    for (int c = lid; c < cols; c += LS) y[c] = y[c] * invq * wq[c];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int b = lid; b < cols / 32; b += LS) quant_store32_l(y + b * 32, Xp, Xs, t, b, cols);
}

// One WG per (head slot, token): slots [0,nq) q-heads (norm qw + rope), [nq,nq+nkv) k-heads (norm kw + rope),
// [nq+nkv, nq+2nkv) v-heads (weightless norm). In place in C [T][rowc].
__kernel void qkv_post(__global float* C, int rowc, int nq, int nkv, int hd, __global const float* qw,
                       __global const float* kw, __global const float* cosb, __global const float* sinb,
                       __global const int* pos, int hlf, float eps) {
  __local float red[256];
  const int slot = get_group_id(0), t = get_group_id(1), lid = get_local_id(0), LS = get_local_size(0);
  __global float* v = C + (size_t)t * rowc + slot * hd;      // q, k, v heads are contiguous in that order
  float ss = 0.f;
  for (int i = lid; i < hd; i += LS) { float x = v[i]; ss = mad(x, x, ss); }
  const float inv = 1.0f / sqrt(wg_sum(ss, red) / (float)hd + eps);
  if (slot >= nq + nkv) { for (int i = lid; i < hd; i += LS) v[i] *= inv; return; }
  __global const float* w = slot < nq ? qw : kw;
  const int p = pos[t];
  for (int i = lid; i < hlf; i += LS) {
    const float x0 = v[i] * inv * w[i], x1 = v[i + hlf] * inv * w[i + hlf];
    const float c = cosb[(size_t)p * hlf + i], s = sinb[(size_t)p * hlf + i];
    v[i] = x0 * c - x1 * s;
    v[i + hlf] = x1 * c + x0 * s;
  }
  for (int i = 2 * hlf + lid; i < hd; i += LS) v[i] = v[i] * inv * w[i];   // unrotated tail (rot < hd)
}

// attention + int8 quantization of its output (input of the wo GEMM; cols = nq*hd).
__kernel void attention_q(__global const float* C, int ldc, int qoff, int koff, int voff,
                          int nq, int nkv, int hd,
                          __global const int* seg_lo, __global const int* seg_hi, int window, float scale,
                          __global uint* Xp, __global float* Xs) {
  __local float p[MAXS];
  __local float qv[512];
  __local float ov[512];
  __local float red[256];
  const int h = get_group_id(0), t = get_group_id(1);
  const int lid = get_local_id(0), LS = get_local_size(0);
  const int kvh = h / (nq / nkv);
  const int lo = seg_lo[t], hi = seg_hi[t], n = hi - lo + 1;
  const __global float* q = C + (size_t)t * ldc + qoff + h * hd;
  for (int i = lid; i < hd; i += LS) qv[i] = q[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = -1e30f;
  for (int k = lid; k < n; k += LS) {
    const int s = lo + k;
    float d;
    if (window > 0 && abs(t - s) >= window) d = -1e30f;
    else {
      const __global float* ks = C + (size_t)s * ldc + koff + kvh * hd;
      float4 a0 = (float4)(0.f), a1 = (float4)(0.f);
      for (int i = 0; i < hd; i += 8) {
        a0 = mad(vload4(0, qv + i), vload4(0, ks + i), a0);
        a1 = mad(vload4(1, qv + i), vload4(1, ks + i), a1);
      }
      a0 += a1;
      d = (a0.x + a0.y + a0.z + a0.w) * scale;
    }
    p[k] = d; mx = fmax(mx, d);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  red[lid] = mx; barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = LS >> 1; s > 0; s >>= 1) { if (lid < s) red[lid] = fmax(red[lid], red[lid + s]); barrier(CLK_LOCAL_MEM_FENCE); }
  mx = red[0];
  float sm = 0.f;
  for (int k = lid; k < n; k += LS) { float e = p[k] < -1e29f ? 0.f : exp(p[k] - mx); p[k] = e; sm += e; }
  const float tot = wg_sum(sm, red);
  const float inv = tot > 0.f ? 1.0f / tot : 0.f;
  barrier(CLK_LOCAL_MEM_FENCE);
  const __global float* vb = C + (size_t)lo * ldc + voff + kvh * hd;
  for (int i = lid; i < hd; i += LS) {
    float a = 0.f;
    for (int k = 0; k < n; ++k) a = mad(p[k], vb[(size_t)k * ldc + i], a);
    ov[i] = a * inv;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int cols = nq * hd;
  for (int b = lid; b < hd / 32; b += LS) quant_store32_l(ov + b * 32, Xp, Xs, t, (h * hd) / 32 + b, cols);
}


// attention v2: LS = hd/4 lanes, each lane owns one float4 of the head dim (coalesced K/V rows);
// per-key dot products reduced with subgroup ops + one SLM pass. Emits int8 for the wo GEMM.
#define KCH 128
__kernel void attention_q2(__global const float* C, int ldc, int qoff, int koff, int voff,
                           int nq, int nkv, int hd,
                           __global const int* seg_lo, __global const int* seg_hi, int window, float scale,
                           __global uint* Xp, __global float* Xs) {
  __local float p[MAXS];
  __local float sp[16 * KCH];
  __local float ov[512];
  __local float red[32];
  const int h = get_group_id(0), t = get_group_id(1);
  const int lid = get_local_id(0), LS = get_local_size(0);
  const int sgid = get_sub_group_id(), nsg = get_num_sub_groups(), sgl = get_sub_group_local_id();
  const int kvh = h / (nq / nkv);
  const int lo = seg_lo[t], hi = seg_hi[t], n = hi - lo + 1;
  const float4 q4 = vload4(lid, C + (size_t)t * ldc + qoff + h * hd);
  const __global float* kb = C + (size_t)lo * ldc + koff + kvh * hd;
  for (int k0 = 0; k0 < n; k0 += KCH) {
    const int kn = min(KCH, n - k0);
    for (int kk = 0; kk < kn; ++kk) {
      const float4 k4 = vload4(lid, kb + (size_t)(k0 + kk) * ldc);
      const float4 pr = q4 * k4;
      const float sg = sub_group_reduce_add(pr.x + pr.y + pr.z + pr.w);
      if (sgl == 0) sp[sgid * KCH + kk] = sg;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = lid; kk < kn; kk += LS) {
      float d = 0.f;
      for (int g = 0; g < nsg; ++g) d += sp[g * KCH + kk];
      const int s = lo + k0 + kk;
      p[k0 + kk] = (window > 0 && abs(t - s) >= window) ? -1e30f : d * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // softmax over p[0..n)
  float mx = -1e30f;
  for (int k = lid; k < n; k += LS) mx = fmax(mx, p[k]);
  mx = sub_group_reduce_max(mx);
  if (sgl == 0) red[sgid] = mx;
  barrier(CLK_LOCAL_MEM_FENCE);
  mx = red[0]; for (int g = 1; g < nsg; ++g) mx = fmax(mx, red[g]);
  barrier(CLK_LOCAL_MEM_FENCE);
  float sm = 0.f;
  for (int k = lid; k < n; k += LS) { float e = p[k] < -1e29f ? 0.f : exp(p[k] - mx); p[k] = e; sm += e; }
  sm = sub_group_reduce_add(sm);
  if (sgl == 0) red[sgid] = sm;
  barrier(CLK_LOCAL_MEM_FENCE);
  float tot = 0.f; for (int g = 0; g < nsg; ++g) tot += red[g];
  const float inv = tot > 0.f ? 1.0f / tot : 0.f;
  // O = P V   (lane = float4 of head dim)
  const __global float* vb = C + (size_t)lo * ldc + voff + kvh * hd;
  float4 a = (float4)(0.f);
  for (int k = 0; k < n; ++k) a = mad((float4)(p[k]), vload4(lid, vb + (size_t)k * ldc), a);
  vstore4(a * inv, lid, ov);
  barrier(CLK_LOCAL_MEM_FENCE);
  const int cols = nq * hd;
  for (int b = lid; b < hd / 32; b += LS) quant_store32_l(ov + b * 32, Xp, Xs, t, (h * hd) / 32 + b, cols);
}

// vectorized gelu*up -> int8: one WI per (block, token), vector loads
__kernel void gelu_mul_q2(__global const float* G, int FF, __global uint* Xp, __global float* Xs) {
  const int b = get_global_id(0), t = get_global_id(1);
  const __global float* g = G + (size_t)t * 2 * FF + b * 32;
  float v[32];
  for (int c = 0; c < 4; ++c) {
    float8 gg = vload8(c, g), uu = vload8(c, g + FF);
    float8 r = gelu8(gg) * uu;
    v[8*c+0] = r.s0; v[8*c+1] = r.s1; v[8*c+2] = r.s2; v[8*c+3] = r.s3;
    v[8*c+4] = r.s4; v[8*c+5] = r.s5; v[8*c+6] = r.s6; v[8*c+7] = r.s7;
  }
  quant_store32(v, Xp, Xs, t, b, FF);
}
__kernel void gelu_mulvec_q2(__global const float* I, __global const float* H, int hoff, int hld, int npl,
                             __global uint* Xp, __global float* Xs) {
  const int b = get_global_id(0), t = get_global_id(1);
  const __global float* x = I + (size_t)t * npl + b * 32;
  const __global float* hh = H + hoff + (size_t)t * hld + b * 32;
  float v[32];
  for (int c = 0; c < 4; ++c) {
    float8 xx = vload8(c, x), m = vload8(c, hh);
    float8 r = gelu8(xx) * m;
    v[8*c+0] = r.s0; v[8*c+1] = r.s1; v[8*c+2] = r.s2; v[8*c+3] = r.s3;
    v[8*c+4] = r.s4; v[8*c+5] = r.s5; v[8*c+6] = r.s6; v[8*c+7] = r.s7;
  }
  quant_store32(v, Xp, Xs, t, b, npl);
}


// ============================ v3: subgroup reductions, coalesced quantization, online-softmax attention ============================
inline float wg_add(float v, __local float* red) {
  const float s = sub_group_reduce_add(v);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float tot = 0.f; const int ng = get_num_sub_groups();
  for (int g = 0; g < ng; ++g) tot += red[g];
  return tot;
}
inline float wg_max(float v, __local float* red) {
  const float s = sub_group_reduce_max(v);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float m = -INFINITY; const int ng = get_num_sub_groups();
  for (int g = 0; g < ng; ++g) m = fmax(m, red[g]);
  return m;
}
// lane owns float4 v (elements 4*lid..+3 of a row of `cols`); blocks of 32 = 8 lanes. Writes packed int8 + scale.
inline void quant_lane4(float4 v, __local float* am, __global uint* Xp, __global float* Xs, int t, int cols) {
  const int lid = get_local_id(0);
  const float4 a = fabs(v);
  barrier(CLK_LOCAL_MEM_FENCE);
  am[lid] = fmax(fmax(a.x, a.y), fmax(a.z, a.w));
  barrier(CLK_LOCAL_MEM_FENCE);
  const int b0 = lid & ~7;
  float amax = am[b0];
  for (int i = 1; i < 8; ++i) amax = fmax(amax, am[b0 + i]);
  const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
  Xp[(size_t)t * (cols / 4) + lid] = as_uint(convert_char4_sat_rte(v * id));
  if ((lid & 7) == 0) Xs[(size_t)t * (cols / 32) + (lid >> 3)] = d;
}

// row_fused v3: one WG per token, LS = cols/4 lanes (lane = float4)
__kernel void row_fused3(__global const float* R, int has_r, __global const float* D, __global const float* wd, int has_d,
                         float scale, __global float* OUT, int has_out,
                         __global const float* wq, int norm_q, int do_q,
                         __global uint* Xp, __global float* Xs, int cols, float eps) {
  __local float red[32];
  __local float am[512];
  const int t = get_group_id(0), lid = get_local_id(0);
  float4 y = has_r ? vload4(lid, R + (size_t)t * cols) : (float4)(0.f);
  if (has_d) {
    const float4 d = vload4(lid, D + (size_t)t * cols);
    const float invd = 1.0f / sqrt(wg_add(dot(d, d), red) / (float)cols + eps);
    y += d * invd * vload4(lid, wd);
  }
  y *= scale;
  if (has_out) vstore4(y, lid, OUT + (size_t)t * cols);
  if (!do_q) return;
  if (norm_q) {
    const float invq = 1.0f / sqrt(wg_add(dot(y, y), red) / (float)cols + eps);
    y = y * invq * vload4(lid, wq);
  }
  quant_lane4(y, am, Xp, Xs, t, cols);
}

// qkv_post v3: WG per (slot, token), LS = hd/8; lane owns float4 at 4l (first half) and hlf+4l (second half). rot == hd.
__kernel void qkv_post3(__global float* C, int rowc, int nq, int nkv, int hd, __global const float* qw,
                        __global const float* kw, __global const float* cosb, __global const float* sinb,
                        __global const int* pos, int hlf, float eps) {
  __local float red[32];
  const int slot = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  __global float* v = C + (size_t)t * rowc + slot * hd;
  float4 a = vload4(lid, v), b = vload4(lid, v + hlf);
  const float inv = 1.0f / sqrt(wg_add(dot(a, a) + dot(b, b), red) / (float)hd + eps);
  if (slot >= nq + nkv) { vstore4(a * inv, lid, v); vstore4(b * inv, lid, v + hlf); return; }
  __global const float* w = slot < nq ? qw : kw;
  a = a * inv * vload4(lid, w);
  b = b * inv * vload4(lid, w + hlf);
  const int p = pos[t];
  const float4 c = vload4(lid, cosb + (size_t)p * hlf), s = vload4(lid, sinb + (size_t)p * hlf);
  vstore4(a * c - b * s, lid, v);
  vstore4(b * c + a * s, lid, v + hlf);
}

// attention v3: online softmax over key chunks of KC3 (no max-segment limit, ~4KB SLM -> high occupancy).
// LS = hd/4 lanes, lane owns a float4 of the head dim.
#define KC3 32
__kernel void attention_q3(__global const float* C, int ldc, int qoff, int koff, int voff,
                           int nq, int nkv, int hd,
                           __global const int* seg_lo, __global const int* seg_hi, int window, float scale,
                           __global uint* Xp, __global float* Xs) {
  __local float sp[16 * KC3];
  __local float pk[KC3];
  __local float am[128];
  const int h = get_group_id(0), t = get_group_id(1);
  const int lid = get_local_id(0), LS = get_local_size(0);
  const int sgid = get_sub_group_id(), nsg = get_num_sub_groups(), sgl = get_sub_group_local_id();
  const int kvh = h / (nq / nkv);
  const int lo = seg_lo[t], hi = seg_hi[t], n = hi - lo + 1;
  const float4 q4 = vload4(lid, C + (size_t)t * ldc + qoff + h * hd);
  const __global float* kb = C + (size_t)lo * ldc + koff + kvh * hd;
  const __global float* vb = C + (size_t)lo * ldc + voff + kvh * hd;
  float m = -1e30f, l = 0.f;
  float4 acc = (float4)(0.f);
  for (int k0 = 0; k0 < n; k0 += KC3) {
    const int kn = min(KC3, n - k0);
    for (int kk = 0; kk < kn; ++kk) {
      const float4 pr = q4 * vload4(lid, kb + (size_t)(k0 + kk) * ldc);
      const float sg = sub_group_reduce_add(pr.x + pr.y + pr.z + pr.w);
      if (sgl == 0) sp[sgid * KC3 + kk] = sg;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk2 = lid; kk2 < kn; kk2 += LS) {   // LS may be < KC3 (head_dim 64 -> 16 lanes)
      float d = 0.f;
      for (int g = 0; g < nsg; ++g) d += sp[g * KC3 + kk2];
      const int s = lo + k0 + kk2;
      pk[kk2] = (window > 0 && abs(t - s) >= window) ? -1e30f : d * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float cm = m;
    for (int kk = 0; kk < kn; ++kk) cm = fmax(cm, pk[kk]);
    const float alpha = exp(m - cm);
    acc *= alpha; l *= alpha; m = cm;
    for (int kk = 0; kk < kn; ++kk) {
      const float sc = pk[kk];
      const float e = sc < -1e29f ? 0.f : exp(sc - m);
      l += e;
      acc = mad((float4)(e), vload4(lid, vb + (size_t)(k0 + kk) * ldc), acc);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float4 o = acc * (l > 0.f ? 1.0f / l : 0.f);
  // quantize: head h occupies cols [h*hd, (h+1)*hd) of the wo input; lane lid -> uint index h*hd/4 + lid
  const float4 a = fabs(o);
  am[lid] = fmax(fmax(a.x, a.y), fmax(a.z, a.w));
  barrier(CLK_LOCAL_MEM_FENCE);
  const int b0 = lid & ~7;
  float amax = am[b0];
  for (int i = 1; i < 8; ++i) amax = fmax(amax, am[b0 + i]);
  const float dq = amax / 127.f, id = dq > 0.f ? 1.f / dq : 0.f;
  const int cols = nq * hd;
  Xp[(size_t)t * (cols / 4) + h * (hd / 4) + lid] = as_uint(convert_char4_sat_rte(o * id));
  if ((lid & 7) == 0) Xs[(size_t)t * (cols / 32) + h * (hd / 32) + (lid >> 3)] = dq;
}

// gelu(gate)*up -> int8, WG per token, LS = FF/4 lanes (float4 each)
__kernel void gelu_mul_q3(__global const float* G, int FF, __global uint* Xp, __global float* Xs) {
  __local float am[1024];
  const int t = get_group_id(0), lid = get_local_id(0);
  const __global float* g = G + (size_t)t * 2 * FF;
  const float4 gg = vload4(lid, g), uu = vload4(lid, g + FF);
  const float8 r8 = gelu8((float8)(gg, (float4)(0.f)));
  quant_lane4(r8.lo * uu, am, Xp, Xs, t, FF);
}
// gelu(I)*H -> int8, WG per token, LS = npl/4
__kernel void gelu_mulvec_q3(__global const float* I, __global const float* H, int hoff, int hld, int npl,
                             __global uint* Xp, __global float* Xs) {
  __local float am[512];
  const int t = get_group_id(0), lid = get_local_id(0);
  const float4 x = vload4(lid, I + (size_t)t * npl);
  const float4 m = vload4(lid, H + hoff + (size_t)t * hld);
  const float8 r8 = gelu8((float8)(x, (float4)(0.f)));
  quant_lane4(r8.lo * m, am, Xp, Xs, t, npl);
}

// F = gelu(gate) * up, quantized per 32-block for the down GEMM. One WI per (block, token).
__kernel void gelu_mul_q(__global const float* G, int FF, __global uint* Xp, __global float* Xs) {
  const int b = get_global_id(0), t = get_global_id(1);
  const __global float* g = G + (size_t)t * 2 * FF + b * 32;
  float v[32];
  for (int i = 0; i < 32; ++i) v[i] = gelu1(g[i]) * g[FF + i];
  quant_store32(v, Xp, Xs, t, b, FF);
}

// I = gelu(I) * H[hoff + ...], quantized for the ple_proj GEMM. One WI per (block, token).
__kernel void gelu_mulvec_q(__global const float* I, __global const float* H, int hoff, int npl,
                            __global uint* Xp, __global float* Xs) {
  const int b = get_global_id(0), t = get_global_id(1);
  const __global float* x = I + (size_t)t * npl + b * 32;
  const __global float* h = H + hoff + (size_t)t * npl + b * 32;
  float v[32];
  for (int i = 0; i < 32; ++i) v[i] = gelu1(x[i]) * h[i];
  quant_store32(v, Xp, Xs, t, b, npl);
}

__kernel void add_scale(__global const float* a, __global const float* b, __global float* y, float s) {
  const int i = get_global_id(0);
  y[i] = (a[i] + b[i]) * s;
}

// ============================ LFM2 decision models (d1-omni) ============================
// lane owns float8 (elements 8*lid..+7); a 32-block = 4 lanes. Same packed layout as quant_x.
inline void quant_lane8(float8 v, __local float* am, __global uint* Xp, __global float* Xs, int t, int cols) {
  const int lid = get_local_id(0);
  const float8 a = fabs(v);
  const float4 m4 = fmax(a.lo, a.hi); const float2 m2 = fmax(m4.lo, m4.hi);
  barrier(CLK_LOCAL_MEM_FENCE);
  am[lid] = fmax(m2.x, m2.y);
  barrier(CLK_LOCAL_MEM_FENCE);
  const int b0 = lid & ~3;
  const float amax = fmax(fmax(am[b0], am[b0 + 1]), fmax(am[b0 + 2], am[b0 + 3]));
  const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
  __global uint* o = Xp + (size_t)t * (cols / 4) + lid * 2;
  o[0] = as_uint(convert_char4_sat_rte(v.lo * id));
  o[1] = as_uint(convert_char4_sat_rte(v.hi * id));
  if ((lid & 3) == 0) Xs[(size_t)t * (cols / 32) + (lid >> 2)] = d;
}
// X += D (+db); then quantize rmsnorm(X)*w  or  layernorm(X)*w+b. WG per token, LS = cols/4.
__kernel void resid_norm_q3(__global float* X, __global const float* D, int has_d, __global const float* db, int has_db,
                            __global const float* w, __global const float* b, int is_ln,
                            __global uint* Xp, __global float* Xs, int cols, float eps) {
  __local float red[32];
  __local float am[512];
  const int t = get_group_id(0), lid = get_local_id(0);
  __global float* xr = X + (size_t)t * cols;
  float4 y = vload4(lid, xr);
  if (has_d) {
    y += vload4(lid, D + (size_t)t * cols);
    if (has_db) y += vload4(lid, db);
    vstore4(y, lid, xr);
  }
  float4 z;
  if (is_ln) {
    const float mean = wg_add(y.x + y.y + y.z + y.w, red) / (float)cols;
    const float4 c = y - mean;
    const float var = wg_add(dot(c, c), red) / (float)cols;
    z = c * (1.0f / sqrt(var + eps)) * vload4(lid, w) + vload4(lid, b);
  } else {
    z = y * (1.0f / sqrt(wg_add(dot(y, y), red) / (float)cols + eps)) * vload4(lid, w);
  }
  quant_lane4(z, am, Xp, Xs, t, cols);
}
// trunk -> head: y = rms(X + D)*ow + te[type]; X = y; quantize layernorm(y)*lw+lb
__kernel void head_in_q3(__global float* X, __global const float* D, __global const float* ow, __global const float* te,
                         __global const int* type, __global const float* lw, __global const float* lb,
                         __global uint* Xp, __global float* Xs, int cols, float rms_eps, float ln_eps) {
  __local float red[32];
  __local float am[512];
  const int t = get_group_id(0), lid = get_local_id(0);
  __global float* xr = X + (size_t)t * cols;
  float4 y = vload4(lid, xr) + vload4(lid, D + (size_t)t * cols);
  y = y * (1.0f / sqrt(wg_add(dot(y, y), red) / (float)cols + rms_eps)) * vload4(lid, ow)
      + vload4(lid, te + (size_t)type[t] * cols);
  vstore4(y, lid, xr);
  const float mean = wg_add(y.x + y.y + y.z + y.w, red) / (float)cols;
  const float4 c = y - mean;
  const float var = wg_add(dot(c, c), red) / (float)cols;
  quant_lane4(c * (1.0f / sqrt(var + ln_eps)) * vload4(lid, lw) + vload4(lid, lb), am, Xp, Xs, t, cols);
}
// centred 3-tap short conv: C = [b | c | x] per token; z = c * (t0*bx[t-1] + t1*bx[t] + t2*bx[t+1]) -> int8
// prev[t] / next[t]: row of the left / right conv neighbour, -1 = none (sequence edge; the last media row of a
// prefix has no right neighbour, the first text row reads the last media row on its left)
__kernel void conv_gate_q3(__global const float* C, __global const int* prev, __global const int* next,
                           __global const float* taps, __global uint* Xp, __global float* Xs, int d) {
  __local float am[512];
  const int t = get_group_id(0), lid = get_local_id(0);
  const __global float* r = C + (size_t)t * 3 * d;
  float4 y = vload4(lid, r) * vload4(lid, r + 2 * d) * vload4(lid, taps + d);
  const int pv = prev[t], nx = next[t];
  if (pv >= 0) { const __global float* p = C + (size_t)pv * 3 * d; y += vload4(lid, p) * vload4(lid, p + 2 * d) * vload4(lid, taps); }
  if (nx >= 0) { const __global float* n = C + (size_t)nx * 3 * d; y += vload4(lid, n) * vload4(lid, n + 2 * d) * vload4(lid, taps + 2 * d); }
  quant_lane4(vload4(lid, r + d) * y, am, Xp, Xs, t, d);
}
// SwiGLU: silu(G[:, :FF]) * G[:, FF:] -> int8. WG per token, LS = FF/8.
__kernel void silu_mul_q8(__global const float* G, int FF, __global uint* Xp, __global float* Xs) {
  __local float am[1024];
  const int t = get_group_id(0), lid = get_local_id(0);
  const __global float* g = G + (size_t)t * 2 * FF;
  const float8 x = vload8(lid, g);
  quant_lane8(x / (1.0f + native_exp2(x * -1.4426950408889634f)) * vload8(lid, g + FF), am, Xp, Xs, t, FF);
}
// act(G + b) -> int8. act: 0 none, 1 relu, 2 gelu (tanh approximation), 3 gelu (erf). WG per token, LS = cols/8.
__kernel void bias_act_q8(__global const float* G, __global const float* b, int act, __global uint* Xp, __global float* Xs, int cols) {
  __local float am[1024];
  const int t = get_group_id(0), lid = get_local_id(0);
  float8 v = vload8(lid, G + (size_t)t * cols) + vload8(lid, b);
  if (act == 1) v = fmax(v, (float8)(0.f));
  else if (act == 2) v = 0.5f * v * (1.0f + tanh(0.7978845608028654f * v * (1.0f + 0.044715f * v * v)));
  else if (act == 3) v = 0.5f * v * (1.0f + erf(v * 0.70710678118654752f));
  quant_lane8(v, am, Xp, Xs, t, cols);
}
// plain quantization of f32 rows. WG per row, LS = cols/4.
__kernel void quant_q3(__global const float* X, __global uint* Xp, __global float* Xs, int cols) {
  __local float am[512];
  const int t = get_group_id(0), lid = get_local_id(0);
  quant_lane4(vload4(lid, X + (size_t)t * cols), am, Xp, Xs, t, cols);
}
// LayerNorm(X[src[v]])*w+b -> int8 row v (src < 0: zeros). With cols = a multiple of 32, k consecutive rows
// form one quantized row of k*cols: this is the vision 2x2 pixel unshuffle, fused into the post-LayerNorm.
__kernel void ln_gather_q3(__global const float* X, __global const int* src, __global const float* w, __global const float* b,
                           __global uint* Xp, __global float* Xs, int cols, float eps) {
  __local float red[32];
  __local float am[512];
  const int v = get_group_id(0), lid = get_local_id(0);
  const int sr = src[v];
  float4 z = (float4)(0.f);
  if (sr >= 0) {
    const float4 y = vload4(lid, X + (size_t)sr * cols);
    const float mean = wg_add(y.x + y.y + y.z + y.w, red) / (float)cols;
    const float4 c = y - mean;
    const float var = wg_add(dot(c, c), red) / (float)cols;
    z = c * (1.0f / sqrt(var + eps)) * vload4(lid, w) + vload4(lid, b);
  }
  quant_lane4(z, am, Xp, Xs, v, cols);
}
__kernel void bias_add(__global float* Y, __global const float* b, int rows) {
  const int i = get_global_id(0), t = get_global_id(1);
  Y[(size_t)t * rows + i] += b[i];
}
// attention v4 for small head dims (hd % 32 == 0, hd <= 128): WG per (head, token), 32 lanes.
// QK: lane = key of a 32-key chunk (full dot vs Q in SLM, no per-key reductions);
// softmax: each lane exps its own score once; PV: lane = output dims lid + 32j (coalesced V rows).
// Online softmax over chunks; output quantized for the wo GEMM like attention_q3.
#define KC4 32
__kernel void attention_q4(__global const float* C, int ldc, int qoff, int koff, int voff,
                           int nq, int nkv, int hd,
                           __global const int* seg_lo, __global const int* seg_hi, int window, float scale,
                           __global uint* Xp, __global float* Xs) {
  __local float qs[128];
  __local float pk[KC4];
  __local float os[128];
  __local float am[32];
  const int h = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  const int kvh = h / (nq / nkv), dpl = hd / 32;
  const int lo = seg_lo[t], hi = seg_hi[t], n = hi - lo + 1;
  for (int i = lid; i < hd; i += 32) qs[i] = C[(size_t)t * ldc + qoff + h * hd + i] * scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  const __global float* kb = C + (size_t)lo * ldc + koff + kvh * hd;
  const __global float* vb = C + (size_t)lo * ldc + voff + kvh * hd;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  float m = -1e30f, l = 0.f;
  for (int k0 = 0; k0 < n; k0 += KC4) {
    const int kn = min(KC4, n - k0);
    float sc = -1e30f;
    if (lid < kn) {
      const __global float* kr = kb + (size_t)(k0 + lid) * ldc;
      float4 s4 = (float4)(0.f);
      for (int i = 0; i < hd / 4; ++i) s4 = mad(vload4(i, kr), vload4(i, qs), s4);
      const int sidx = lo + k0 + lid;
      sc = (window > 0 && abs(t - sidx) >= window) ? -1e30f : (s4.x + s4.y + s4.z + s4.w);
    }
    pk[lid] = sc;
    barrier(CLK_LOCAL_MEM_FENCE);
    float cm = m;
    for (int kk = 0; kk < kn; ++kk) cm = fmax(cm, pk[kk]);
    const float e = (lid < kn && sc > -1e29f) ? exp(sc - cm) : 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);
    pk[lid] = e;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float alpha = exp(m - cm);
    m = cm;
    float csum = 0.f;
    acc0 *= alpha; acc1 *= alpha; acc2 *= alpha; acc3 *= alpha;
    for (int kk = 0; kk < kn; ++kk) {
      const float p = pk[kk];
      csum += p;
      const __global float* vr = vb + (size_t)(k0 + kk) * ldc + lid;
      acc0 = mad(p, vr[0], acc0);
      if (dpl > 1) acc1 = mad(p, vr[32], acc1);
      if (dpl > 2) acc2 = mad(p, vr[64], acc2);
      if (dpl > 3) acc3 = mad(p, vr[96], acc3);
    }
    l = l * alpha + csum;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = l > 0.f ? 1.0f / l : 0.f;
  os[lid] = acc0 * inv;
  if (dpl > 1) os[lid + 32] = acc1 * inv;
  if (dpl > 2) os[lid + 64] = acc2 * inv;
  if (dpl > 3) os[lid + 96] = acc3 * inv;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int nl = hd / 4;                   // lanes that quantize one float4 each (8 lanes per 32-block)
  float4 o = (float4)(0.f);
  if (lid < nl) { o = vload4(lid, os); const float4 a = fabs(o); am[lid] = fmax(fmax(a.x, a.y), fmax(a.z, a.w)); }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < nl) {
    const int b0 = lid & ~7;
    float amax = am[b0];
    for (int i = 1; i < 8; ++i) amax = fmax(amax, am[b0 + i]);
    const float dq = amax / 127.f, id = dq > 0.f ? 1.f / dq : 0.f;
    const int cols = nq * hd;
    Xp[(size_t)t * (cols / 4) + h * (hd / 4) + lid] = as_uint(convert_char4_sat_rte(o * id));
    if ((lid & 7) == 0) Xs[(size_t)t * (cols / 32) + h * (hd / 32) + (lid >> 3)] = dq;
  }
}

// attention v5 (hd == 64): query-tiled flash attention. WG = (head, QT5 consecutive queries), one query per
// lane; each K/V chunk of KT5 keys is staged in SLM once for the whole tile (v4 re-read every key row per
// query: ~20 GB/layer at 2.2k tokens). Softmax stays inside the lane (no cross-lane reductions). Queries of a
// tile may span segments: the tile walks the union of their key ranges and each lane masks foreign keys.
// t_base: first query of this launch (long batches are split into several launches).
#define QT5 32
#define KT5 16
// pre_lo/pre_hi (has_pre): a second key range per query, the media prefix the text of a question reads
// (empty when pre_lo > pre_hi). Each lane reads its prefix keys first, then its own range, ascending.
__kernel void attention_q5(__global const float* C, int ldc, int qoff, int koff, int voff,
                           int nq, int nkv, __global const int* seg_lo, __global const int* seg_hi,
                           int window, float scale, __global uint* Xp, __global float* Xs, int T, int t_base,
                           __global const int* pre_lo, __global const int* pre_hi, int has_pre) {
  __local float Qs[QT5 * 68];
  __local float Ks[KT5 * 64];
  __local float Vs[KT5 * 64];
  __local int rmin[QT5], rmax[QT5];
  const int h = get_group_id(0), lid = get_local_id(0);
  const int t0 = t_base + get_group_id(1) * QT5;
  const int tq = t0 + lid, t = min(tq, T - 1);
  const int kvh = h / (nq / nkv);
  int lo = seg_lo[t], hi = seg_hi[t];
  const int olo = lo, ohi = hi;
  int plo = 1, phi = 0;
  if (has_pre) { plo = pre_lo[t]; phi = pre_hi[t]; }
  rmin[lid] = plo <= phi ? plo : 0x7fffffff;
  rmax[lid] = plo <= phi ? phi : -1;
  barrier(CLK_LOCAL_MEM_FENCE);
  int ptlo = 0x7fffffff, pthi = -1;
  for (int i = 0; i < QT5; ++i) { ptlo = min(ptlo, rmin[i]); pthi = max(pthi, rmax[i]); }
  const int otlo = seg_lo[t0], othi = seg_hi[min(t0 + QT5 - 1, T - 1)];
  const __global float* qr = C + (size_t)t * ldc + qoff + h * 64;
  for (int i = 0; i < 16; ++i) vstore4(vload4(i, qr) * (scale * 1.4426950408889634f), i, Qs + lid * 68);
  float4 acc[16];
  for (int i = 0; i < 16; ++i) acc[i] = (float4)(0.f);
  float m = -1e30f, l = 0.f;
  for (int pass = has_pre ? 0 : 1; pass < 2; ++pass) {
  const int tlo = pass == 0 ? ptlo : otlo, thi = pass == 0 ? pthi : othi;
  lo = pass == 0 ? plo : olo; hi = pass == 0 ? phi : ohi;
  for (int k0 = tlo; k0 <= thi; k0 += KT5) {
    const int kn = min(KT5, thi - k0 + 1);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < kn * 16; i += QT5) {
      const int r = i >> 4, c4 = i & 15;
      const __global float* row = C + (size_t)(k0 + r) * ldc + kvh * 64;
      vstore4(vload4(c4, row + koff), c4, Ks + r * 64);
      vstore4(vload4(c4, row + voff), c4, Vs + r * 64);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float sc[KT5];
    for (int j = 0; j < kn; ++j) {
      float4 s4 = (float4)(0.f);
      for (int i = 0; i < 16; ++i) s4 = mad(vload4(i, Qs + lid * 68), vload4(i, Ks + j * 64), s4);
      const int key = k0 + j;
      const bool vis = key >= lo && key <= hi && !(window > 0 && abs(t - key) >= window);
      sc[j] = vis ? (s4.x + s4.y + s4.z + s4.w) : -1e30f;
    }
    // per-key online softmax (rescale only when the running max grows): a lane's arithmetic depends only on its
    // own keys in order, never on chunk boundaries, so results are identical however the batch is packed
    for (int j = 0; j < kn; ++j) {
      const float sj = sc[j];
      if (sj < -1e29f) continue;
      if (sj > m) {
        const float a = native_exp2(m - sj);
        for (int i = 0; i < 16; ++i) acc[i] *= a;
        l *= a; m = sj;
      }
      const float p = native_exp2(sj - m);
      l += p;
      for (int i = 0; i < 16; ++i) acc[i] = mad((float4)(p), vload4(i, Vs + j * 64), acc[i]);
    }
  }
  }
  if (tq >= T) return;
  const float inv = l > 0.f ? 1.0f / l : 0.f;
  const int cols = nq * 64;
  for (int b = 0; b < 2; ++b) {                    // two 32-dim quant blocks, entirely inside the lane
    float amax = 0.f;
    for (int i = 0; i < 8; ++i) { const float4 a = fabs(acc[b * 8 + i] * inv); amax = fmax(amax, fmax(fmax(a.x, a.y), fmax(a.z, a.w))); }
    const float dq = amax / 127.f, id = dq > 0.f ? 1.f / dq : 0.f;
    for (int i = 0; i < 8; ++i)
      Xp[(size_t)tq * (cols / 4) + h * 16 + b * 8 + i] = as_uint(convert_char4_sat_rte(acc[b * 8 + i] * inv * id));
    Xs[(size_t)tq * (cols / 32) + h * 2 + b] = dq;
  }
}

// ---- dense attention for long fully-visible segments (vision tiles). Scores are materialized:
// S[h][i][j] = q_i . k_j (one segment of n rows starting at row r0), softmax rows in place, O = P V.
// 64x64 output tile per WG, 16x16 lanes, 4x4 per lane; the inner dimension is staged in SLM.
__kernel void dattn_qk(__global const float* C, int ldc, int qoff, int koff, int r0, int n, __global float* S) {
  // 64x64 score tile, 16x16 lanes; lane (tx,ty) owns rows 4ty..4ty+3 x cols 4tx..4tx+3 (float4 SLM reads: 2 loads / 16 FMA).
  // Operands are staged transposed ([k][row], pitch 64) with consecutive lanes on consecutive rows: no bank conflicts.
  __local float As[64 * 64];
  __local float Bs[64 * 64];
  const int h = get_group_id(2), tx = get_local_id(0), ty = get_local_id(1);
  const int i0 = get_group_id(1) * 64, j0 = get_group_id(0) * 64, lid = ty * 16 + tx;
  {
    const int r = lid & 63, g = lid >> 6;
    const bool ra = i0 + r < n, rb = j0 + r < n;
    const __global float* qa = C + (size_t)(r0 + i0 + r) * ldc + qoff + h * 64;
    const __global float* kb = C + (size_t)(r0 + j0 + r) * ldc + koff + h * 64;
    for (int c4 = g; c4 < 16; c4 += 4) {
      const float4 a = ra ? vload4(c4, qa) : (float4)(0.f);
      const float4 b = rb ? vload4(c4, kb) : (float4)(0.f);
      As[(c4 * 4 + 0) * 64 + r] = a.x; As[(c4 * 4 + 1) * 64 + r] = a.y; As[(c4 * 4 + 2) * 64 + r] = a.z; As[(c4 * 4 + 3) * 64 + r] = a.w;
      Bs[(c4 * 4 + 0) * 64 + r] = b.x; Bs[(c4 * 4 + 1) * 64 + r] = b.y; Bs[(c4 * 4 + 2) * 64 + r] = b.z; Bs[(c4 * 4 + 3) * 64 + r] = b.w;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 acc0 = (float4)(0.f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
  for (int k = 0; k < 64; ++k) {
    const float4 qa = vload4(0, As + k * 64 + ty * 4);
    const float4 kb = vload4(0, Bs + k * 64 + tx * 4);
    acc0 = mad((float4)(qa.x), kb, acc0);
    acc1 = mad((float4)(qa.y), kb, acc1);
    acc2 = mad((float4)(qa.z), kb, acc2);
    acc3 = mad((float4)(qa.w), kb, acc3);
  }
  __global float* Sh = S + (size_t)h * n * n;
  const int jj = j0 + tx * 4;
#define DQK_ROW(A, ACC) { const int i = i0 + ty * 4 + (A); if (i < n) { \
    if (jj + 3 < n && (n & 3) == 0) vstore4(ACC, 0, Sh + (size_t)i * n + jj); \
    else { if (jj < n) Sh[(size_t)i * n + jj] = ACC.x; if (jj + 1 < n) Sh[(size_t)i * n + jj + 1] = ACC.y; \
           if (jj + 2 < n) Sh[(size_t)i * n + jj + 2] = ACC.z; if (jj + 3 < n) Sh[(size_t)i * n + jj + 3] = ACC.w; } } }
  DQK_ROW(0, acc0) DQK_ROW(1, acc1) DQK_ROW(2, acc2) DQK_ROW(3, acc3)
#undef DQK_ROW
}
// per-row max of S (the softmax shift): st[h*n + i] = max_j S[h][i][j]. WG per (row, head), LS = 256.
// P is never materialized: dattn_pv applies exp() while staging and normalizes by the row sum at the end.
__kernel void dattn_stats(__global const float* S, int n, float scale, __global float2* st) {
  __local float red[32];
  const int i = get_group_id(0), h = get_group_id(1), lid = get_local_id(0);
  const __global float* row = S + ((size_t)h * n + i) * n;
  float m = -1e30f;
  for (int j = lid * 4; j + 3 < n; j += 1024) { const float4 v = vload4(0, row + j); m = fmax(m, fmax(fmax(v.x, v.y), fmax(v.z, v.w))); }
  for (int j = (n & ~3) + lid; j < n; j += 256) m = fmax(m, row[j]);     // tail when n % 4 != 0
  m = wg_max(m, red);
  if (lid == 0) st[(size_t)h * n + i] = (float2)(m, 0.f);
}
// O[r0 + i][h*64 + c] = sum_j P[h][i][j] * V[r0 + j][h*64 + c]. 64 rows x 64 dims per WG, keys in chunks of 32,
// same 4x4 float4 register tiling as dattn_qk (P is staged transposed [k][row]).
__kernel void dattn_pv(__global const float* S, __global const float2* st, float scale, __global const float* C, int ldc,
                       int voff, int r0, int n, __global float* O, int ldo) {
  __local float Ps[32 * 64];
  __local float Vs[32 * 64];
  const int h = get_group_id(1), tx = get_local_id(0), ty = get_local_id(1);
  const int i0 = get_group_id(0) * 64, lid = ty * 16 + tx;
  const __global float* Sh = S + (size_t)h * n * n;
  float4 acc0 = (float4)(0.f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
  const int r = lid & 63, g = lid >> 6;
  const bool rok = i0 + r < n;
  const float L2E = 1.4426950408889634f;
  const float sl2 = scale * L2E, ml2 = rok ? st[(size_t)h * n + i0 + r].x * sl2 : 0.f;
  float psum = 0.f;
  __local float tot[256];
  for (int k0 = 0; k0 < n; k0 += 32) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int c4 = g; c4 < 8; c4 += 4) {           // P: row r, keys k0 + 4*c4 .. +3 (a ragged tail is handled below)
      const int kk = k0 + c4 * 4;
      float4 p = (float4)(0.f);
      if (rok && kk + 3 < n) { p = native_exp2(vload4(0, Sh + (size_t)(i0 + r) * n + kk) * sl2 - ml2); psum += p.x + p.y + p.z + p.w; }
      else if (rok && kk < n) {                      // ragged tail (n % 4 != 0): in-range keys as scalars, the rest stay 0
        const __global float* sr = Sh + (size_t)(i0 + r) * n + kk;
        p.x = native_exp2(sr[0] * sl2 - ml2);
        if (kk + 1 < n) p.y = native_exp2(sr[1] * sl2 - ml2);
        if (kk + 2 < n) p.z = native_exp2(sr[2] * sl2 - ml2);
        psum += p.x + p.y + p.z + p.w;
      }
      Ps[(c4 * 4 + 0) * 64 + r] = p.x; Ps[(c4 * 4 + 1) * 64 + r] = p.y; Ps[(c4 * 4 + 2) * 64 + r] = p.z; Ps[(c4 * 4 + 3) * 64 + r] = p.w;
    }
    for (int e = lid; e < 32 * 16; e += 256) {     // V: 32 key rows x 64 dims
      const int vr = e >> 4, vc = e & 15;
      vstore4(k0 + vr < n ? vload4(vc, C + (size_t)(r0 + k0 + vr) * ldc + voff + h * 64) : (float4)(0.f), vc, Vs + vr * 64);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < 32; ++k) {
      const float4 pa = vload4(0, Ps + k * 64 + ty * 4);
      const float4 vb = vload4(0, Vs + k * 64 + tx * 4);
      acc0 = mad((float4)(pa.x), vb, acc0);
      acc1 = mad((float4)(pa.y), vb, acc1);
      acc2 = mad((float4)(pa.z), vb, acc2);
      acc3 = mad((float4)(pa.w), vb, acc3);
    }
  }
  tot[g * 64 + r] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
#define DPV_ROW(A, ACC) { const int rr = ty * 4 + (A), i = i0 + rr; \
    const float inv = 1.0f / (tot[rr] + tot[64 + rr] + tot[128 + rr] + tot[192 + rr]); \
    if (i < n) vstore4(ACC * inv, 0, O + (size_t)(r0 + i) * ldo + h * 64 + tx * 4); }
  DPV_ROW(0, acc0) DPV_ROW(1, acc1) DPV_ROW(2, acc2) DPV_ROW(3, acc3)
#undef DPV_ROW
}

// decision scorer tail at the marker rows: out[k] = gelu_erf(O[m] + cb) . w + wb. WG per marker, LS = d/4.
__kernel void score_out(__global const float* O, __global const float* cb, __global const float* w, float wb,
                        __global const int* marks, __global float* out, int d) {
  __local float red[32];
  const int k = get_group_id(0), lid = get_local_id(0);
  const float4 v = vload4(lid, O + (size_t)marks[k] * d) + vload4(lid, cb);
  const float4 g = 0.5f * v * (1.0f + erf(v * 0.70710678118654752f));
  const float s = wg_add(dot(g, vload4(lid, w)), red);
  if (lid == 0) out[k] = s + wb;
}

// Gemma 4 vision qkv post-process: WG per (slot, token), LS = 64 (hd = 64 only, one lane per head dim).
// q/k: RMSNorm(weight) then 2-D neox RoPE (x position on dims [0,hd/2), y position on dims [hd/2,hd), each half a
// rotate-half block of hn = hd/4 pairs, angle = pos * theta^(-i/hn)); v: RMSNorm without weight; no RoPE.
__kernel void qkv_post_g4(__global float* C, int rowc, int nq, int nkv, int hd, __global const float* qw,
                          __global const float* kw, __global const float* invf, __global const int* posx,
                          __global const int* posy, float eps) {
  __local float red[32];
  __local float buf[64];
  const int slot = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  __global float* v = C + (size_t)t * rowc + slot * hd;
  const float x = v[lid];
  const float inv = 1.0f / sqrt(wg_add(x * x, red) / (float)hd + eps);
  if (slot >= nq + nkv) { v[lid] = x * inv; return; }
  const float y = x * inv * (slot < nq ? qw : kw)[lid];
  buf[lid] = y;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int hn = hd / 4;                       // 16 pairs per block
  const int blk = lid / (2 * hn);              // 0: x block, 1: y block
  const int w = lid - blk * 2 * hn;            // index within the block [0, 2*hn)
  const int i = w < hn ? w : w - hn;           // pair index
  const float p = (float)(blk == 0 ? posx[t] : posy[t]);
  const float a = p * invf[i], c = cos(a), sn = sin(a);
  const float x0 = buf[blk * 2 * hn + i], x1 = buf[blk * 2 * hn + i + hn];
  v[lid] = w < hn ? x0 * c - x1 * sn : x0 * sn + x1 * c;
}


// dattn_stats with a band mask: only keys with |i - j| < window (window <= 0: all keys) take part in the row max
__kernel void dattn_stats_w(__global const float* S, int n, float scale, __global float2* st, int window, int q0, int nr) {
  __local float red[32];
  const int i = get_group_id(0), h = get_group_id(1), lid = get_local_id(0);
  const __global float* row = S + ((size_t)h * nr + i) * n;
  const int gi = q0 + i;
  const int jlo = window > 0 ? max(0, gi - window + 1) : 0, jhi = window > 0 ? min(n - 1, gi + window - 1) : n - 1;
  float m = -1e30f;
  for (int j = jlo + lid; j <= jhi; j += 256) m = fmax(m, row[j]);
  m = wg_max(m, red);
  if (lid == 0) st[(size_t)h * nr + i] = (float2)(m, 0.f);
}

// ---- head-dim generic dense attention (hd multiple of 64), grouped-query: head h reads kv head h / (nq/nkv).
// QK: identical tile layout to dattn_qk; the 64-wide dot is repeated over hd/64 dim chunks, accumulating in registers.
__kernel void dattn_qk_h(__global const float* C, int ldc, int qoff, int koff, int n, int hd, int grp, __global float* S, int q0, int nr) {
  __local float As[64 * 64];
  __local float Bs[64 * 64];
  const int h = get_group_id(2), tx = get_local_id(0), ty = get_local_id(1);
  const int kvh = h / grp;
  const int i0 = get_group_id(1) * 64, j0 = get_group_id(0) * 64, lid = ty * 16 + tx;
  float4 acc0 = (float4)(0.f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
  for (int d0 = 0; d0 < hd; d0 += 64) {
    barrier(CLK_LOCAL_MEM_FENCE);
    {
      const int r = lid & 63, g = lid >> 6;
      const bool ra = i0 + r < nr, rb = j0 + r < n;
      const __global float* qa = C + (size_t)(q0 + i0 + r) * ldc + qoff + h * hd + d0;
      const __global float* kb = C + (size_t)(j0 + r) * ldc + koff + kvh * hd + d0;
      for (int c4 = g; c4 < 16; c4 += 4) {
        const float4 a = ra ? vload4(c4, qa) : (float4)(0.f);
        const float4 b = rb ? vload4(c4, kb) : (float4)(0.f);
        As[(c4 * 4 + 0) * 64 + r] = a.x; As[(c4 * 4 + 1) * 64 + r] = a.y; As[(c4 * 4 + 2) * 64 + r] = a.z; As[(c4 * 4 + 3) * 64 + r] = a.w;
        Bs[(c4 * 4 + 0) * 64 + r] = b.x; Bs[(c4 * 4 + 1) * 64 + r] = b.y; Bs[(c4 * 4 + 2) * 64 + r] = b.z; Bs[(c4 * 4 + 3) * 64 + r] = b.w;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < 64; ++k) {
      const float4 qa = vload4(0, As + k * 64 + ty * 4);
      const float4 kb = vload4(0, Bs + k * 64 + tx * 4);
      acc0 = mad((float4)(qa.x), kb, acc0);
      acc1 = mad((float4)(qa.y), kb, acc1);
      acc2 = mad((float4)(qa.z), kb, acc2);
      acc3 = mad((float4)(qa.w), kb, acc3);
    }
  }
  __global float* Sh = S + (size_t)h * nr * n;
  const int jj = j0 + tx * 4;
#define DQKH_ROW(A, ACC) { const int i = i0 + ty * 4 + (A); if (i < nr) { \
    if (jj + 3 < n && (n & 3) == 0) vstore4(ACC, 0, Sh + (size_t)i * n + jj); \
    else { if (jj < n) Sh[(size_t)i * n + jj] = ACC.x; if (jj + 1 < n) Sh[(size_t)i * n + jj + 1] = ACC.y; \
           if (jj + 2 < n) Sh[(size_t)i * n + jj + 2] = ACC.z; if (jj + 3 < n) Sh[(size_t)i * n + jj + 3] = ACC.w; } } }
  DQKH_ROW(0, acc0) DQKH_ROW(1, acc1) DQKH_ROW(2, acc2) DQKH_ROW(3, acc3)
#undef DQKH_ROW
}
// PV: O[i][h*hd + c0 + col] = sum_j P[h][i][j] V[j][kvh*hd + c0 + col]; one WG per (64 rows, head, 64-wide output column block).
__kernel void dattn_pv_h(__global const float* S, __global const float2* st, float scale, __global const float* C, int ldc,
                         int voff, int n, int hd, int grp, __global float* O, int ldo, int window, int q0, int nr) {
  __local float Ps[32 * 64];
  __local float Vs[32 * 64];
  const int hb = get_group_id(1), tx = get_local_id(0), ty = get_local_id(1);
  const int nblk = hd / 64, h = hb / nblk, cb = hb % nblk, kvh = h / grp;
  const int i0 = get_group_id(0) * 64, lid = ty * 16 + tx;
  const __global float* Sh = S + (size_t)h * nr * n;
  float4 acc0 = (float4)(0.f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
  const int r = lid & 63, g = lid >> 6;
  const bool rok = i0 + r < nr;
  const float L2E = 1.4426950408889634f;
  const float sl2 = scale * L2E, ml2 = rok ? st[(size_t)h * nr + i0 + r].x * sl2 : 0.f;
  float psum = 0.f;
  __local float tot[256];
  for (int k0 = 0; k0 < n; k0 += 32) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int c4 = g; c4 < 8; c4 += 4) {
      const int kk = k0 + c4 * 4;
      float4 p = (float4)(0.f);
      if (rok && kk + 3 < n) { p = native_exp2(vload4(0, Sh + (size_t)(i0 + r) * n + kk) * sl2 - ml2); }
      else if (rok && kk < n) {
        const __global float* sr = Sh + (size_t)(i0 + r) * n + kk;
        p.x = native_exp2(sr[0] * sl2 - ml2);
        if (kk + 1 < n) p.y = native_exp2(sr[1] * sl2 - ml2);
        if (kk + 2 < n) p.z = native_exp2(sr[2] * sl2 - ml2);
      }
      if (window > 0 && rok) {                        // band mask: keys with |i - j| >= window contribute nothing
        const int gi = q0 + i0 + r;
        if (abs(gi - (kk + 0)) >= window) p.x = 0.f;
        if (abs(gi - (kk + 1)) >= window) p.y = 0.f;
        if (abs(gi - (kk + 2)) >= window) p.z = 0.f;
        if (abs(gi - (kk + 3)) >= window) p.w = 0.f;
      }
      psum += p.x + p.y + p.z + p.w;                 // every column block accumulates the same row sums (cheap adds on p)
      Ps[(c4 * 4 + 0) * 64 + r] = p.x; Ps[(c4 * 4 + 1) * 64 + r] = p.y; Ps[(c4 * 4 + 2) * 64 + r] = p.z; Ps[(c4 * 4 + 3) * 64 + r] = p.w;
    }
    for (int e = lid; e < 32 * 16; e += 256) {
      const int vr = e >> 4, vc = e & 15;
      vstore4(k0 + vr < n ? vload4(vc, C + (size_t)(k0 + vr) * ldc + voff + kvh * hd + cb * 64) : (float4)(0.f), vc, Vs + vr * 64);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < 32; ++k) {
      const float4 pa = vload4(0, Ps + k * 64 + ty * 4);
      const float4 vb = vload4(0, Vs + k * 64 + tx * 4);
      acc0 = mad((float4)(pa.x), vb, acc0);
      acc1 = mad((float4)(pa.y), vb, acc1);
      acc2 = mad((float4)(pa.z), vb, acc2);
      acc3 = mad((float4)(pa.w), vb, acc3);
    }
  }
  tot[g * 64 + r] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
#define DPVH_ROW(A, ACC) { const int rr = ty * 4 + (A), i = i0 + rr; \
    const float inv = 1.0f / (tot[rr] + tot[64 + rr] + tot[128 + rr] + tot[192 + rr]); \
    if (i < nr) vstore4(ACC * inv, 0, O + (size_t)(q0 + i) * ldo + h * hd + cb * 64 + tx * 4); }
  DPVH_ROW(0, acc0) DPVH_ROW(1, acc1) DPVH_ROW(2, acc2) DPVH_ROW(3, acc3)
#undef DPVH_ROW
}
)CL";

// ---------------------------------------------------------------- state
struct Slot { cl_mem wp = nullptr, ws = nullptr; int rows = 0, cols = 0, nb = 0; size_t lws = 0; int kb = FERRO_GPU_KB; };
static bool g_ok = false;
static std::string g_name;
static cl_context g_ctx = nullptr;
static cl_command_queue g_q = nullptr;
static cl_program g_prog = nullptr;
// second build with KB=8 for weights whose input width is not a multiple of KB*32 (e.g. the 768-wide vision tower)
static cl_program g_prog8 = nullptr;
static cl_kernel k_gemm8 = nullptr, k_gemm_sw8 = nullptr;
static cl_device_id g_dev = nullptr;
static bool ensure_kb8();
static cl_kernel k_quant = nullptr, k_gemm = nullptr, k_gemm_sw = nullptr, k_rms = nullptr, k_rope = nullptr, k_attn = nullptr,
                 k_gmul = nullptr, k_gmulv = nullptr, k_add = nullptr,
                 k_rowf = nullptr, k_qkvp = nullptr, k_attq = nullptr, k_gmq = nullptr, k_gmvq = nullptr,
                 k_attq2 = nullptr, k_gmq2 = nullptr, k_gmvq2 = nullptr,
                 k_rowf3 = nullptr, k_qkvp3 = nullptr, k_attq3 = nullptr, k_gmq3 = nullptr, k_gmvq3 = nullptr,
                 k_rnq = nullptr, k_hinq = nullptr, k_convq = nullptr, k_siluq = nullptr, k_bactq = nullptr,
                 k_badd = nullptr, k_score = nullptr, k_attq4 = nullptr, k_attq5 = nullptr, k_quantq = nullptr, k_lngq = nullptr, k_dqk = nullptr, k_dsm = nullptr, k_dpv = nullptr, k_qkvg4 = nullptr, k_dqkh = nullptr, k_dpvh = nullptr, k_dsmw = nullptr;
static std::vector<Slot> g_slots;
static size_t g_bytes = 0;
static cl_mem g_Xp = nullptr, g_Xs = nullptr;           // quantized-activation scratch
static size_t g_Xpcap = 0, g_Xscap = 0;
static cl_mem g_hX = nullptr, g_hY = nullptr;           // legacy host-pointer GEMM staging
static size_t g_hXcap = 0, g_hYcap = 0;
static std::mutex g_mu;

static bool load_api(bool v) {
    const char* names[] = {"libOpenCL.so.1", "libOpenCL.so"};
    for (const char* n : names) if ((g_lib = dlopen(n, RTLD_NOW | RTLD_LOCAL))) break;
    if (!g_lib) { if (v) fprintf(stderr, "[gpu] libOpenCL not found -> CPU\n"); return false; }
#define LOAD(f) cl.f = (decltype(cl.f))dlsym(g_lib, #f); \
    if (!cl.f) { if (v) fprintf(stderr, "[gpu] missing symbol %s -> CPU\n", #f); return false; }
    CL_FUNCS(LOAD)
#undef LOAD
    return true;
}

#define CK(expr) do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
    if (verbose) fprintf(stderr, "[gpu] %s failed (%d) -> CPU\n", #expr, (int)_e); return false; } } while (0)

bool init(bool verbose) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_ok) return true;
    if (!load_api(verbose)) return false;
    cl_uint np = 0;
    if (cl.clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        if (verbose) fprintf(stderr, "[gpu] no OpenCL platforms -> CPU\n"); return false; }
    std::vector<cl_platform_id> plats(np);
    CK(cl.clGetPlatformIDs(np, plats.data(), nullptr));
    cl_device_id dev = nullptr;
    for (auto p : plats) {
        cl_uint nd = 0;
        if (cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, &nd) == CL_SUCCESS && nd > 0) {
            char ext[8192] = {0};
            cl.clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof ext - 1, ext, nullptr);
            if (strstr(ext, "cl_khr_integer_dot_product")) break;
            dev = nullptr;
        }
    }
    if (!dev) { if (verbose) fprintf(stderr, "[gpu] no GPU with cl_khr_integer_dot_product -> CPU\n"); return false; }
    char nm[256] = {0};
    cl.clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof nm - 1, nm, nullptr);
    g_name = nm;
    cl_int err = 0;
    g_ctx = cl.clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CK(err);
    g_q = cl.clCreateCommandQueueWithProperties(g_ctx, dev, nullptr, &err); CK(err);
    g_dev = dev;
    g_prog = cl.clCreateProgramWithSource(g_ctx, 1, &kSrc, nullptr, &err); CK(err);
    char opts[256];
    // no -cl-fast-relaxed-math: exp/erf/sqrt must stay accurate (softmax, gelu, rmsnorm)
    snprintf(opts, sizeof opts, "-cl-std=CL3.0 -DTT=%d -DKB=%d -DMAXS=%d -cl-mad-enable",
             FERRO_GPU_TT, FERRO_GPU_KB, ATTN_MAXS);
    if (cl.clBuildProgram(g_prog, 1, &dev, opts, nullptr, nullptr) != CL_SUCCESS) {
        if (verbose) {
            size_t n = 0; cl.clGetProgramBuildInfo(g_prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::string log(n, '\0'); cl.clGetProgramBuildInfo(g_prog, dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
            fprintf(stderr, "[gpu] kernel build failed -> CPU\n%s\n", log.c_str());
        }
        return false;
    }
    struct { cl_kernel* k; const char* n; } ks[] = {
        {&k_quant, "quant_x"}, {&k_gemm, "q8gemm"}, {&k_gemm_sw, "q8gemm_sw"}, {&k_rms, "rmsnorm"}, {&k_rope, "rope"},
        {&k_attn, "attention"}, {&k_gmul, "gelu_mul"}, {&k_gmulv, "gelu_mulvec"}, {&k_add, "add_scale"},
        {&k_rowf, "row_fused"}, {&k_qkvp, "qkv_post"}, {&k_attq, "attention_q"}, {&k_gmq, "gelu_mul_q"},
        {&k_gmvq, "gelu_mulvec_q"}, {&k_attq2, "attention_q2"}, {&k_gmq2, "gelu_mul_q2"}, {&k_gmvq2, "gelu_mulvec_q2"},
        {&k_qkvg4, "qkv_post_g4"}, {&k_dsmw, "dattn_stats_w"}, {&k_dqkh, "dattn_qk_h"}, {&k_dpvh, "dattn_pv_h"}, {&k_rowf3, "row_fused3"}, {&k_qkvp3, "qkv_post3"}, {&k_attq3, "attention_q3"}, {&k_gmq3, "gelu_mul_q3"},
        {&k_gmvq3, "gelu_mulvec_q3"}, {&k_rnq, "resid_norm_q3"}, {&k_hinq, "head_in_q3"}, {&k_convq, "conv_gate_q3"},
        {&k_siluq, "silu_mul_q8"}, {&k_bactq, "bias_act_q8"}, {&k_badd, "bias_add"}, {&k_score, "score_out"}, {&k_attq4, "attention_q4"}, {&k_attq5, "attention_q5"}, {&k_quantq, "quant_q3"}, {&k_lngq, "ln_gather_q3"}, {&k_dqk, "dattn_qk"}, {&k_dsm, "dattn_stats"}, {&k_dpv, "dattn_pv"}};
    for (auto& k : ks) { *k.k = cl.clCreateKernel(g_prog, k.n, &err); CK(err); }
    g_ok = true;
    if (verbose) fprintf(stderr, "[gpu] %s ready (int8 DP4A GEMM TT=%d KB=%d, on-device layer loop)\n",
                         g_name.c_str(), FERRO_GPU_TT, FERRO_GPU_KB);
    return true;
}
#undef CK

bool available() { return g_ok; }
const char* device_name() { return g_name.c_str(); }
size_t resident_bytes() { return g_bytes; }
int attn_max_seg() { return ATTN_MAXS; }

// ---------------------------------------------------------------- buffers
Buf alloc(size_t bytes) {
    cl_int e = 0;
    cl_mem m = cl.clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, bytes ? bytes : 4, nullptr, &e);
    return e == CL_SUCCESS ? (Buf)m : nullptr;
}
Buf upload(const void* src, size_t bytes) {
    cl_int e = 0;
    cl_mem m = cl.clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void*>(src), &e);
    if (e != CL_SUCCESS) return nullptr;
    g_bytes += bytes;
    return (Buf)m;
}
bool write(Buf b, const void* src, size_t bytes) {
    return cl.clEnqueueWriteBuffer(g_q, (cl_mem)b, CL_FALSE, 0, bytes, src, 0, nullptr, nullptr) == CL_SUCCESS;
}
bool read(Buf b, void* dst, size_t bytes) {
    return cl.clEnqueueReadBuffer(g_q, (cl_mem)b, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr) == CL_SUCCESS;
}
bool finish() { return cl.clFinish(g_q) == CL_SUCCESS; }
bool copy(Buf dst, size_t dst_off, Buf src, size_t src_off, size_t bytes) {
    if (!bytes) return true;
    return cl.clEnqueueCopyBuffer(g_q, (cl_mem)src, (cl_mem)dst, src_off, dst_off, bytes, 0, nullptr, nullptr) == CL_SUCCESS;
}
void release(Buf b) { if (b) cl.clReleaseMemObject((cl_mem)b); }

static bool ensure(cl_mem& m, size_t& cap, size_t need) {
    if (m && cap >= need) return true;
    if (m) cl.clReleaseMemObject(m);
    cl_int e = 0;
    m = cl.clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, need, nullptr, &e);
    cap = e == CL_SUCCESS ? need : 0;
    return e == CL_SUCCESS;
}

// ---------------------------------------------------------------- arg helper + launch
template <class T> static inline void sa(cl_kernel k, cl_uint i, const T& v) { cl.clSetKernelArg(k, i, sizeof(T), &v); }
static inline void sa(cl_kernel k, cl_uint i, Buf v) { cl_mem m = (cl_mem)v; cl.clSetKernelArg(k, i, sizeof(cl_mem), &m); }
template <class... A> static void set_args(cl_kernel k, A... a) { cl_uint i = 0; (sa(k, i++, a), ...); }
static bool launch(cl_kernel k, cl_uint dims, const size_t* g, const size_t* l) {
    return cl.clEnqueueNDRangeKernel(g_q, k, dims, nullptr, g, l, 0, nullptr, nullptr) == CL_SUCCESS;
}

// ---------------------------------------------------------------- weights
static bool ensure_kb8() {
    if (k_gemm8) return true;
    if (!g_ok || FERRO_GPU_KB == 8) return false;
    cl_int err = 0;
    g_prog8 = cl.clCreateProgramWithSource(g_ctx, 1, &kSrc, nullptr, &err);
    if (err != CL_SUCCESS) return false;
    char opts[256];
    snprintf(opts, sizeof opts, "-cl-std=CL3.0 -DTT=%d -DKB=8 -DMAXS=%d -cl-mad-enable", FERRO_GPU_TT, ATTN_MAXS);
    if (cl.clBuildProgram(g_prog8, 1, &g_dev, opts, nullptr, nullptr) != CL_SUCCESS) return false;
    k_gemm8 = cl.clCreateKernel(g_prog8, "q8gemm", &err);
    if (err != CL_SUCCESS) { k_gemm8 = nullptr; return false; }
    k_gemm_sw8 = cl.clCreateKernel(g_prog8, "q8gemm_sw", &err);
    if (err != CL_SUCCESS) { k_gemm8 = k_gemm_sw8 = nullptr; return false; }
    return true;
}

int upload_q8(const WMat& W) {
    if (!g_ok) return -1;
    const int rows = W.rows, cols = W.cols, nb = W.nb;
    if (cols % 32 || rows % 4 || W.wq.size() != (size_t)rows * cols) return -1;
    int kb = FERRO_GPU_KB;
    if (nb % FERRO_GPU_KB) {
        if (nb % 8 || !ensure_kb8()) return -1;
        kb = 8;
    }
    size_t lws = 0;
    for (size_t l : {128, 64, 32, 16}) if ((rows / 4) % l == 0) { lws = l; break; }
    if (!lws) return -1;
    std::vector<uint32_t> wp((size_t)nb * 8 * rows);
    std::vector<float> ws((size_t)nb * rows);
    for (int r = 0; r < rows; ++r) {
        const int8_t* src = W.wq.data() + (size_t)r * cols;
        for (int b = 0; b < nb; ++b) {
            for (int k = 0; k < 8; ++k) {
                uint32_t u; memcpy(&u, src + b * 32 + 4 * k, 4);
                wp[((size_t)b * 8 + k) * rows + r] = u;
            }
            ws[(size_t)b * rows + r] = W.ws[(size_t)r * nb + b];
        }
    }
    std::lock_guard<std::mutex> lk(g_mu);
    Slot s; s.rows = rows; s.cols = cols; s.nb = nb; s.lws = lws; s.kb = kb;
    s.wp = (cl_mem)upload(wp.data(), wp.size() * 4);
    s.ws = (cl_mem)upload(ws.data(), ws.size() * 4);
    if (!s.wp || !s.ws) { release(s.wp); release(s.ws); return -1; }
    g_slots.push_back(s);
    return (int)g_slots.size() - 1;
}

// ---------------------------------------------------------------- ops
bool op_gemm(int slot, Buf X, int T, Buf Y) {
    if (slot < 0 || slot >= (int)g_slots.size()) return false;
    const Slot& s = g_slots[slot];
    const int rows = s.rows, cols = s.cols, nb = s.nb;
    if (!ensure(g_Xp, g_Xpcap, (size_t)T * cols)) return false;
    if (!ensure(g_Xs, g_Xscap, (size_t)T * nb * 4)) return false;
    set_args(k_quant, X, (Buf)g_Xp, (Buf)g_Xs, (cl_int)cols);
    size_t gq[2] = {(size_t)nb, (size_t)T};
    if (!launch(k_quant, 2, gq, nullptr)) return false;
    cl_kernel kg = s.kb == FERRO_GPU_KB ? k_gemm : k_gemm8;
    set_args(kg, (Buf)s.wp, (Buf)s.ws, (Buf)g_Xp, (Buf)g_Xs, Y, (cl_int)rows, (cl_int)cols, (cl_int)T);
    size_t gg[2] = {(size_t)rows / 4, (size_t)((T + FERRO_GPU_TT - 1) / FERRO_GPU_TT)};
    size_t lg[2] = {s.lws, 1};
    return launch(kg, 2, gg, lg);
}

// swizzle when the weights overflow the LLC and there are enough token tiles to reuse each slice
static bool use_swizzle(const Slot& s, int T) {
    static const int mode = getenv("FERRO_GPU_SWIZZLE") ? atoi(getenv("FERRO_GPU_SWIZZLE")) : -1;   // -1 auto
    if (mode >= 0) return mode != 0;
    return (size_t)s.rows * s.cols >= ((size_t)1 << 20) && (T + FERRO_GPU_TT - 1) / FERRO_GPU_TT >= 8;
}

bool op_gemm_q(int slot, int T, Buf Y) {
    if (slot < 0 || slot >= (int)g_slots.size()) return false;
    const Slot& s = g_slots[slot];
    const bool k8 = s.kb != FERRO_GPU_KB;
    cl_kernel k = use_swizzle(s, T) ? (k8 ? k_gemm_sw8 : k_gemm_sw) : (k8 ? k_gemm8 : k_gemm);
    set_args(k, (Buf)s.wp, (Buf)s.ws, (Buf)g_Xp, (Buf)g_Xs, Y, (cl_int)s.rows, (cl_int)s.cols, (cl_int)T);
    size_t gg[2] = {(size_t)s.rows / 4, (size_t)((T + FERRO_GPU_TT - 1) / FERRO_GPU_TT)};
    size_t lg[2] = {s.lws, 1};
    return launch(k, 2, gg, lg);
}

bool ensure_qscratch(int T, int max_cols) {
    return ensure(g_Xp, g_Xpcap, (size_t)T * max_cols) && ensure(g_Xs, g_Xscap, (size_t)T * (max_cols / 32) * 4);
}

static bool use_v3() { static const bool v = getenv("FERRO_GPU_V2") == nullptr; return v; }

bool op_row_fused(Buf R, Buf D, Buf wd, float scale, Buf OUT, Buf wq, bool norm_q, bool do_q, int T, int cols, float eps) {
    if (cols > 2048 || cols % 32) return false;
    if (use_v3()) {
        const size_t LS3 = (size_t)cols / 4;
        set_args(k_rowf3, R ? R : D, (cl_int)(R != nullptr), D ? D : R, wd ? wd : (R ? R : D), (cl_int)(D != nullptr),
                 scale, OUT ? OUT : (R ? R : D), (cl_int)(OUT != nullptr), wq ? wq : (R ? R : D), (cl_int)norm_q,
                 (cl_int)do_q, (Buf)g_Xp, (Buf)g_Xs, (cl_int)cols, eps);
        size_t g3[1] = {(size_t)T * LS3}, l3[1] = {LS3};
        return launch(k_rowf3, 1, g3, l3);
    }
    const size_t LS = 128;
    set_args(k_rowf, R ? R : D, (cl_int)(R != nullptr), D ? D : R, wd ? wd : (R ? R : D), (cl_int)(D != nullptr),
             scale, OUT ? OUT : (R ? R : D), (cl_int)(OUT != nullptr), wq ? wq : (R ? R : D), (cl_int)norm_q,
             (cl_int)do_q, (Buf)g_Xp, (Buf)g_Xs, (cl_int)cols, eps);
    size_t g[1] = {(size_t)T * LS}, l[1] = {LS};
    return launch(k_rowf, 1, g, l);
}

bool op_qkv_post_g4(Buf C, int rowc, int T, int nq, int nkv, int hd, Buf qw, Buf kw, Buf invf, Buf posx, Buf posy, float eps) {
    if (hd != 64) return false;
    set_args(k_qkvg4, C, (cl_int)rowc, (cl_int)nq, (cl_int)nkv, (cl_int)hd, qw, kw, invf, posx, posy, eps);
    size_t g[2] = {(size_t)(nq + 2 * nkv) * 64, (size_t)T}, l[2] = {64, 1};
    return launch(k_qkvg4, 2, g, l);
}

// head-dim generic dense attention for one fully visible segment of n tokens (hd multiple of 64): f32 output O[n][nq*hd].
// S: scratch of nq*n*n floats. The softmax max reuses dattn_stats (dimension independent).
bool op_dense_attention_h(Buf C, int ldc, int qoff, int koff, int voff, int n, int nq, int nkv, int hd, float scale, Buf S, Buf O, int ldo, int window, size_t s_floats) {
    if (hd % 64 || hd > 512 || nq % nkv) return false;
    const int grp = nq / nkv;
    // query tile: the score scratch holds nq * rows * n floats
    int rows = (int)std::min<size_t>((size_t)n, s_floats / ((size_t)nq * n));
    rows = std::max(64, rows & ~63);                                 // whole 64-row tiles
    if ((size_t)nq * rows * n > s_floats) return false;
    static Buf g_dst = nullptr; static size_t g_dstcap = 0;
    const size_t stb = (size_t)nq * rows * 8;
    if (stb > g_dstcap) { if (g_dst) release(g_dst); g_dst = alloc(stb); g_dstcap = g_dst ? stb : 0; if (!g_dst) return false; }
    static const bool aprof = getenv("FERRO_TIME") != nullptr;   // ATTNH_PROF
    static double at[3] = {0, 0, 0}; static bool areg = false;
    if (aprof && !areg) { areg = true; atexit([] { fprintf(stderr, "[gpu] dense attention_h: qk %.0f ms  stats %.0f ms  pv %.0f ms\n", at[0] * 1e3, at[1] * 1e3, at[2] * 1e3); }); }
    auto lapa = [&](int k, std::chrono::steady_clock::time_point& t0) { if (!aprof) return; finish(); auto t = std::chrono::steady_clock::now(); at[k] += std::chrono::duration<double>(t - t0).count(); t0 = t; };
    for (int q0 = 0; q0 < n; q0 += rows) {
        auto ta0 = std::chrono::steady_clock::now();
        const int nr = std::min(rows, n - q0);
        const size_t qt = (size_t)(nr + 63) / 64, kt = (size_t)(n + 63) / 64;
        set_args(k_dqkh, C, (cl_int)ldc, (cl_int)qoff, (cl_int)koff, (cl_int)n, (cl_int)hd, (cl_int)grp, S, (cl_int)q0, (cl_int)nr);
        size_t g1[3] = {kt * 16, qt * 16, (size_t)nq}, l1[3] = {16, 16, 1};
        if (!launch(k_dqkh, 3, g1, l1)) return false;
        lapa(0, ta0);
        set_args(k_dsmw, S, (cl_int)n, scale, g_dst, (cl_int)window, (cl_int)q0, (cl_int)nr);
        size_t g2[2] = {(size_t)nr * 256, (size_t)nq}, l2[2] = {256, 1};
        if (!launch(k_dsmw, 2, g2, l2)) return false;
        lapa(1, ta0);
        set_args(k_dpvh, S, g_dst, scale, C, (cl_int)ldc, (cl_int)voff, (cl_int)n, (cl_int)hd, (cl_int)grp, O, (cl_int)ldo, (cl_int)window, (cl_int)q0, (cl_int)nr);
        size_t g3[2] = {qt * 16, (size_t)nq * (hd / 64) * 16}, l3[2] = {16, 16};
        if (!launch(k_dpvh, 2, g3, l3)) return false;
        lapa(2, ta0);
        if (q0 + rows < n && !finish()) return false;                 // bounded submissions (i915 watchdog); also orders the scratch reuse
    }
    return true;
}

bool op_qkv_post(Buf C, int rowc, int T, int nq, int nkv, int hd, Buf qw, Buf kw, Buf cosb, Buf sinb, Buf pos, int hlf, float eps, bool norm_v) {
    if (!norm_v && !(use_v3() && 2 * hlf == hd && hd % 8 == 0)) return false;   // only v3 can skip the V slots
    if (use_v3() && 2 * hlf == hd && hd % 8 == 0) {
        const size_t LS3 = (size_t)hd / 8;
        set_args(k_qkvp3, C, (cl_int)rowc, (cl_int)nq, (cl_int)nkv, (cl_int)hd, qw, kw, cosb, sinb, pos, (cl_int)hlf, eps);
        size_t g3[2] = {(size_t)(nq + (norm_v ? 2 : 1) * nkv) * LS3, (size_t)T}, l3[2] = {LS3, 1};
        return launch(k_qkvp3, 2, g3, l3);
    }
    const size_t LS = 64;
    set_args(k_qkvp, C, (cl_int)rowc, (cl_int)nq, (cl_int)nkv, (cl_int)hd, qw, kw, cosb, sinb, pos, (cl_int)hlf, eps);
    size_t g[2] = {(size_t)(nq + 2 * nkv) * LS, (size_t)T}, l[2] = {LS, 1};
    return launch(k_qkvp, 2, g, l);
}

static long attn_work_estimate(int);
bool op_attention_q(Buf C, int ldc, int qoff, int koff, int voff, int T, int nq, int nkv, int hd,
                    Buf seg_lo, Buf seg_hi, int window, float scale, Buf pre_lo, Buf pre_hi) {
    if (hd > 512 || hd % 32) return false;
    if (pre_lo && hd != 64) return false;   // prefix ranges: query-tiled kernel only
    static const bool v1 = getenv("FERRO_GPU_ATTN_V1") != nullptr;
    static const bool no4 = getenv("FERRO_GPU_ATTN_NO4") != nullptr;
    static const bool no5 = getenv("FERRO_GPU_ATTN_NO5") != nullptr;
    if ((!v1 && !no5 && use_v3() && hd == 64) || pre_lo) {
        // query-tiled; long batches are split into launches of bounded work with a sync between them, so no
        // single submission runs long enough for the i915 watchdog to reset the context (silent wrong results)
        static const long budget = getenv("FERRO_GPU_ATTN_BUDGET") ? atol(getenv("FERRO_GPU_ATTN_BUDGET")) : (64L << 20);
        const long work = attn_work_estimate(T);
        const int ntile = (T + 31) / 32;
        const int pieces = work <= budget ? 1 : (int)std::min<long>(ntile, (work + budget - 1) / budget);
        const int per = (ntile + pieces - 1) / pieces;
        for (int p0 = 0; p0 < ntile; p0 += per) {
            const int nt = std::min(per, ntile - p0);
            set_args(k_attq5, C, (cl_int)ldc, (cl_int)qoff, (cl_int)koff, (cl_int)voff, (cl_int)nq, (cl_int)nkv,
                     seg_lo, seg_hi, (cl_int)window, scale, (Buf)g_Xp, (Buf)g_Xs, (cl_int)T, (cl_int)(p0 * 32),
                     pre_lo ? pre_lo : seg_lo, pre_hi ? pre_hi : seg_hi, (cl_int)(pre_lo != nullptr));
            size_t g5[2] = {(size_t)nq * 32, (size_t)nt}, l5[2] = {32, 1};
            if (!launch(k_attq5, 2, g5, l5)) return false;
            if (pieces > 1 && !finish()) return false;
        }
        return true;
    }
    if (!v1 && !no4 && use_v3() && hd <= 128) {      // small head dims: lane-per-key kernel
        set_args(k_attq4, C, (cl_int)ldc, (cl_int)qoff, (cl_int)koff, (cl_int)voff, (cl_int)nq, (cl_int)nkv, (cl_int)hd,
                 seg_lo, seg_hi, (cl_int)window, scale, (Buf)g_Xp, (Buf)g_Xs);
        size_t g4[2] = {(size_t)nq * 32, (size_t)T}, l4[2] = {32, 1};
        return launch(k_attq4, 2, g4, l4);
    }
    if (!v1 && use_v3()) {
        const size_t LS3 = (size_t)hd / 4;
        set_args(k_attq3, C, (cl_int)ldc, (cl_int)qoff, (cl_int)koff, (cl_int)voff, (cl_int)nq, (cl_int)nkv, (cl_int)hd,
                 seg_lo, seg_hi, (cl_int)window, scale, (Buf)g_Xp, (Buf)g_Xs);
        size_t g3[2] = {(size_t)nq * LS3, (size_t)T}, l3[2] = {LS3, 1};
        return launch(k_attq3, 2, g3, l3);
    }
    if (!v1) {
        const size_t LS2 = (size_t)hd / 4;   // one float4 of the head dim per lane
        set_args(k_attq2, C, (cl_int)ldc, (cl_int)qoff, (cl_int)koff, (cl_int)voff, (cl_int)nq, (cl_int)nkv, (cl_int)hd,
                 seg_lo, seg_hi, (cl_int)window, scale, (Buf)g_Xp, (Buf)g_Xs);
        size_t g2[2] = {(size_t)nq * LS2, (size_t)T}, l2[2] = {LS2, 1};
        return launch(k_attq2, 2, g2, l2);
    }
    const size_t LS = 64;
    set_args(k_attq, C, (cl_int)ldc, (cl_int)qoff, (cl_int)koff, (cl_int)voff, (cl_int)nq, (cl_int)nkv, (cl_int)hd,
             seg_lo, seg_hi, (cl_int)window, scale, (Buf)g_Xp, (Buf)g_Xs);
    size_t g[2] = {(size_t)nq * LS, (size_t)T}, l[2] = {LS, 1};
    return launch(k_attq, 2, g, l);
}

bool op_gelu_mul_q(Buf G, int T, int FF) {
    if (use_v3() && FF % 32 == 0 && FF / 4 <= 1024) {
        const size_t LS3 = (size_t)FF / 4;
        set_args(k_gmq3, G, (cl_int)FF, (Buf)g_Xp, (Buf)g_Xs);
        size_t g3[1] = {(size_t)T * LS3}, l3[1] = {LS3};
        return launch(k_gmq3, 1, g3, l3);
    }
    set_args(k_gmq2, G, (cl_int)FF, (Buf)g_Xp, (Buf)g_Xs);
    size_t g[2] = {(size_t)FF / 32, (size_t)T};
    return launch(k_gmq2, 2, g, nullptr);
}

bool op_gelu_mulvec_q(Buf I, Buf H, int hoff, int hld, int T, int npl) {
    if (use_v3() && npl % 32 == 0 && npl / 4 <= 512) {
        const size_t LS3 = (size_t)npl / 4;
        set_args(k_gmvq3, I, H, (cl_int)hoff, (cl_int)hld, (cl_int)npl, (Buf)g_Xp, (Buf)g_Xs);
        size_t g3[1] = {(size_t)T * LS3}, l3[1] = {LS3};
        return launch(k_gmvq3, 1, g3, l3);
    }
    set_args(k_gmvq2, I, H, (cl_int)hoff, (cl_int)hld, (cl_int)npl, (Buf)g_Xp, (Buf)g_Xs);
    size_t g[2] = {(size_t)npl / 32, (size_t)T};
    return launch(k_gmvq2, 2, g, nullptr);
}

bool op_rmsnorm(Buf x, int xoff, int ldx, Buf y, int yoff, int ldy, int T, int nh, int cols, Buf w, float eps) {
    const size_t LS = cols >= 256 ? 128 : 64;
    set_args(k_rms, x, (cl_int)xoff, (cl_int)ldx, y, (cl_int)yoff, (cl_int)ldy, (cl_int)cols,
             w ? w : x, (cl_int)(w != nullptr), eps);
    size_t g[2] = {(size_t)nh * LS, (size_t)T}, l[2] = {LS, 1};
    return launch(k_rms, 2, g, l);
}

bool op_rope(Buf x, int xoff, int ld, int T, int heads, int hd, int half, Buf cosb, Buf sinb, Buf pos) {
    set_args(k_rope, x, (cl_int)xoff, (cl_int)ld, (cl_int)hd, (cl_int)half, cosb, sinb, pos);
    size_t g[3] = {(size_t)half, (size_t)heads, (size_t)T};
    return launch(k_rope, 3, g, nullptr);
}

bool op_attention(Buf C, int ldc, int qoff, int koff, int voff, int T, int nq, int nkv, int hd,
                  Buf seg_lo, Buf seg_hi, int window, float scale, Buf O, int ldo) {
    if (hd > 512 || hd % 4) return false;
    const size_t LS = 64;
    set_args(k_attn, C, (cl_int)ldc, (cl_int)qoff, (cl_int)koff, (cl_int)voff, (cl_int)nq, (cl_int)nkv,
             (cl_int)hd, seg_lo, seg_hi, (cl_int)window, scale, O, (cl_int)ldo);
    size_t g[2] = {(size_t)nq * LS, (size_t)T}, l[2] = {LS, 1};
    return launch(k_attn, 2, g, l);
}

bool op_gelu_mul(Buf G, int T, int FF, Buf F) {
    set_args(k_gmul, G, (cl_int)FF, F);
    size_t g[2] = {(size_t)FF, (size_t)T};
    return launch(k_gmul, 2, g, nullptr);
}

bool op_gelu_mulvec(Buf I, Buf H, int hoff, int n) {
    set_args(k_gmulv, I, H, (cl_int)hoff);
    size_t g[1] = {(size_t)n};
    return launch(k_gmulv, 1, g, nullptr);
}

bool op_add_scale(Buf a, Buf b, Buf y, int n, float s) {
    set_args(k_add, a, b, y, s);
    size_t g[1] = {(size_t)n};
    return launch(k_add, 1, g, nullptr);
}

// sum over queries of their key-range length (x heads) for the current batch; set by the model per forward
static long g_attn_work = 0;
void set_attn_work(long w) { g_attn_work = w; }
static long attn_work_estimate(int) { return g_attn_work; }

// ---- LFM2 decision ops ----
bool op_resid_norm_q(Buf X, Buf D, Buf db, Buf w, Buf b, bool is_ln, int T, int cols, float eps) {
    if (cols % 32 || cols / 4 > 512) return false;
    set_args(k_rnq, X, D ? D : X, (cl_int)(D != nullptr), db ? db : X, (cl_int)(db != nullptr), w, b ? b : w,
             (cl_int)is_ln, (Buf)g_Xp, (Buf)g_Xs, (cl_int)cols, eps);
    size_t g[1] = {(size_t)T * (cols / 4)}, l[1] = {(size_t)cols / 4};
    return launch(k_rnq, 1, g, l);
}
bool op_head_in_q(Buf X, Buf D, Buf ow, Buf te, Buf type, Buf lw, Buf lb, int T, int cols, float rms_eps, float ln_eps) {
    if (cols % 32 || cols / 4 > 512) return false;
    set_args(k_hinq, X, D, ow, te, type, lw, lb, (Buf)g_Xp, (Buf)g_Xs, (cl_int)cols, rms_eps, ln_eps);
    size_t g[1] = {(size_t)T * (cols / 4)}, l[1] = {(size_t)cols / 4};
    return launch(k_hinq, 1, g, l);
}
bool op_conv_gate_q(Buf C, Buf prev, Buf next, Buf taps, int T, int d) {
    if (d % 32 || d / 4 > 512) return false;
    set_args(k_convq, C, prev, next, taps, (Buf)g_Xp, (Buf)g_Xs, (cl_int)d);
    size_t g[1] = {(size_t)T * (d / 4)}, l[1] = {(size_t)d / 4};
    return launch(k_convq, 1, g, l);
}
bool op_silu_mul_q(Buf G, int T, int FF) {
    if (FF % 32 || FF / 8 > 1024) return false;
    set_args(k_siluq, G, (cl_int)FF, (Buf)g_Xp, (Buf)g_Xs);
    size_t g[1] = {(size_t)T * (FF / 8)}, l[1] = {(size_t)FF / 8};
    return launch(k_siluq, 1, g, l);
}
bool op_bias_act_q(Buf G, Buf b, bool relu, int T, int cols) { return op_bias_act_mode_q(G, b, relu ? 1 : 0, T, cols); }
bool op_bias_act_mode_q(Buf G, Buf b, int act, int T, int cols) {
    if (cols % 32 || cols / 8 > 1024) return false;
    set_args(k_bactq, G, b, (cl_int)act, (Buf)g_Xp, (Buf)g_Xs, (cl_int)cols);
    size_t g[1] = {(size_t)T * (cols / 8)}, l[1] = {(size_t)cols / 8};
    return launch(k_bactq, 1, g, l);
}
bool op_quant_q(Buf X, int T, int cols) {
    if (cols % 32 || cols / 4 > 512) return false;
    set_args(k_quantq, X, (Buf)g_Xp, (Buf)g_Xs, (cl_int)cols);
    size_t g[1] = {(size_t)T * (cols / 4)}, l[1] = {(size_t)cols / 4};
    return launch(k_quantq, 1, g, l);
}
// dense attention over one fully-visible segment [r0, r0+n): S scratch of nh*n*n floats, O f32 (ldo = nh*64)
bool op_dense_attention(Buf C, int ldc, int qoff, int koff, int voff, int r0, int n, int nh, float scale, Buf S, Buf O, int ldo) {
    static const bool prof = getenv("FERRO_TIME") != nullptr;
    static double tt[3] = {0, 0, 0};
    static bool reg = false;
    if (prof && !reg) { reg = true; atexit([] { fprintf(stderr, "[gpu] dense attention: qk %.0f ms  softmax %.0f ms  pv %.0f ms\n", tt[0] * 1e3, tt[1] * 1e3, tt[2] * 1e3); }); }
    auto t0 = std::chrono::steady_clock::now();
    auto lap = [&](int k) { if (!prof) return; finish(); auto t = std::chrono::steady_clock::now(); tt[k] += std::chrono::duration<double>(t - t0).count(); t0 = t; };
    const size_t nt = (size_t)(n + 63) / 64;
    set_args(k_dqk, C, (cl_int)ldc, (cl_int)qoff, (cl_int)koff, (cl_int)r0, (cl_int)n, S);
    size_t g1[3] = {nt * 16, nt * 16, (size_t)nh}, l1[3] = {16, 16, 1};
    if (!launch(k_dqk, 3, g1, l1)) return false;
    lap(0);
    static Buf g_dst = nullptr; static size_t g_dstcap = 0;
    const size_t stb = (size_t)nh * n * 8;
    if (stb > g_dstcap) { if (g_dst) release(g_dst); g_dst = alloc(stb); g_dstcap = g_dst ? stb : 0; if (!g_dst) return false; }
    set_args(k_dsm, S, (cl_int)n, scale, g_dst);
    size_t g2[2] = {(size_t)n * 256, (size_t)nh}, l2[2] = {256, 1};
    if (!launch(k_dsm, 2, g2, l2)) return false;
    lap(1);
    set_args(k_dpv, S, g_dst, scale, C, (cl_int)ldc, (cl_int)voff, (cl_int)r0, (cl_int)n, O, (cl_int)ldo);
    size_t g3[2] = {nt * 16, (size_t)nh * 16}, l3[2] = {16, 16};
    bool ok = launch(k_dpv, 2, g3, l3);
    lap(2);
    return ok;
}
bool op_ln_gather_q(Buf X, Buf src, Buf w, Buf b, int V, int cols, float eps) {
    if (cols % 32 || cols / 4 > 512) return false;
    set_args(k_lngq, X, src, w, b, (Buf)g_Xp, (Buf)g_Xs, (cl_int)cols, eps);
    size_t g[1] = {(size_t)V * (cols / 4)}, l[1] = {(size_t)cols / 4};
    return launch(k_lngq, 1, g, l);
}
bool op_bias_add(Buf Y, Buf b, int T, int rows) {
    set_args(k_badd, Y, b, (cl_int)rows);
    size_t g[2] = {(size_t)rows, (size_t)T};
    return launch(k_badd, 2, g, nullptr);
}
bool op_score_out(Buf O, Buf cb, Buf w, float wb, Buf marks, Buf out, int M, int d) {
    if (d % 4 || d / 4 > 512) return false;
    set_args(k_score, O, cb, w, wb, marks, out, (cl_int)d);
    size_t g[1] = {(size_t)M * (d / 4)}, l[1] = {(size_t)d / 4};
    return launch(k_score, 1, g, l);
}

// legacy host-pointer GEMM: one sync per call
bool gemm(int slot, const float* X, int T, float* Y) {
    if (!g_ok || slot < 0 || slot >= (int)g_slots.size() || T <= 0) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    const Slot& s = g_slots[slot];
    const size_t xb = (size_t)T * s.cols * 4, yb = (size_t)T * s.rows * 4;
    if (!ensure(g_hX, g_hXcap, xb) || !ensure(g_hY, g_hYcap, yb)) return false;
    if (!write((Buf)g_hX, X, xb)) return false;
    if (!op_gemm(slot, (Buf)g_hX, T, (Buf)g_hY)) return false;
    return read((Buf)g_hY, Y, yb);
}

void shutdown() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_ok) return;
    for (auto& s : g_slots) { release(s.wp); release(s.ws); }
    g_slots.clear();
    for (cl_mem m : {g_Xp, g_Xs, g_hX, g_hY}) if (m) cl.clReleaseMemObject(m);
    g_Xp = g_Xs = g_hX = g_hY = nullptr;
    for (cl_kernel k : {k_quant, k_gemm, k_gemm_sw, k_rms, k_rope, k_attn, k_gmul, k_gmulv, k_add, k_rowf, k_qkvp, k_attq, k_gmq, k_gmvq,
                        k_attq2, k_gmq2, k_gmvq2, k_rowf3, k_qkvp3, k_attq3, k_gmq3, k_gmvq3,
                        k_rnq, k_hinq, k_convq, k_siluq, k_bactq, k_badd, k_score, k_attq4, k_attq5, k_quantq, k_lngq, k_dqk, k_dsm, k_dpv, k_qkvg4, k_dqkh, k_dpvh, k_dsmw})
        if (k) cl.clReleaseKernel(k);
    cl.clReleaseProgram(g_prog); cl.clReleaseCommandQueue(g_q); cl.clReleaseContext(g_ctx);
    g_ok = false;
}

}}  // namespace ferro::gpu
