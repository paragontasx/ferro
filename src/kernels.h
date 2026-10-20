// kernels.h — SIMD compute kernels (AVX2/FMA + AVX-VNNI dispatch, scalar fallback)
#pragma once
#include "types.h"
#include <vector>
#include <cstdint>

namespace ferro {

// ---- quantized activation buffer: u8-biased int8, per (token, 32-col block) scale ----
struct ActQ {
    std::vector<uint8_t> qx;   // [T*cols]  value = clamp(round(x/s)+128, 0, 255)
    std::vector<float>   sx;   // [T*nb]    block scales (dequant multiplier)
    int T = 0, cols = 0, nb = 0;
    void ensure(int t, int c);
};

// quantize x [T x cols] f32 to u8 blocks of 32
void quant_act(const float* x, int T, int cols, ActQ& out);

// GEMM operand in plane layout: wq [rows*cols] int8 (per 32-block),
// ws [rows*nb] f32 (dequant scale), rs [rows*nb] f32 (-128 * sum_int8(rowblock) * ws)
struct WMat {
    int32_t rows = 0, cols = 0, nb = 0;
    std::vector<int8_t> wq;
    std::vector<float>  ws;
    std::vector<float>  rs;    // pre-folded bias term
    std::vector<float>  wf;    // optional pre-dequantized f32 weights [rows*cols]
    // int8 row-domain weights for fast int8 GEMM (regenerated from wf/q8 as needed)
    std::vector<int8_t> wq_row;   // [rows*cols] signed, row-scaled
    std::vector<float>  w_scale;  // [rows] dequant multiplier for wq_row
    std::vector<float>  w_bias;   // [rows] fold of activation zero-point term
    const char* name = "";
    int gpu_slot = -1;            // >= 0: weights resident on the iGPU (gpu_cl.h), GEMM routed there
    bool empty() const { return rows == 0; }
};

// build a WMat from ggml Q8_0 raw data (row-major blocks, rows*row_stride bytes)
WMat pack_q8(const uint8_t* raw, int rows, int cols, int row_stride_bytes, const char* name);
// concatenate multiple WMat row-wise (must share cols)
WMat pack_concat(std::vector<const WMat*> parts, const char* name);
// convert f32 source into WMat (quantize per row, per block-32, symmetric)
WMat pack_from_f32(const float* src, int rows, int cols, int row_stride_floats, const char* name);

// Y[t*rows + r] = sum_c W[r][c]*X[t][c]   (fp32 out)
void gemm_q8(const WMat& W, const ActQ& X, int T, float* Y, int y_ld);
// with fp32 bias add + optional residual: Y = X_res + W@Xq^T + b
void gemm_q8_full(const WMat& W, const ActQ& X, int T,
                  const float* bias /*rows or null*/, const float* resid /*[T x rows] or null*/,
                  float* Y, int y_ld);

// MAIN PATH: fp32 activations, Q8 planes dequantized to f32 tiles once per chunk.
// Y = resid + (W dequant) @ X^T + bias.  resid rows are W.rows apart; resid == Y is allowed
// (snapshotted internally — costs a copy, so ping-pong buffers on hot paths).
void pack_q8_to_f32(WMat& W);
void pack_q8_rowdomain(WMat& W);            // build wq_row/w_scale/w_bias from wq/ws
void gemm_q8_int8(const WMat& W, const float* X, int T, float* Y, int y_ld);  // int8 madd GEMM
void gemm_q8_fma(const WMat& W, const float* X, int T,
                 const float* bias, const float* resid, float* Y, int y_ld);

// scalar reference GEMM (correctness oracle)
void gemm_q8_ref(const WMat& W, const ActQ& X, int T, float* Y, int y_ld);

// f32 GEMM from raw f32 weights (for BF16-sourced matrices), row-major
void gemm_f32(const float* W, int rows, int cols, const float* X, int T, float* Y, int y_ld);

// ---- fused vector ops (all row-major [T x ld] unless noted) ----
// y row r = rmsnorm(x row r) * w (eps); rows are (r/cols_in_row) * ld + ... caller passes
// ld = stride between successive rows of the same "row vector" (0 = use cols).
void rmsnorm_q(const float* x, int T, int cols, const float* w, float eps, float* y, int ld_x = 0, int ld_y = 0);
void rmsnorm_w1(const float* x, int T, int cols, float eps, float* y, int ld_x = 0, int ld_y = 0); // weightless (V-norm)
void add_rows(const float* a, const float* b, int n, float* y);           // y=a+b
void add_scaled(const float* a, const float* b, float s, int n, float* y);// y=a+s*b
void scale_rows(float s, int n, float* y);
// y[t*cols + i] = gelu(g[t*ldg + i]) * u[t*ldg + cols + i]  (fused gate/up block, erf-exact)
void gelu_mul_block(const float* gu, int T, int cols, float* y);
void gelu_i(float* x, int n);
void gelu_fma(float* x, int n);
void gelu_mulvec(float* x, const float* h, int n);

// fused FFN: W rows interleaved (gate_r, up_r); produces Y[t][r] = gelu(gate_r) * up_r
WMat pack_concat_inter(const WMat& a, const WMat& b, const char* name);
void gemm_gateup_fused(const WMat& W, const float* X, int T, float* Y, int FF);
void gelu_mul_block_inter(const float* G, int T, int FF, float* Y);

// rope: x [T][heads][head_dim] rotated in place (GPT-J pairs), cos/sin cached per position
struct RopeTable {
    int rot = 0;                       // rotated dims per head (even)
    std::vector<float> cosb, sinb;     // [max_pos][rot/2]
    void build(double freq_base, int rot, int max_pos, double attn_factor = 1.0);
};
void rope_apply(float* x, int T, int heads, int head_dim, const RopeTable& rt, const int* pos, int ld = 0);

// attention (no cache): scores = Q@K^T*scale (windowed, bidirectional), softmax, @V -> O
// Q: rows [t][h*hd..] at t*ldq + h*hd, K/V: t*ldk + h*hd; GQA: q-head h uses kv-head h/(nq/nkv)
// window: symmetric; <0 = full. mask blocks across sequences via seg[t] ids.
void attention(const float* Q, const float* K, const float* V, float* O,
               int T, int nq, int nkv, int hd, float scale,
               const int* seg, int window, int ldq, int ldk, int ldo);

// mean pool rows [T x cols] of segment tokens -> y [cols] (valid tokens), then L2 normalize
void mean_pool(const float* x, int T, int cols, float* y, bool l2);

// gather+dequant embedding rows: y[t] = tok_embd[id[t]] * scale_mul
void gather_embd(const WMat& tok, const int* ids, int T, float scale_mul, float* y);

// runtime ISA level: 0=scalar 1=avx2 2=avx2+vnni
int isa_level();          // detected
// per-thread: when set, GEMMs never route to the iGPU (CPU half of a heterogeneous split)
void set_thread_cpu_only(bool on);
bool thread_cpu_only();
void set_isa_level(int l);

} // namespace ferro
