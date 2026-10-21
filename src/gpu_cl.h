// gpu_cl.h — optional Intel iGPU (OpenCL) backend for ferro-embed.
// libOpenCL is dlopen'd at runtime: the binary runs unchanged on machines without a GPU.
// All ops enqueue onto ONE in-order queue without host syncs; read() is the only sync point.
#pragma once
#include <cstddef>

namespace ferro {
struct WMat;
namespace gpu {

bool init(bool verbose);
bool available();
const char* device_name();
size_t resident_bytes();
void shutdown();

// ---- weights ----
int upload_q8(const WMat& W);     // int8 DP4A layout; slot >= 0 or -1 (unsupported shape)

// ---- device buffers (opaque cl_mem) ----
using Buf = void*;
Buf alloc(size_t bytes);
Buf upload(const void* src, size_t bytes);          // alloc + blocking copy (load-time constants)
bool write(Buf b, const void* src, size_t bytes);   // non-blocking enqueue (src must stay valid until next read/finish)
bool read(Buf b, void* dst, size_t bytes);
bool copy(Buf dst, size_t dst_off, Buf src, size_t src_off, size_t bytes);   // device-side, enqueued          // blocking: the per-forward sync point
bool finish();
void release(Buf b);

// ---- ops (offsets/strides in floats) ----
// y[t*ldy + yoff + h*cols + c] = x[...] * rsqrt(mean(x^2)+eps) * (w ? w[c] : 1),  h < nh
bool op_rmsnorm(Buf x, int xoff, int ldx, Buf y, int yoff, int ldy, int T, int nh, int cols, Buf w, float eps);
// rotate-half rope on heads at x[t*ld + xoff + h*hd]: pairs (i, i+half), tables [pos][half]
bool op_rope(Buf x, int xoff, int ld, int T, int heads, int hd, int half, Buf cosb, Buf sinb, Buf pos);
// bidirectional segment-masked attention; seg bounds lo/hi per token; window<=0 = full
bool op_attention(Buf C, int ldc, int qoff, int koff, int voff, int T, int nq, int nkv, int hd,
                  Buf seg_lo, Buf seg_hi, int window, float scale, Buf O, int ldo);
bool op_gelu_mul(Buf G, int T, int FF, Buf F);              // F[t][i] = gelu(G[t][i]) * G[t][FF+i]
bool op_gelu_mulvec(Buf I, Buf H, int hoff, int n);         // I[i] = gelu(I[i]) * H[hoff+i]
bool op_add_scale(Buf a, Buf b, Buf y, int n, float s);     // y = (a + b) * s
bool op_gemm(int slot, Buf X, int T, Buf Y);                // Y[T x rows] = X[T x cols] @ W^T (int8 DP4A)

// ---- fused path: producers write the int8 GEMM input (shared scratch) directly ----
bool ensure_qscratch(int T, int max_cols);
bool warm_small_gemm(int n = 3);   // build the 1/2/4(/8)-token decode GEMM variants now instead of on first use
const char* build_id();             // compile time of the kernels: part of the identity of persisted device state
void set_row_policy(int p);         // 0 auto; 1 rows (GEMV for any row count: batch-invariant decode); 2 tile (tile GEMM for any row count: chunk-invariant prefill)
bool op_gemm_q(int slot, int T, Buf Y);                     // GEMM on the current quantized scratch
// y = ((R?R:0) + (D? rms(D)*wd : 0))*scale -> OUT (if OUT); then quantize (norm_q ? rms(y)*wq : y) if do_q
bool op_row_fused(Buf R, Buf D, Buf wd, float scale, Buf OUT, Buf wq, bool norm_q, bool do_q, int T, int cols, float eps);
bool op_qkv_post(Buf C, int rowc, int T, int nq, int nkv, int hd, Buf qw, Buf kw, Buf cosb, Buf sinb, Buf pos, int hlf, float eps,
                 bool norm_v = true);   // norm_v=false: V untouched (models without a V-norm)
// Gemma 4 vision (hd = 64): q/k RMSNorm(w) + 2-D RoPE (posx on dims [0,32), posy on [32,64), invf[16]); v RMSNorm (no weight)
bool op_qkv_post_g4(Buf C, int rowc, int T, int nq, int nkv, int hd, Buf qw, Buf kw, Buf invf, Buf posx, Buf posy, float eps);
// dense attention for one fully visible segment, any head dim multiple of 64 (<= 512), grouped-query; f32 out O[n][nq*hd]
bool op_dense_attention_h(Buf C, int ldc, int qoff, int koff, int voff, int n, int nq, int nkv, int hd, float scale, Buf S, Buf O, int ldo, int window = -1, size_t s_floats = (size_t)1 << 62);   // S capacity in floats: queries are tiled to fit   // window > 0: keys with |i-j| >= window are masked (SWA)
// ---- Gemma 4 audio conformer (see mm_gemma4a.cpp); all rows are f32 [T][cols], quantized outputs go to the shared int8 scratch
bool op_a_norm_quant(Buf X, Buf w, bool do_rms, float lo, float hi, int T, int cols, float eps);   // [rms] [*w] clamp -> int8
bool op_a_act_quant(Buf Y, int act, float olo, float ohi, float ilo, float ihi, int T, int cols);   // clamp [silu] clamp -> int8 (act 1 = silu)
bool op_a_resid(Buf X, Buf Y, float olo, float ohi, bool do_rms, Buf w, float scale, Buf w2, int T, int cols, float eps);  // X += scale*[rms](clamp Y)*w; w2: X = rms(X)*w2
bool op_a_conv_mid(Buf G2, int d, float olo, float ohi, Buf dw, Buf wn, float ilo, float ihi, int T, float eps);          // GLU + causal dw conv k5 + rms + silu -> int8
bool op_a_attn(Buf C, int ldc, Buf Pk, Buf pds, float lo, float hi, float qs, float ks, int N, int nh, int d, Buf O);      // chunked local attention, hd 128
bool op_attention_q(Buf C, int ldc, int qoff, int koff, int voff, int T, int nq, int nkv, int hd,
                    Buf seg_lo, Buf seg_hi, int window, float scale,
                    Buf pre_lo = nullptr, Buf pre_hi = nullptr);
bool op_gelu_mul_q(Buf G, int T, int FF);
bool op_gelu_mulvec_q(Buf I, Buf H, int hoff, int hld, int T, int npl);   // H row stride hld

// ---- LFM2 decision models: producers that quantize into the shared int8 GEMM scratch ----
// X += D (+db) (if D); quantize rms(X)*w, or layernorm(X)*w+b when is_ln
bool op_resid_norm_q(Buf X, Buf D, Buf db, Buf w, Buf b, bool is_ln, int T, int cols, float eps);
// X = rms(X + D)*ow + te[type[t]]; quantize layernorm(X)*lw + lb
bool op_head_in_q(Buf X, Buf D, Buf ow, Buf te, Buf type, Buf lw, Buf lb, int T, int cols, float rms_eps, float ln_eps);
// centred 3-tap short conv gate over C = [b|c|x] (segment bounds inclusive) -> quantized z
bool op_conv_gate_q(Buf C, Buf prev, Buf next, Buf taps, int T, int d);   // prev/next: neighbour rows, -1 = none
// ---- LFM2 generative (causal) models; hd = 64, outputs go to the shared int8 scratch
bool op_lm_conv_q(Buf C, Buf src1, Buf src2, Buf st, Buf taps, int T, int d);     // causal conv gate; src: row >= 0 or state ref -1-(slot*2+k)
bool op_lm_conv_state(Buf C, Buf st, Buf slot, Buf r1, Buf r2, int ns, int d);    // st[slot] <- b*x of rows r2 (or old newer slot), r1
bool op_lm_conv_win_q(Buf C, Buf src1, Buf src2, Buf st, Buf taps, int T, int d);   // speculative-verification conv: lm_conv_dec_q arithmetic over consecutive rows, state untouched
bool op_lm_kv_write(Buf C, int rowc, int koff, Buf KV, Buf dst, int T, int kvw);  // KV[dst[t]] <- C[t][koff .. koff+kvw)
// decode GEMV with the activation prologue fused in (T<=4). mode 1: x=rmsnorm(A+B)*wn (+ store A+B to Xo); mode 2: x=silu(A[:ff])*A[ff:2ff]
bool gemv_pro_ok(int slot, int T, int mode, int ff);
bool op_gemv_pro(int slot, int T, Buf Y, int mode, Buf A, Buf B, Buf wn, Buf Xo, int ff, float eps);
// fused decode versions (one row per sequence): conv gate + state update; q/k norm + rope + K/V cache write
bool op_lm_conv_dec_q(Buf C, Buf st, Buf slot, Buf taps, int T, int d);
bool op_lm_qkv_post_kv(Buf C, int rowc, int T, int nq, int nkv, int hd, Buf qw, Buf kw, Buf cosb, Buf sinb, Buf pos, int hlf, float eps, Buf KV, Buf dst, int kvw);
bool op_lm_attn_pf(Buf C, int ldc, Buf KV, int ldkv, int voff, int kvbase, int nq, int nkv, int pos0, float scale, int T);
bool op_lm_attn_dec(Buf C, int ldc, Buf KV, int ldkv, int voff, Buf bases, Buf lens, int nq, int nkv, float scale, int B);
// int8 KV cache variants: data bytes KVq[row][ldkv] + one fp32 scale per 32 values KVs[row][ldkv/32]
bool op_lm_attn_pf_q(Buf C, int ldc, Buf KVq, Buf KVs, int ldkv, int voff, int kvbase, int nq, int nkv, int pos0, float scale, int T);
bool op_lm_attn_dec_q(Buf C, int ldc, Buf KVq, Buf KVs, int ldkv, int voff, Buf bases, Buf lens, int nq, int nkv, float scale, int B);
bool op_lm_qkv_post_kvq(Buf C, int rowc, int T, int nq, int nkv, int hd, Buf qw, Buf kw, Buf cosb, Buf sinb, Buf pos, int hlf, float eps, Buf KVq, Buf KVs, Buf dst, int kvw);
bool op_silu_mul_q(Buf G, int T, int FF);                    // silu(G[:, :FF]) * G[:, FF:] -> int8
bool op_bias_act_q(Buf G, Buf b, bool relu, int T, int cols);// (G + b) [relu] -> int8
bool op_bias_add(Buf Y, Buf b, int T, int rows);
// act(G + b) -> int8 (act: 0 none, 1 relu, 2 gelu-tanh, 3 gelu-erf)
bool op_bias_act_mode_q(Buf G, Buf b, int act, int T, int cols);
bool op_quant_q(Buf X, int T, int cols);
// dense attention over one fully-visible segment [r0, r0+n) of C (q|k|v at qoff/koff/voff, head dim 64):
// S = scratch of nh*n*n floats; O receives the f32 output (row stride ldo)
bool op_dense_attention(Buf C, int ldc, int qoff, int koff, int voff, int r0, int n, int nh, float scale, Buf S, Buf O, int ldo);                                   // f32 rows -> int8 scratch
// int8 row v = LayerNorm(X[src[v]])*w+b (src[v] < 0: zeros); V rows of cols
bool op_ln_gather_q(Buf X, Buf src, Buf w, Buf b, int V, int cols, float eps);             // Y[t][r] += b[r]
// out[k] = gelu_erf(O[marks[k]] + cb) . w + wb
bool op_score_out(Buf O, Buf cb, Buf w, float wb, Buf marks, Buf out, int M, int d);
// attention work of the current batch: sum over tokens of their segment length (bounds work per submission)
void set_attn_work(long w);

// legacy host-pointer GEMM (one sync per call) — kept for diagnostics
bool gemm(int slot, const float* X, int T, float* Y);

// max segment length the attention kernel supports (local-memory score buffer)
int attn_max_seg();

}  // namespace gpu
}  // namespace ferro
