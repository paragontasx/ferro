// kernels_fma.cpp — AVX2 + FMA fast paths for the N100 (Gracemont) core
// Primary kernel: 8x8 register-tiled f32 GEMM fed by once-dequantized Q8_0 tiles.
#include "kernels.h"
#include "gpu_cl.h"
#include "threadpool.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <algorithm>

#pragma GCC target("avx2,fma")

namespace ferro {

struct AlignedF32 {
    float* p = nullptr; size_t n = 0;
    ~AlignedF32() { if (p) aligned_free(p); }
    void resize(size_t count) {
        if (n >= count) return;
        if (p) aligned_free(p);
        n = (count + 15) & ~size_t(15);   // round to 16 floats = 64B
        p = (float*)::aligned_alloc(64, ((n * 4 + 63) / 64) * 64);
    }
    static void aligned_free(void* p) { ::free(p); }
};

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// ---------------- 8x8 micro-kernel (f32 row-major W, X row per token) ----------------
// Y[t][r] = sum_c W[r][c] * X[t][c].
// Strategy: transpose 8-token tile of X into Xt[c][8] (L1), then each row accumulates
// acc[i] (lanes = 8 tokens) with broadcast weight scalars -> 8 MAC per FMA instruction.
// Requires rows multiple of 8; tail rows handled by scalar loop.
// transpose X into tile-major layout: Xt[tile][c][8], zero-padded — done ONCE per gemm
static void transpose_x_tiles(const float* __restrict__ X, int T, int cols, float* __restrict__ Xt) {
    for (int t0 = 0; t0 < T; t0 += 8) {
        int nt = std::min(8, T - t0);
        float* xt = Xt + (size_t)(t0 / 8) * (size_t)cols * 8;
        if (nt == 8) {
            const float* xp[8];
            for (int j = 0; j < 8; ++j) xp[j] = X + (size_t)(t0 + j) * cols;
            for (int c = 0; c < cols; ++c)
                for (int j = 0; j < 8; ++j) xt[c * 8 + j] = xp[j][c];
                } else {
            for (int c = 0; c < cols; ++c)
                for (int j = 0; j < 8; ++j)
                    xt[c * 8 + j] = (j < nt) ? X[(size_t)(t0 + j) * cols + c] : 0.0f;
        }
    }
}

static void micro_tiles(const float* __restrict__ Wf, int rows, int cols,
                        const float* __restrict__ X, int T, float* __restrict__ Y, int y_ld,
                        const float* __restrict__ Xt) {
    // 4 rows x 16 tokens tile: 8 accumulators, each weight broadcast feeds 2 FMAs.
    // Measured +12% over 8x8 on N100 (halves load-port pressure per FMA).
    // Rows blocked so a weight block (RBLK x cols x 4B) stays in L2 across token tiles.
    int rblk = std::max(64, (1 << 20) / (cols * 4));
    rblk = (rblk / 4) * 4;
    for (int rb = 0; rb < rows; rb += rblk) {
    int re = std::min(rows, rb + rblk);
    int rsend = (re / 4) * 4;
    for (int t0 = 0; t0 < T; t0 += 16) {
        int nt = std::min(16, T - t0);
        int nh = (nt + 7) / 8;
        const float* xa = Xt + (size_t)(t0 / 8) * (size_t)cols * 8;
        const float* xb = xa + (size_t)cols * 8;
        for (int r0 = (rb / 4) * 4; r0 < rsend; r0 += 4) {
            __m256 acc[4][2];
            for (int i = 0; i < 4; ++i) { acc[i][0] = _mm256_setzero_ps(); acc[i][1] = _mm256_setzero_ps(); }
            const float* w = Wf + (size_t)r0 * cols;
            if (nh > 1) {
                for (int c = 0; c < cols; ++c) {
                    __m256 b0 = _mm256_load_ps(xa + c * 8), b1 = _mm256_load_ps(xb + c * 8);
                    for (int i = 0; i < 4; ++i) {
                        __m256 wv = _mm256_broadcast_ss(w + (size_t)i * cols + c);
                        acc[i][0] = _mm256_fmadd_ps(wv, b0, acc[i][0]);
                        acc[i][1] = _mm256_fmadd_ps(wv, b1, acc[i][1]);
                    }
                }
            } else {
                for (int c = 0; c < cols; ++c) {
                    __m256 b0 = _mm256_load_ps(xa + c * 8);
                    for (int i = 0; i < 4; ++i)
                        acc[i][0] = _mm256_fmadd_ps(_mm256_broadcast_ss(w + (size_t)i * cols + c), b0, acc[i][0]);
                }
            }
            alignas(32) float Ct[4][16];
            for (int i = 0; i < 4; ++i) {
                _mm256_store_ps(Ct[i], acc[i][0]);
                if (nh > 1) _mm256_store_ps(Ct[i] + 8, acc[i][1]);
            }
            for (int j = 0; j < nt; ++j)
                for (int i = 0; i < 4; ++i)
                    Y[(size_t)(t0 + j) * y_ld + r0 + i] = Ct[i][j];
        }
        // tail rows within block
        for (int r = rsend; r < re; ++r) {
            const float* w = Wf + (size_t)r * cols;
            for (int j = 0; j < nt; ++j) {
                const float* xr = X + (size_t)(t0 + j) * cols;
                float s = 0;
                for (int c = 0; c < cols; ++c) s += w[c] * xr[c];
                Y[(size_t)(t0 + j) * y_ld + r] = s;
            }
        }
    }
    }
}

void gemm_f32_micro(const float* W, int rows, int cols, const float* X, int T, float* Y, int y_ld) {
    static thread_local AlignedF32 xt_all;
    xt_all.resize((size_t)cols * 8 * ((T + 7) / 8));
    transpose_x_tiles(X, T, cols, xt_all.p);
    float* const xtp = xt_all.p;
    Pool::get().parallel_for((rows + 3) / 4, [&](int lo, int hi) {
        int r0 = lo * 4, r1 = std::min(rows, hi * 4);
        micro_tiles(W + (size_t)r0 * cols, r1 - r0, cols, X, T, Y + r0, y_ld, xtp);
    });
}

// ---------------- Q8 plane -> f32 tile dequant + GEMM ----------------
void pack_q8_to_f32(WMat& W) {
    W.wf.resize((size_t)W.rows * W.cols);
    const int nb = W.nb, cols = W.cols;
    for (int r = 0; r < W.rows; ++r) {
        const int8_t* src = W.wq.data() + (size_t)r * cols;
        const float* srow = W.ws.data() + (size_t)r * nb;
        float* dst = W.wf.data() + (size_t)r * cols;
        for (int b = 0; b < nb; ++b) {
            __m256 s = _mm256_set1_ps(srow[b]);
            int c0 = b * QK8_0;
            for (int i = 0; i < QK8_0; i += 8) {
                __m128i i8 = _mm_loadl_epi64((const __m128i*)(src + c0 + i));
                _mm256_storeu_ps(dst + c0 + i, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8)), s));
            }
        }
    }
}


// ---- int8 row-domain GEMM (VPMADDUBSW fast path) ----
// Build signed row-domain weights from the per-block q8 planes.
void pack_q8_rowdomain(WMat& W) {
    const int rows = W.rows, cols = W.cols, nb = W.nb;
    // per-row max |w| from block data: max_b(|q|max * ws[b]); exact max over stored ints:
    W.w_scale.assign(rows, 0.f);
    W.wq_row.assign((size_t)rows * cols, 0);
    W.w_bias.assign(rows, 0.f);
    for (int r = 0; r < rows; ++r) {
        const int8_t* src = W.wq.data() + (size_t)r * cols;
        const float* srow = W.ws.data() + (size_t)r * nb;
        float amax = 0.f;
        for (int b = 0; b < nb; ++b) {
            int c0 = b * QK8_0, c1 = std::min(c0 + QK8_0, cols);
            int qm = 0;
            for (int c = c0; c < c1; ++c) qm = std::max(qm, (int)std::abs((int)src[c]));
            amax = std::max(amax, (float)qm * srow[b]);
        }
        if (amax <= 0.f) amax = 1.f;
        float s = amax / 127.f;
        W.w_scale[r] = s;
        int8_t* dst = W.wq_row.data() + (size_t)r * cols;
        int64_t sum = 0;
        float inv = 1.f / s;
        for (int b = 0; b < nb; ++b) {
            int c0 = b * QK8_0, c1 = std::min(c0 + QK8_0, cols);
            float sb = srow[b] * inv;
            for (int c = c0; c < c1; ++c) {
                int q = (int)std::lroundf((float)src[c] * sb);
                q = q > 127 ? 127 : (q < -127 ? -127 : q);
                dst[c] = (int8_t)q;
                sum += q;
            }
        }
        W.w_bias[r] = (float)sum;   // folded later with activation offset
    }
}

// Y[t][r] = sw[r] * sa[t] * dot(u8(xq+128), wq_row) - sw[r]*128*sa[t]*sum_w[r]... via w_bias
void gemm_q8_int8(const WMat& W, const float* X, int T, float* Y, int y_ld) {
    const int rows = W.rows, cols = W.cols;
    // quantize activations row-wise (single scale per token row), u8 domain (+128)
    static thread_local std::vector<uint8_t> aq;
    static thread_local std::vector<float> as;
    aq.assign((size_t)T * cols, 128);
    as.assign(T, 1.f);
    {
        const size_t nc = (size_t)cols;
        for (int t = 0; t < T; ++t) {
            const float* xr = X + (size_t)t * cols;
            float amax = 0.f;
            for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(xr[c]));
            float s = amax > 1e-30f ? amax / 127.f : 1.f;
            as[t] = s;
            uint8_t* q = aq.data() + (size_t)t * nc;
            float inv = 1.f / s;
            for (int c = 0; c < cols; ++c) {
                int qq = (int)std::lroundf(xr[c] * inv) + 128;
                q[c] = (uint8_t)(qq < 0 ? 0 : (qq > 255 ? 255 : qq));
            }
        }
    }
    const int8_t* wq = W.wq_row.data();
    const uint8_t* aqp = aq.data();
    const float* asp = as.data();
    Pool::get().parallel_for((rows + 3) / 4, [&](int lo, int hi) {
        int r0 = lo * 4, r1 = std::min(rows, hi * 4);
        for (int t = 0; t < T; ++t) {
            const uint8_t* a = aqp + (size_t)t * cols;
            float sa = asp[t];
            __m256i ones = _mm256_set1_epi16(1);
            for (int r = r0; r < r1; ++r) {
                const int8_t* w = wq + (size_t)r * cols;
                __m256i acc = _mm256_setzero_si256();
                for (int c = 0; c + 32 <= cols; c += 32) {
                    __m256i av = _mm256_loadu_si256((const __m256i*)(a + c));
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(w + c));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(av, wv), ones));
                }
                for (int c = (cols / 32) * 32; c < cols; ++c) {
                    ((int32_t*)&acc)[0] += (int)a[c] * (int)w[c];
                }
                alignas(32) int32_t lanes[8];
                _mm256_store_si256((__m256i*)lanes, acc);
                int32_t dot = 0; for (int k = 0; k < 8; ++k) dot += lanes[k];
                float v = (float)dot - 128.f * W.w_bias[r];
                Y[(size_t)t * y_ld + r] = W.w_scale[r] * sa * v;
            }
        }
    });
}

void gemm_q8_fma(const WMat& W, const float* X, int T,
                 const float* bias, const float* resid, float* Y, int y_ld) {
    // iGPU route: int8 DP4A GEMM on the resident weights (falls through to CPU on any failure)
    if (W.gpu_slot >= 0 && !thread_cpu_only() && !bias && !resid && y_ld == W.rows && gpu::gemm(W.gpu_slot, X, T, Y)) return;
    if (W.wf.empty() && W.gpu_slot >= 0) {
        // GPU failed at runtime and f32 planes were skipped: fall back to on-the-fly dequant below
    }
    const int rows = W.rows, cols = W.cols, nb = W.nb;
    // resid may alias Y (in-place residual add): micro_tiles overwrites Y before the epilogue
    // reads resid, so snapshot it first (hot paths should ping-pong buffers instead)
    std::vector<float> resid_copy;
    if (resid && resid == Y) {
        resid_copy.resize((size_t)T * rows);
        for (int t = 0; t < T; ++t) memcpy(resid_copy.data() + (size_t)t * rows, Y + (size_t)t * y_ld, (size_t)rows * 4);
        resid = resid_copy.data();
    }
    // chunk rows so the dequantized tile stays L2-friendly (~256KB)
    static const int chunk_env = getenv("FERRO_CHUNK") ? atoi(getenv("FERRO_CHUNK")) : 0;
    int chunk = chunk_env ? chunk_env : std::max(8, std::min(64, (256 * 1024 / 4) / std::max(1, cols)));
    static thread_local AlignedF32 xt_all;
    xt_all.resize((size_t)cols * 8 * ((T + 7) / 8));
    transpose_x_tiles(X, T, cols, xt_all.p);
    float* const xtp = xt_all.p;
    Pool::get().parallel_for((rows + chunk - 1) / chunk, [&](int lo, int hi) {
        static thread_local AlignedF32 scratch;
        int r0 = lo * chunk, r1 = std::min(rows, hi * chunk);
        int n = r1 - r0;
        const float* wf;
        if (!W.wf.empty()) {
            wf = W.wf.data() + (size_t)r0 * cols;
        } else {
        scratch.resize((size_t)n * cols);
        wf = scratch.p;
        const int8_t* wq = W.wq.data() + (size_t)r0 * cols;
        const float* ws = W.ws.data() + (size_t)r0 * nb;
        for (int r = 0; r < n; ++r) {
            const int8_t* src = wq + (size_t)r * cols;
            float* dst = scratch.p + (size_t)r * cols;
            const float* srow = ws + (size_t)r * nb;
            for (int b = 0; b < nb; ++b) {
                __m256 s = _mm256_set1_ps(srow[b]);
                int c0 = b * QK8_0;
                for (int i = 0; i < QK8_0; i += 8) {
                    __m128i i8 = _mm_loadl_epi64((const __m128i*)(src + c0 + i));
                    __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8));
                    _mm256_store_ps(dst + c0 + i, _mm256_mul_ps(f, s));
                }
            }
        }
        }
        micro_tiles(wf, n, cols, X, T, Y + r0, y_ld, xtp);
        if (resid || bias) {
            for (int t = 0; t < T; ++t) {
                float* y = Y + (size_t)t * y_ld;
                const float* res = resid ? resid + (size_t)t * rows : nullptr;
                for (int r = r0; r < r1; ++r) {
                    float v = y[r];
                    if (res) v += res[r];
                    if (bias) v += bias[r];
                    y[r] = v;
                }
            }
        }
    });
}

// ---------------- RMSNorm ----------------
static inline void rms_row(const float* __restrict__ x, int cols, const float* __restrict__ w,
                           float eps, float* __restrict__ y) {
    __m256 sum = _mm256_setzero_ps();
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
        __m256 v = _mm256_loadu_ps(x + c);
        sum = _mm256_fmadd_ps(v, v, sum);
    }
    float ss = hsum256(sum);
    for (; c < cols; ++c) ss += x[c] * x[c];
    float inv1 = 1.0f / std::sqrt(ss / cols + eps);
    __m256 inv = _mm256_set1_ps(inv1);
    c = 0;
    for (; c + 8 <= cols; c += 8) {
        __m256 v = _mm256_loadu_ps(x + c);
        __m256 o = _mm256_mul_ps(_mm256_mul_ps(v, inv),
                                 w ? _mm256_loadu_ps(w + c) : _mm256_set1_ps(1.0f));
        _mm256_storeu_ps(y + c, o);
    }
    for (; c < cols; ++c) y[c] = x[c] * inv1 * (w ? w[c] : 1.0f);
}
void rmsnorm_q_fma(const float* x, int T, int cols, const float* w, float eps, float* y, int ld_x, int ld_y) {
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) rms_row(x + (size_t)t * ld_x, cols, w, eps, y + (size_t)t * ld_y);
    });
}
void rmsnorm_w1_fma(const float* x, int T, int cols, float eps, float* y, int ld_x, int ld_y) {
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) rms_row(x + (size_t)t * ld_x, cols, nullptr, eps, y + (size_t)t * ld_y);
    });
}

// ---------------- gelu_mul over fused gate/up block (exact erf GELU, matches ggml gelu) ----------------
static inline float gelu1(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f)); }

// Vectorized erf-gelu: Abramowitz-Stegun 7.1.26 via exp; |err| <= ~1.5e-7 (erf) per element.
static inline __m256 gelu1_v8(__m256 xv) {
    __m256 c1 = _mm256_set1_ps(0.70710678118654752440f);
    __m256 z = _mm256_mul_ps(xv, c1);
    // erf(z) = sign(z) * (1 - p(t)), t = 1/(1+0.3275911|z|), p = polynomial in t * exp(-z^2)
    __m256 az = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), z);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 t = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(_mm256_set1_ps(0.3275911f), az)));
    // A&S 7.1.26 coefficients
    __m256 p = _mm256_set1_ps(1.061405429f);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.453152027f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421413741f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284496736f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254829592f));
    p = _mm256_mul_ps(p, t);
    // exp(-z^2): exp2(-z^2 * log2e), degree-9 poly on 2^f after range reduction
    __m256 zz = _mm256_mul_ps(z, z);
    __m256 k = _mm256_round_ps(_mm256_mul_ps(zz, _mm256_set1_ps(-1.44269504088896341f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_fmadd_ps(k, _mm256_set1_ps(-0.69314718055994531f), _mm256_mul_ps(zz, _mm256_set1_ps(-1.0f)));
    __m256 ep = _mm256_set1_ps(1.388888888888889e-3f);
    ep = _mm256_fmadd_ps(ep, f, _mm256_set1_ps(8.333333333333333e-3f));
    ep = _mm256_fmadd_ps(ep, f, _mm256_set1_ps(4.166666666666666e-2f));
    ep = _mm256_fmadd_ps(ep, f, _mm256_set1_ps(1.666666666666667e-1f));
    ep = _mm256_fmadd_ps(ep, f, _mm256_set1_ps(5.0e-1f));
    ep = _mm256_fmadd_ps(ep, f, one);            // c1 = 1
    ep = _mm256_fmadd_ps(ep, f, one);            // c0 = 1
    __m256i ki = _mm256_cvtps_epi32(k);
    ki = _mm256_max_epi32(ki, _mm256_set1_epi32(-126));   // exp underflow -> 0, no garbage bits
    __m256 two_k = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23));
    __m256 ez = _mm256_mul_ps(ep, two_k);
    __m256 erfv = _mm256_sub_ps(one, _mm256_mul_ps(p, ez));
    // erf is odd: copy sign of z onto |erf| result
    __m256 sgn = _mm256_and_ps(z, _mm256_set1_ps(-0.0f));
    erfv = _mm256_or_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), erfv), sgn);
    return _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(xv, _mm256_add_ps(one, erfv)));
}
void gelu_fma(float* x, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, gelu1_v8(_mm256_loadu_ps(x + i)));
    for (; i < n; ++i) { float g = x[i]; x[i] = 0.5f * g * (1.0f + (float)erf((double)g * 0.70710678118654752440)); }
}

void gelu_mulvec_fma(float* x, const float* h, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(gelu1_v8(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(h + i)));
    for (; i < n; ++i) { float g = x[i]; x[i] = 0.5f * g * (1.0f + (float)erf((double)g * 0.70710678118654752440)) * h[i]; }
}

void gemm_gateup_fused(const WMat& W, const float* X, int T, float* Y, int FF) {
    // W: rows=2*FF interleaved (gate_r, up_r); Y: T x FF = gelu*up
    const int cols = W.cols;
    static thread_local AlignedF32 xt_all;
    xt_all.resize((size_t)cols * 8 * ((T + 7) / 8));
    transpose_x_tiles(X, T, cols, xt_all.p);
    float* const xtp = xt_all.p;
    const float* const Wf = W.wf.data();
    Pool::get().parallel_for(FF, [&](int lo, int hi) {
        int r0 = lo * 2, r1 = std::min(2 * FF, hi * 2);   // W-row range, even
        for (int t0 = 0; t0 < T; t0 += 16) {
            int nt = std::min(16, T - t0);
            int nh = (nt + 7) / 8;
            const float* xa = xtp + (size_t)(t0 / 8) * (size_t)cols * 8;
            const float* xb = xa + (size_t)cols * 8;
            for (int r = r0; r + 4 <= r1; r += 4) {
                __m256 acc[4][2];
                for (int i = 0; i < 4; ++i) { acc[i][0] = _mm256_setzero_ps(); acc[i][1] = _mm256_setzero_ps(); }
                const float* w = Wf + (size_t)r * cols;
                if (nh > 1) {
                    for (int c = 0; c < cols; ++c) {
                        __m256 b0 = _mm256_load_ps(xa + c * 8), b1 = _mm256_load_ps(xb + c * 8);
                        for (int i = 0; i < 4; ++i) {
                            __m256 wv = _mm256_broadcast_ss(w + (size_t)i * cols + c);
                            acc[i][0] = _mm256_fmadd_ps(wv, b0, acc[i][0]);
                            acc[i][1] = _mm256_fmadd_ps(wv, b1, acc[i][1]);
                        }
                    }
                } else {
                    for (int c = 0; c < cols; ++c) {
                        __m256 b0 = _mm256_load_ps(xa + c * 8);
                        for (int i = 0; i < 4; ++i)
                            acc[i][0] = _mm256_fmadd_ps(_mm256_broadcast_ss(w + (size_t)i * cols + c), b0, acc[i][0]);
                    }
                }
                alignas(32) float Ct[4][16];
                for (int i = 0; i < 4; ++i) {
                    _mm256_store_ps(Ct[i], acc[i][0]);
                    if (nh > 1) _mm256_store_ps(Ct[i] + 8, acc[i][1]);
                }
                // epilogue: F[t][r/2] = gelu(gate_r) * up_r
                __m256 gv0 = gelu1_v8(_mm256_load_ps(Ct[0]));  _mm256_store_ps(Ct[0], gv0);
                __m256 gv1 = gelu1_v8(_mm256_load_ps(Ct[2]));  _mm256_store_ps(Ct[2], gv1);
                if (nh > 1) {
                    __m256 gv0b = gelu1_v8(_mm256_load_ps(Ct[0] + 8)); _mm256_store_ps(Ct[0] + 8, gv0b);
                    __m256 gv1b = gelu1_v8(_mm256_load_ps(Ct[2] + 8)); _mm256_store_ps(Ct[2] + 8, gv1b);
                }
                for (int j = 0; j < nt; ++j) {
                    Y[(size_t)(t0 + j) * FF + r / 2]     = Ct[0][j] * Ct[1][j];
                    Y[(size_t)(t0 + j) * FF + r / 2 + 1] = Ct[2][j] * Ct[3][j];
                }
            }
        }
        for (int r = r0 + ((r1 - r0) / 4) * 4; r < r1; r += 2) {   // leftover pair(s)
            const float* wg = Wf + (size_t)r * cols;
            const float* wu = Wf + (size_t)(r + 1) * cols;
            for (int t = 0; t < T; ++t) {
                const float* xr = X + (size_t)t * cols;
                float sg = 0, su = 0;
                for (int c = 0; c < cols; ++c) { sg += wg[c] * xr[c]; su += wu[c] * xr[c]; }
                float g = sg;
                Y[(size_t)t * FF + r / 2] = 0.5f * g * (1.0f + (float)erf((double)g * 0.70710678118654752440)) * su;
            }
        }
    });
}

void gelu_mul_block_inter_fma(const float* G, int T, int FF, float* Y) {
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const float* g = G + (size_t)t * 2 * FF;
            float* yy = Y + (size_t)t * FF;
            int r = 0;
            alignas(32) float tmp[8], gt[8];
            for (; r + 8 <= FF; r += 8) {
                for (int j = 0; j < 8; ++j) tmp[j] = g[2 * (r + j)];
                _mm256_store_ps(gt, gelu1_v8(_mm256_load_ps(tmp)));
                for (int j = 0; j < 8; ++j) yy[r + j] = gt[j] * g[2 * (r + j) + 1];
            }
            for (; r < FF; ++r) { float gg = g[2 * r];
                yy[r] = 0.5f * gg * (1.0f + (float)erf((double)gg * 0.70710678118654752440)) * g[2 * r + 1]; }
        }
    });
}

void gelu_mul_block_fma(const float* gu, int T, int cols, float* y) {
    const int ldg = 2 * cols;
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const float* up = gu + (size_t)t * ldg;
            const float* gt = up + cols;
            float* yy = y + (size_t)t * cols;
            int i = 0;
            for (; i + 8 <= cols; i += 8) {
                __m256 g = _mm256_loadu_ps(up + i);
                __m256 u = _mm256_loadu_ps(gt + i);
                _mm256_storeu_ps(yy + i, _mm256_mul_ps(gelu1_v8(g), u));
            }
            for (; i < cols; ++i) yy[i] = gelu1(up[i]) * gt[i];
        }
    });
}

// ---------------- rope (GPT-J adjacent pairs) ----------------
void rope_apply_fma(float* x, int T, int heads, int head_dim, const RopeTable& rt, const int* pos, int ld) {
    int half = rt.rot / 2;
    for (int t = 0; t < T; ++t) {
        const float* cb = rt.cosb.data() + (size_t)pos[t] * half;
        const float* sb = rt.sinb.data() + (size_t)pos[t] * half;
        for (int h = 0; h < heads; ++h) {
            float* v = x + (size_t)t * ld + h * head_dim;
            for (int i = 0; i < half; i += 4) {
                __m128 c = _mm_setr_ps(cb[i], cb[i+1], cb[i+2], cb[i+3]);
                __m128 s = _mm_setr_ps(sb[i], sb[i+1], sb[i+2], sb[i+3]);
                __m128 x1 = _mm_loadu_ps(v + i);          // first half
                __m128 x2 = _mm_loadu_ps(v + i + half);   // second half (rotate_half pairs)
                __m128 re = _mm_sub_ps(_mm_mul_ps(x1, c), _mm_mul_ps(x2, s));
                __m128 ro = _mm_add_ps(_mm_mul_ps(x2, c), _mm_mul_ps(x1, s));
                _mm_storeu_ps(v + i, re);
                _mm_storeu_ps(v + i + half, ro);
            }
        }
    }
}

// ---------------- attention (bidirectional, segment-masked, windowed) ----------------
void gather_embd_fma(const WMat& tok, const int* ids, int T, float scale_mul, float* y) {
    const int cols = tok.cols, nb = tok.nb;
    const int8_t* const wqp = tok.wq.data();
    const float* const wsp = tok.ws.data();
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const int id = ids[t];
            const int8_t* wq = wqp + (size_t)id * cols;
            const float* ws = wsp + (size_t)id * nb;
            float* yt = y + (size_t)t * cols;
            for (int b = 0; b < nb; ++b) {
                __m256 s = _mm256_set1_ps(ws[b] * scale_mul);
                int c0 = b * QK8_0;
                int c1 = std::min(c0 + QK8_0, cols);
                for (int c = c0; c + 8 <= c1; c += 8) {
                    __m128i i8 = _mm_loadl_epi64((const __m128i*)(wq + c));
                    _mm256_storeu_ps(yt + c, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8)), s));
                }
                for (int c = c0 + ((c1 - c0) / 8) * 8; c < c1; ++c) yt[c] = (float)wq[c] * scale_mul * ws[b];
            }
        }
    });
}

void attention_fma(const float* Q, const float* K, const float* V, float* O,
                   int T, int nq, int nkv, int hd, float scale,
                   const int* seg, int window, int ldq, int ldk, int ldo) {
    int g = nq / nkv;
    Pool::get().parallel_for(nq * T, [&](int lo, int hi) {
        static thread_local std::vector<float> scores;
        scores.resize(T);
        for (int idx = lo; idx < hi; ++idx) {
            int h = idx / T, t = idx % T;
            int kvh = h / g;
            const float* qt = Q + (size_t)t * ldq + h * hd;
            int seg_t = seg[t];
            int lo_s = t, hi_s = t;
            while (lo_s > 0 && seg[lo_s - 1] == seg_t) --lo_s;
            while (hi_s + 1 < T && seg[hi_s + 1] == seg_t) ++hi_s;
            float maxs = -1e30f;
            for (int s = lo_s; s <= hi_s; ++s) {
                if (window > 0 && std::abs(t - s) >= window) { scores[s] = -1e30f; continue; }
                const float* ks = K + (size_t)s * ldk + kvh * hd;
                __m256 sum = _mm256_setzero_ps();
                for (int i = 0; i + 8 <= hd; i += 8)
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(qt + i), _mm256_loadu_ps(ks + i), sum);
                float d = hsum256(sum) * scale;
                scores[s] = d;
                maxs = std::max(maxs, d);
            }
            float sum = 0;
            for (int s = lo_s; s <= hi_s; ++s) {
                float e = scores[s] < -1e29f ? 0.f : std::exp(scores[s] - maxs);
                scores[s] = e; sum += e;
            }
            float inv = sum > 0 ? 1.0f / sum : 0.f;
            float* ot = O + (size_t)t * ldo + h * hd;
            int s0 = lo_s;
            while (s0 <= hi_s && scores[s0] == 0.f) ++s0;
            int s1 = hi_s;
            while (s1 >= s0 && scores[s1] == 0.f) --s1;
            for (int s = s0; s <= s1; ++s) scores[s] *= inv;
            for (int i0 = 0; i0 + 8 <= hd; i0 += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (int s = s0; s <= s1; ++s)
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(scores[s]),
                                          _mm256_loadu_ps(V + (size_t)s * ldk + kvh * hd + i0), acc);
                _mm256_storeu_ps(ot + i0, acc);
            }
            for (int i = (hd / 8) * 8; i < hd; ++i) {
                float a = 0;
                for (int s = s0; s <= s1; ++s) a += scores[s] * V[(size_t)s * ldk + kvh * hd + i];
                ot[i] = a;
            }
        }
    });
}

} // namespace ferro
