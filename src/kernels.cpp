// kernels.cpp — packing + scalar reference kernels + ISA dispatch
#include "kernels.h"
#include <immintrin.h>
#include "threadpool.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace ferro {

static thread_local bool g_tl_cpu_only = false;
void set_thread_cpu_only(bool on) { g_tl_cpu_only = on; }
bool thread_cpu_only() { return g_tl_cpu_only; }

// ---------- ISA detection ----------
static int g_isa = -1;
static void detect() {
    if (g_isa >= 0) return;
    __builtin_cpu_init();
    int l = 0;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) l = 1;
    if (l && __builtin_cpu_supports("avxvnni")) l = 2;
    g_isa = l;
}
int isa_level() { detect(); return g_isa; }
void set_isa_level(int l) { detect(); g_isa = l; }

// forward decls of FMA fast paths (kernels_fma.cpp)
void rmsnorm_q_fma(const float*, int, int, const float*, float, float*, int, int);
void rmsnorm_w1_fma(const float*, int, int, float, float*, int, int);
void gelu_mul_block_fma(const float*, int, int, float*);
void gelu_mulvec_fma(float*, const float*, int);
void rope_apply_fma(float*, int, int, int, const RopeTable&, const int*, int);
void attention_fma(const float*, const float*, const float*, float*, int, int, int, int, float, const int*, int, int, int, int);
void gemm_f32_micro(const float* Wf, int rows, int cols, const float* X, int T, float* Y, int y_ld);

// ---------- activation quantization (int8 path) ----------
void ActQ::ensure(int t, int c) {
    T = t; cols = c; nb = (c + QK8_0 - 1) / QK8_0;
    qx.assign((size_t)t * c, 0);
    sx.assign((size_t)t * nb, 1.0f);
}

void quant_act(const float* x, int T, int cols, ActQ& out) {
    out.ensure(T, cols);
    int nb = out.nb;
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t)t * cols;
        uint8_t* qr = out.qx.data() + (size_t)t * cols;
        float* sr = out.sx.data() + (size_t)t * nb;
        for (int b = 0; b < nb; ++b) {
            int c0 = b * QK8_0, c1 = std::min(c0 + QK8_0, cols);
            float maxabs = 0;
            for (int c = c0; c < c1; ++c) maxabs = std::max(maxabs, std::fabs(xr[c]));
            float s = maxabs > 1e-30f ? maxabs / 127.0f : 1.0f;
            sr[b] = s;
            float inv = 1.0f / s;
            for (int c = c0; c < c1; ++c) {
                int q = (int)std::lroundf(xr[c] * inv) + 128;
                qr[c] = (uint8_t)std::min(255, std::max(0, q));
            }
        }
    }
}

// ---------- weight packing ----------
WMat pack_q8(const uint8_t* raw, int rows, int cols, int row_stride_bytes, const char* name) {
    WMat W; W.rows = rows; W.cols = cols; W.nb = (cols + QK8_0 - 1) / QK8_0; W.name = name;
    W.wq.assign((size_t)rows * cols, 0);
    W.ws.assign((size_t)rows * W.nb, 0.f);
    W.rs.assign((size_t)rows * W.nb, 0.f);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* src = raw + (size_t)r * row_stride_bytes;
        int8_t* dst = W.wq.data() + (size_t)r * cols;
        for (int b = 0; b < W.nb; ++b) {
            const block_q8_0* blk = (const block_q8_0*)(src + (size_t)b * sizeof(block_q8_0));
            int c1 = std::min((b + 1) * QK8_0, cols);
            int n = c1 - b * QK8_0;
            float s = fp16_to_f32((uint16_t)blk->d);
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                dst[b * QK8_0 + i] = blk->qs[i];
                sum += blk->qs[i];
            }
            W.ws[(size_t)r * W.nb + b] = s;
            W.rs[(size_t)r * W.nb + b] = -128.0f * (float)sum * s;
        }
    }
    return W;
}

WMat pack_concat(std::vector<const WMat*> parts, const char* name) {
    int cols = parts[0]->cols, rows = 0;
    for (auto* p : parts) { rows += p->rows; }
    WMat W; W.rows = rows; W.cols = cols; W.nb = (cols + QK8_0 - 1) / QK8_0; W.name = name;
    W.wq.resize((size_t)rows * cols);
    W.ws.resize((size_t)rows * W.nb);
    W.rs.resize((size_t)rows * W.nb);
    if (!parts.empty() && !parts[0]->wf.empty()) W.wf.resize((size_t)rows * cols);
    int r0 = 0;
    for (auto* p : parts) {
        memcpy(W.wq.data() + (size_t)r0 * cols, p->wq.data(), (size_t)p->rows * cols);
        if (!p->wf.empty())
            memcpy(W.wf.data() + (size_t)r0 * cols, p->wf.data(), (size_t)p->rows * cols * sizeof(float));
        memcpy(W.ws.data() + (size_t)r0 * W.nb, p->ws.data(), (size_t)p->rows * W.nb * sizeof(float));
        memcpy(W.rs.data() + (size_t)r0 * W.nb, p->rs.data(), (size_t)p->rows * W.nb * sizeof(float));
        r0 += p->rows;
    }
    return W;
}

WMat pack_concat_inter(const WMat& a, const WMat& b, const char* name) {
    // rows alternate: 2r = a row r, 2r+1 = b row r (same cols, same packing)
    int cols = a.cols, rows = a.rows + b.rows;
    WMat W; W.rows = rows; W.cols = cols; W.nb = a.nb; W.name = name;
    W.wq.resize((size_t)rows * cols);
    W.ws.resize((size_t)rows * W.nb);
    W.rs.resize((size_t)rows * W.nb);
    bool has_wf = !a.wf.empty() && !b.wf.empty();
    if (has_wf) W.wf.resize((size_t)rows * cols);
    for (int r = 0; r < a.rows; ++r) {
        memcpy(W.wq.data() + (size_t)(2*r) * cols, a.wq.data() + (size_t)r * cols, (size_t)cols);
        memcpy(W.wq.data() + (size_t)(2*r+1) * cols, b.wq.data() + (size_t)r * cols, (size_t)cols);
        memcpy(W.ws.data() + (size_t)(2*r) * W.nb, a.ws.data() + (size_t)r * a.nb, (size_t)W.nb * sizeof(float));
        memcpy(W.ws.data() + (size_t)(2*r+1) * W.nb, b.ws.data() + (size_t)r * b.nb, (size_t)W.nb * sizeof(float));
        memcpy(W.rs.data() + (size_t)(2*r) * W.nb, a.rs.data() + (size_t)r * a.nb, (size_t)W.nb * sizeof(float));
        memcpy(W.rs.data() + (size_t)(2*r+1) * W.nb, b.rs.data() + (size_t)r * b.nb, (size_t)W.nb * sizeof(float));
        if (has_wf) {
            memcpy(W.wf.data() + (size_t)(2*r) * cols, a.wf.data() + (size_t)r * cols, (size_t)cols * sizeof(float));
            memcpy(W.wf.data() + (size_t)(2*r+1) * cols, b.wf.data() + (size_t)r * cols, (size_t)cols * sizeof(float));
        }
    }
    return W;
}

static float gelu1(float x);
void gelu_mul_block_inter_fma(const float* G, int T, int FF, float* Y);
void gelu_mul_block_inter(const float* G, int T, int FF, float* Y) {
    if (isa_level() >= 1) { gelu_mul_block_inter_fma(G, T, FF, Y); return; }
    // G[t][2r]=gate, G[t][2r+1]=up  ->  Y[t][r] = gelu(gate)*up
    for (int t = 0; t < T; ++t) {
        const float* g = G + (size_t)t * 2 * FF;
        float* yy = Y + (size_t)t * FF;
        for (int r = 0; r < FF; ++r) yy[r] = gelu1(g[2*r]) * g[2*r+1];
    }
}

WMat pack_from_f32(const float* src, int rows, int cols, int row_stride_floats, const char* name) {
    WMat W; W.rows = rows; W.cols = cols; W.nb = (cols + QK8_0 - 1) / QK8_0; W.name = name;
    W.wq.assign((size_t)rows * cols, 0);
    W.ws.assign((size_t)rows * W.nb, 0.f);
    W.rs.assign((size_t)rows * W.nb, 0.f);
    for (int r = 0; r < rows; ++r) {
        const float* srow = src + (size_t)r * row_stride_floats;
        for (int b = 0; b < W.nb; ++b) {
            int c0 = b * QK8_0, c1 = std::min(c0 + QK8_0, cols);
            float maxabs = 0;
            for (int c = c0; c < c1; ++c) maxabs = std::max(maxabs, std::fabs(srow[c]));
            float s = maxabs > 1e-30f ? maxabs / 127.0f : 1.0f;
            int sum = 0;
            for (int c = c0; c < c1; ++c) {
                int q = (int)std::lroundf(srow[c] / s);
                q = std::min(127, std::max(-128, q));
                W.wq[(size_t)r * cols + c] = (int8_t)q;
                sum += q;
            }
            W.ws[(size_t)r * W.nb + b] = s;
            W.rs[(size_t)r * W.nb + b] = -128.0f * (float)sum * s;
        }
    }
    return W;
}

// ---------- reference GEMMs ----------
void gemm_q8_ref(const WMat& W, const ActQ& X, int T, float* Y, int y_ld) {
    int nb = W.nb, cols = W.cols;
    for (int t = 0; t < T; ++t) {
        const uint8_t* xq = X.qx.data() + (size_t)t * cols;
        const float* xs = X.sx.data() + (size_t)t * nb;
        float* y = Y + (size_t)t * y_ld;
        for (int r = 0; r < W.rows; ++r) {
            const int8_t* wq = W.wq.data() + (size_t)r * cols;
            const float* ws = W.ws.data() + (size_t)r * nb;
            const float* rs = W.rs.data() + (size_t)r * nb;
            double acc = 0;
            for (int b = 0; b < nb; ++b) {
                int c1 = std::min((b + 1) * QK8_0, cols);
                int64_t dot = 0;
                for (int c = b * QK8_0; c < c1; ++c) dot += (int)xq[c] * (int)wq[c];   // u8 domain; rs folds bias
                acc += (double)xs[b] * ((double)ws[b] * (double)dot + (double)rs[b]);
            }
            y[r] = (float)acc;
        }
    }
}

void gemm_q8(const WMat& W, const ActQ& X, int T, float* Y, int y_ld) {
    gemm_q8_ref(W, X, T, Y, y_ld);   // int8 path: FMA impl registered later if faster
}

void gemm_q8_full(const WMat& W, const ActQ& X, int T,
                  const float* bias, const float* resid, float* Y, int y_ld) {
    gemm_q8_ref(W, X, T, Y, y_ld);
    for (int t = 0; t < T; ++t) {
        float* y = Y + (size_t)t * y_ld;
        const float* rs = resid ? resid + (size_t)t * W.rows : nullptr;
        for (int r = 0; r < W.rows; ++r)
            y[r] += (rs ? rs[r] : 0.f) + (bias ? bias[r] : 0.f);
    }
}

// scalar dequant-on-the-fly f32 GEMM (fallback when no AVX2)
void gemm_q8_fma_scalar(const WMat& W, const float* X, int T,
                        const float* bias, const float* resid, float* Y, int y_ld) {
    int nb = W.nb, cols = W.cols;
    for (int t = 0; t < T; ++t) {
        const float* x = X + (size_t)t * cols;
        float* y = Y + (size_t)t * y_ld;
        const float* res = resid ? resid + (size_t)t * W.rows : nullptr;
        for (int r = 0; r < W.rows; ++r) {
            const int8_t* wq = W.wq.data() + (size_t)r * cols;
            const float* ws = W.ws.data() + (size_t)r * nb;
            float acc = res ? res[r] : 0.f;
            for (int b = 0; b < nb; ++b) {
                float s = ws[b];
                int c1 = std::min((b + 1) * QK8_0, cols);
                float part = 0;
                for (int c = b * QK8_0; c < c1; ++c) part += (float)wq[c] * x[c];
                acc += s * part;
            }
            if (bias) acc += bias[r];
            y[r] = acc;
        }
    }
}

// f32 GEMM: Y[t][r] = sum_c W[r][c] * X[t][c]
void gemm_f32(const float* W, int rows, int cols, const float* X, int T, float* Y, int y_ld) {
    if (isa_level() >= 1) { gemm_f32_micro(W, rows, cols, X, T, Y, y_ld); return; }
    for (int t = 0; t < T; ++t) {
        const float* x = X + (size_t)t * cols;
        float* y = Y + (size_t)t * y_ld;
        for (int r = 0; r < rows; ++r) {
            const float* w = W + (size_t)r * cols;
            float acc = 0;
            for (int c = 0; c < cols; ++c) acc += w[c] * x[c];
            y[r] = acc;
        }
    }
}

// ---------- fused vector ops (scalar refs) ----------
static void rms_row(const float* x, int cols, const float* w, float eps, float* y) {
    double ss = 0;
    for (int c = 0; c < cols; ++c) ss += (double)x[c] * x[c];
    float inv = (float)(1.0 / std::sqrt(ss / cols + eps));
    for (int c = 0; c < cols; ++c) y[c] = x[c] * inv * (w ? w[c] : 1.0f);
}
void rmsnorm_q(const float* x, int T, int cols, const float* w, float eps, float* y, int ld_x, int ld_y) {
    if (!ld_x) ld_x = cols;
    if (!ld_y) ld_y = cols;
    if (isa_level() >= 1) { rmsnorm_q_fma(x, T, cols, w, eps, y, ld_x, ld_y); return; }
    for (int t = 0; t < T; ++t) rms_row(x + (size_t)t * ld_x, cols, w, eps, y + (size_t)t * ld_y);
}
void rmsnorm_w1(const float* x, int T, int cols, float eps, float* y, int ld_x, int ld_y) {
    if (!ld_x) ld_x = cols;
    if (!ld_y) ld_y = cols;
    if (isa_level() >= 1) { rmsnorm_w1_fma(x, T, cols, eps, y, ld_x, ld_y); return; }
    for (int t = 0; t < T; ++t) rms_row(x + (size_t)t * ld_x, cols, nullptr, eps, y + (size_t)t * ld_y);
}
void add_rows(const float* a, const float* b, int n, float* y) {
    for (int i = 0; i < n; ++i) y[i] = a[i] + b[i];
}
void add_scaled(const float* a, const float* b, float s, int n, float* y) {
    for (int i = 0; i < n; ++i) y[i] = a[i] + s * b[i];
}
void scale_rows(float s, int n, float* y) { for (int i = 0; i < n; ++i) y[i] *= s; }

// exact erf GELU (matches ggml gelu)
static inline float gelu1(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f)); }
void gelu_i(float* x, int n) {
    if (isa_level() >= 1) { gelu_fma(x, n); return; }
    for (int i = 0; i < n; ++i) x[i] = gelu1(x[i]);
}
void gelu_mulvec(float* x, const float* h, int n) {
    if (isa_level() >= 1) { gelu_mulvec_fma(x, h, n); return; }
    for (int i = 0; i < n; ++i) x[i] = gelu1(x[i]) * h[i];
}

void gelu_mul_block(const float* gu, int T, int cols, float* y) {
    if (isa_level() >= 1) { gelu_mul_block_fma(gu, T, cols, y); return; }
    int ldg = 2 * cols;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < cols; ++i)
            y[(size_t)t * cols + i] = gelu1(gu[(size_t)t * ldg + i]) * gu[(size_t)t * ldg + cols + i];
}

// ---------- rope ----------
void RopeTable::build(double freq_base, int r, int max_pos, double attn_factor) {
    rot = r;
    int half = rot / 2;
    cosb.resize((size_t)max_pos * half);
    sinb.resize((size_t)max_pos * half);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            double th = p / std::pow(freq_base, (2.0 * i) / rot);
            cosb[(size_t)p * half + i] = (float)(attn_factor * std::cos(th));
            sinb[(size_t)p * half + i] = (float)(attn_factor * std::sin(th));
        }
    }
}
void rope_apply(float* x, int T, int heads, int head_dim, const RopeTable& rt, const int* pos, int ld) {
    if (!ld) ld = heads * head_dim;
    if (isa_level() >= 1) { rope_apply_fma(x, T, heads, head_dim, rt, pos, ld); return; }
    int half = rt.rot / 2;
    for (int t = 0; t < T; ++t) {
        const float* cb = rt.cosb.data() + (size_t)pos[t] * half;
        const float* sb = rt.sinb.data() + (size_t)pos[t] * half;
        for (int h = 0; h < heads; ++h) {
            float* v = x + (size_t)t * ld + h * head_dim;
            for (int i = 0; i < half; ++i) {
                float x0 = v[i], x1 = v[i + half];   // llama.cpp rotate_half pairs
                v[i]        = x0 * cb[i] - x1 * sb[i];
                v[i + half] = x1 * cb[i] + x0 * sb[i];
            }
        }
    }
}

// ---------- attention (bidirectional, segment-masked, windowed) ----------
void attention(const float* Q, const float* K, const float* V, float* O,
               int T, int nq, int nkv, int hd, float scale,
               const int* seg, int window, int ldq, int ldk, int ldo) {
    if (isa_level() >= 1) { attention_fma(Q, K, V, O, T, nq, nkv, hd, scale, seg, window, ldq, ldk, ldo); return; }
    int g = nq / nkv;
    std::vector<float> scores(T), expv(T);
    for (int h = 0; h < nq; ++h) {
        int kvh = h / g;
        for (int t = 0; t < T; ++t) {
            const float* qt = Q + (size_t)t * ldq + h * hd;
            int seg_t = seg[t];
            // contiguous segments: expand bounds
            int lo = t, hi = t;
            while (lo > 0 && seg[lo - 1] == seg_t) --lo;
            while (hi + 1 < T && seg[hi + 1] == seg_t) ++hi;
            float maxs = -1e30f;
            for (int s = lo; s <= hi; ++s) {
                if (window > 0 && std::abs(t - s) >= window) { scores[s] = -1e30f; continue; }
                const float* ks = K + (size_t)s * ldk + kvh * hd;
                float d = 0; for (int i = 0; i < hd; ++i) d += qt[i] * ks[i];
                d *= scale;
                scores[s] = d;
                maxs = std::max(maxs, d);
            }
            for (int s = 0; s < lo; ++s) scores[s] = -1e30f;
            for (int s = hi + 1; s < T; ++s) scores[s] = -1e30f;
            float sum = 0;
            for (int s = lo; s <= hi; ++s) { float e = scores[s] < -1e29f ? 0.f : std::exp(scores[s] - maxs); expv[s] = e; sum += e; }
            float inv = sum > 0 ? 1.0f / sum : 0.f;
            float* ot = O + (size_t)t * ldo + h * hd;
            for (int i = 0; i < hd; ++i) ot[i] = 0;
            for (int s = lo; s <= hi; ++s) {
                float w = expv[s] * inv;
                if (w == 0.f) continue;
                const float* vs = V + (size_t)s * ldk + kvh * hd;
                for (int i = 0; i < hd; ++i) ot[i] += w * vs[i];
            }
        }
    }
}

void mean_pool(const float* x, int T, int cols, float* y, bool l2) {
    for (int c = 0; c < cols; ++c) y[c] = 0;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < cols; ++c) y[c] += x[(size_t)t * cols + c];
    float inv = 1.0f / T;
    for (int c = 0; c < cols; ++c) y[c] *= inv;
    if (l2) {
        double ss = 0; for (int c = 0; c < cols; ++c) ss += (double)y[c] * y[c];
        float invn = (float)(1.0 / (std::sqrt(ss) + 1e-12));
        for (int c = 0; c < cols; ++c) y[c] *= invn;
    }
}

void gather_embd_fma(const WMat& tok, const int* ids, int T, float scale_mul, float* y);
void gather_embd(const WMat& tok, const int* ids, int T, float scale_mul, float* y) {
    if (isa_level() >= 1) { gather_embd_fma(tok, ids, T, scale_mul, y); return; }
    int cols = tok.cols, nb = tok.nb;
    for (int t = 0; t < T; ++t) {
        int id = ids[t];
        const int8_t* wq = tok.wq.data() + (size_t)id * cols;
        const float* ws = tok.ws.data() + (size_t)id * nb;
        float* yt = y + (size_t)t * cols;
        for (int b = 0; b < nb; ++b) {
            float s = ws[b] * scale_mul;
            int c1 = std::min((b + 1) * QK8_0, cols);
            for (int c = b * QK8_0; c < c1; ++c) yt[c] = (float)wq[c] * s;
        }
    }
}

} // namespace ferro
