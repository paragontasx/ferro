// mm_gemma4a.cpp - Gemma 4 audio encoder (gemma4a conformer) for EmbeddingGemma 2: CPU reference.
// Mirrors tools/mtmd/models/gemma4a.cpp. Layout facts were pinned against ggml op by op (bench/mm/ggml_ops_ref.cpp,
// pin_sscp.py, pin_attn_conv.py): conv input x[frame][mel], freq-major flatten (feature = freq*32 + ch), fp16-rounded conv
// operands (ggml_conv_2d), blocked K/V windows (C = 12, P = 12, S = 24), blocked relative shift, causal depthwise conv k = 5.
#include "mm_gemma4_impl.h"
#include "threadpool.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>
#include <stdexcept>

namespace ferro {

namespace {
// fp16 round-trip (what ggml_conv_2d does to the kernel and the im2col patches before the matmul)
// F16C vcvtps2ph with imm 0 = round-to-nearest-even, the same rounding ggml uses for fp32 -> fp16
__attribute__((target("f16c"))) inline uint16_t f32_to_f16_rne(float x) { return (uint16_t)_cvtss_sh(x, 0); }
inline float to_h(float x) { return fp16_to_f32(f32_to_f16_rne(x)); }

void rms_rows(const float* x, int T, int d, const float* w, float eps, float* y) {   // y = x * rsqrt(mean(x^2)+eps) * (w ? w : 1)
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const float* a = x + (size_t)t * d; float* o = y + (size_t)t * d;
            double ss = 0; for (int i = 0; i < d; ++i) ss += (double)a[i] * a[i];
            const float s = 1.0f / std::sqrt((float)(ss / d) + eps);
            for (int i = 0; i < d; ++i) o[i] = a[i] * s * (w ? w[i] : 1.0f);
        }
    });
}
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace

void load_audio_weights(const GgufFile& g, Gemma4Proj::Impl& I) {
    I.a_d = (int)g.get_int("clip.audio.embedding_length");
    I.a_layers = (int)g.get_int("clip.audio.block_count");
    I.a_heads = (int)g.get_int("clip.audio.attention.head_count");
    I.a_ffn = (int)g.get_int("clip.audio.feed_forward_length");
    I.a_mel = (int)g.get_int("clip.audio.num_mel_bins", 128);
    I.a_hd = I.a_d / I.a_heads;
    for (int i = 0; i < 2; ++i) {
        const std::string b = "a.conv1d." + std::to_string(i);
        I.a_conv_w[i] = tf32(g, b + ".weight");
        I.a_conv_norm[i] = tf32(g, b + ".norm.weight");
        for (float& v : I.a_conv_w[i]) v = to_h(v);                  // ggml_conv_2d converts the kernel to fp16
    }
    I.a_inp_proj = tf32(g, "a.input_projection.weight");
    I.a_out_proj = clin(g, "a.pre_encode.out.weight");
    I.a_out_bias = tf32(g, "a.pre_encode.out.bias");
    I.a_mm_proj = clin(g, "mm.a.input_projection.weight");
    I.al.resize(I.a_layers);
    for (int l = 0; l < I.a_layers; ++l) {
        const std::string b = "a.blk." + std::to_string(l) + ".";
        auto& L = I.al[l];
        L.ffn_norm = tf32(g, b + "ffn_norm.weight");        L.ffn_post_norm = tf32(g, b + "ffn_post_norm.weight");
        L.ffn_norm1 = tf32(g, b + "ffn_norm_1.weight");     L.ffn_post_norm1 = tf32(g, b + "ffn_post_norm_1.weight");
        L.attn_pre_norm = tf32(g, b + "attn_pre_norm.weight"); L.attn_post_norm = tf32(g, b + "attn_post_norm.weight");
        // GGUF names are crossed on purpose (clip.cpp): conv_norm.weight = PRE-conv norm, norm_conv.weight = norm after the dw conv
        L.norm_conv = tf32(g, b + "conv_norm.weight");      L.conv_norm = tf32(g, b + "norm_conv.weight");
        L.ln2 = tf32(g, b + "ln2.weight");                  L.per_dim_scale = tf32(g, b + "per_dim_scale.weight");
        L.conv_dw = tf32(g, b + "conv_dw.weight");          // [1024][5]
        L.ffn_up = clin(g, b + "ffn_up.weight");   L.ffn_down = clin(g, b + "ffn_down.weight");
        L.ffn_up1 = clin(g, b + "ffn_up_1.weight"); L.ffn_down1 = clin(g, b + "ffn_down_1.weight");
        L.q = clin(g, b + "attn_q.weight"); L.k = clin(g, b + "attn_k.weight"); L.v = clin(g, b + "attn_v.weight");
        L.o = clin(g, b + "attn_out.weight"); L.k_rel = clin(g, b + "attn_k_rel.weight");
        L.pw1 = clin(g, b + "conv_pw1.weight"); L.pw2 = clin(g, b + "conv_pw2.weight");
    }
}

// ---- SSCP: two (conv3x3 s2 p1 -> LayerNorm over channels -> ReLU), flatten freq-major, input projection
// mel.data is [mel][frame]; the conv input is x[frame][mel]. Output [n_pos][a_d].
static std::vector<float> sscp(const Gemma4Proj::Impl& I, const Mel& mel, int& n_pos) {
    const int F0 = mel.n_mel, T0 = mel.n_len;
    std::vector<float> x((size_t)T0 * F0);
    for (int t = 0; t < T0; ++t) for (int f = 0; f < F0; ++f) x[(size_t)t * F0 + f] = to_h(mel.data[(size_t)f * T0 + t]);
    int C = 1, H = T0, W = F0;                                   // [C][H=time][W=mel]
    std::vector<float> cur = std::move(x);
    const int couts[2] = {128, 32};
    for (int li = 0; li < 2; ++li) {
        const int Co = couts[li], Ho = (H - 1) / 2 + 1, Wo = (W - 1) / 2 + 1;
        // im2col [Ho*Wo][C*9] (patch order c, kh, kw = the weight's [Co][C][3][3] order) then one GEMM: out[pos][Co]
        const int K = C * 9, NP = Ho * Wo;
        std::vector<float> col((size_t)NP * K), gem((size_t)NP * Co);
        Pool::get().parallel_for(Ho, [&](int lo, int hi) {
            for (int y = lo; y < hi; ++y)
                for (int xx = 0; xx < Wo; ++xx) {
                    float* o = col.data() + ((size_t)y * Wo + xx) * K;
                    for (int c = 0; c < C; ++c)
                        for (int kh = 0; kh < 3; ++kh) {
                            const int iy = 2 * y - 1 + kh;
                            for (int kw = 0; kw < 3; ++kw) {
                                const int ix = 2 * xx - 1 + kw;
                                *o++ = (iy < 0 || iy >= H || ix < 0 || ix >= W) ? 0.0f : cur[((size_t)c * H + iy) * W + ix];
                            }
                        }
                }
        });
        gemm_f32(I.a_conv_w[li].data(), Co, K, col.data(), NP, gem.data(), Co);
        std::vector<float> out((size_t)Co * NP);                 // back to [Co][Ho][Wo]
        Pool::get().parallel_for(Co, [&](int lo, int hi) {
            for (int o = lo; o < hi; ++o) for (int p = 0; p < NP; ++p) out[(size_t)o * NP + p] = gem[(size_t)p * Co + o];
        });
        // LayerNorm over channels at every (y, x): weight, no bias, eps 1e-6; then ReLU; round for the next conv's im2col
        const float* nw = I.a_conv_norm[li].data();
        Pool::get().parallel_for(Ho, [&](int lo, int hi) {
            std::vector<float> v(Co);
            for (int y = lo; y < hi; ++y)
                for (int xx = 0; xx < Wo; ++xx) {
                    double mu = 0; for (int o = 0; o < Co; ++o) { v[o] = out[((size_t)o * Ho + y) * Wo + xx]; mu += v[o]; }
                    mu /= Co; double var = 0; for (int o = 0; o < Co; ++o) { const double d = v[o] - mu; var += d * d; }
                    var /= Co; const float inv = 1.0f / std::sqrt((float)var + 1e-6f);
                    for (int o = 0; o < Co; ++o) {
                        float r = (float)((v[o] - mu) * inv) * nw[o];
                        r = r > 0 ? r : 0;
                        out[((size_t)o * Ho + y) * Wo + xx] = (li == 0) ? to_h(r) : r;
                    }
                }
        });
        cur = std::move(out); C = Co; H = Ho; W = Wo;
    }
    n_pos = H;                                                   // time
    const int feat = W * C;                                      // freq-major flatten: feature = freq * C + ch
    std::vector<float> flat((size_t)n_pos * feat);
    for (int t = 0; t < n_pos; ++t) for (int f = 0; f < W; ++f) for (int c = 0; c < C; ++c)
        flat[(size_t)t * feat + (size_t)f * C + c] = cur[((size_t)c * H + t) * W + f];
    std::vector<float> proj((size_t)n_pos * I.a_d);
    gemm_f32(I.a_inp_proj.data(), I.a_d, feat, flat.data(), n_pos, proj.data(), I.a_d);
    return proj;
}

std::vector<float> Gemma4Proj::encode_audio(const Mel& mel) {
    if (!a_ok_) throw std::runtime_error("mmproj has no gemma4a audio encoder");
    Impl& I = *impl_;
    int N = 0;
    static const bool tm = getenv("G4A_TIME") != nullptr;     // phase timing (ms): sscp | ffn | attention | conv module | tail
    double tp[5] = {0, 0, 0, 0, 0}; auto t0 = std::chrono::steady_clock::now();
    auto lap = [&](int k) { if (!tm) return; auto t = std::chrono::steady_clock::now(); tp[k] += std::chrono::duration<double, std::milli>(t - t0).count(); t0 = t; };
    std::vector<float> X = sscp(I, mel, N);                      // [N][d]
    lap(0);
    const int d = I.a_d, H = I.a_heads, hd = I.a_hd, ffn = I.a_ffn;
    const float eps = 1e-6f;
    constexpr int C = 12, P = 12, S = C + P, R = P + 1;
    const int B = (N + C - 1) / C;
    static const bool pds_sp = getenv("FERRO_G4A_PDS") && !strcmp(getenv("FERRO_G4A_PDS"), "softplus");
    // sinusoidal relative positions [R][d]: position = P - p, sin | cos halves
    std::vector<float> pos_emb((size_t)R * d);
    {
        const int nts = d / 2; const float inc = std::log(10000.0f) / std::max(nts - 1, 1);
        for (int p = 0; p < R; ++p) for (int i = 0; i < nts; ++i) {
            const float sc = (float)(P - p) * std::exp(-(float)i * inc);
            pos_emb[(size_t)p * d + i] = std::sin(sc); pos_emb[(size_t)p * d + i + nts] = std::cos(sc);
        }
    }
    std::vector<float> scratch, xn((size_t)N * d), F((size_t)N * ffn), Y((size_t)N * d), Q((size_t)N * d), K((size_t)N * d), V((size_t)N * d), O((size_t)N * d);
    auto ffn_block = [&](const CLin& up, const CLin& down, const std::vector<float>& nw, const std::vector<float>& pw) {
        rms_rows(X.data(), N, d, nw.data(), eps, xn.data());
        capply(up, xn.data(), N, F.data(), scratch);
        for (size_t i = 0; i < F.size(); ++i) F[i] = silu(F[i]);
        capply(down, F.data(), N, Y.data(), scratch);
        rms_rows(Y.data(), N, d, pw.data(), eps, Y.data());
        for (size_t i = 0; i < (size_t)N * d; ++i) X[i] += 0.5f * Y[i];            // half-step residual
    };
    for (int l = 0; l < I.a_layers; ++l) {
        auto& L = I.al[l];
        ffn_block(L.ffn_up, L.ffn_down, L.ffn_norm, L.ffn_post_norm);
        lap(1);
        // ---- chunked local self-attention with relative position bias
        rms_rows(X.data(), N, d, L.attn_pre_norm.data(), eps, xn.data());
        capply(L.q, xn.data(), N, Q.data(), scratch); capply(L.k, xn.data(), N, K.data(), scratch); capply(L.v, xn.data(), N, V.data(), scratch);
        const float q_scale = (1.0f / std::sqrt((float)hd)) / std::log(2.0f);
        const float k_scale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
        std::vector<float> pds(hd);
        for (int e = 0; e < hd; ++e) pds[e] = pds_sp ? std::log1p(std::exp(L.per_dim_scale[e])) : L.per_dim_scale[e];
        for (int t = 0; t < N; ++t) for (int h = 0; h < H; ++h) for (int e = 0; e < hd; ++e) {
            Q[(size_t)t * d + h * hd + e] *= q_scale * pds[e];
            K[(size_t)t * d + h * hd + e] *= k_scale;
        }
        std::vector<float> Pk((size_t)R * d);                     // pos_emb projected through attn_k_rel (a plain linear, no clamp here)
        capply(L.k_rel, pos_emb.data(), R, Pk.data(), scratch);
        // ---- per (head, block): content scores over the 24-wide key window + the blocked relative term
        //   matrix_ac[q][s]  = Q[b*C+q] . K[b*C - P + s]                 (zero key outside [0, N) -> score 0, masked below)
        //   matrix_bd_raw[q][r] = Q[b*C+q] . Pk[r]                        (r = 0..R-1)
        //   blocked shift (pinned vs ggml): pad each row to S+1, flatten the C rows, keep C*S, reshape [C][S]  => bd[q][s]
        //   scores = softcap(ac + bd) + mask ; softmax over s ; out = attn . V window
        std::fill(O.begin(), O.end(), 0.0f);
        Pool::get().parallel_for(H * B, [&](int lo, int hi) {
            std::vector<float> bdraw((size_t)C * R), bdflat((size_t)C * (S + 1)), sc((size_t)C * S);
            for (int idx = lo; idx < hi; ++idx) {
                const int h = idx / B, b = idx % B;
                for (int q = 0; q < C; ++q) {
                    const int gq = b * C + q;
                    for (int r = 0; r < R; ++r) {
                        float a = 0;
                        if (gq < N) { const float* qv = Q.data() + (size_t)gq * d + h * hd; const float* pk = Pk.data() + (size_t)r * d + h * hd; for (int e = 0; e < hd; ++e) a += qv[e] * pk[e]; }
                        bdraw[(size_t)q * R + r] = a;
                    }
                }
                std::fill(bdflat.begin(), bdflat.end(), 0.0f);
                for (int q = 0; q < C; ++q) for (int r = 0; r < R; ++r) bdflat[(size_t)q * (S + 1) + r] = bdraw[(size_t)q * R + r];
                for (int q = 0; q < C; ++q) {
                    const int gq = b * C + q;
                    float mx = -FLT_MAX;
                    for (int s = 0; s < S; ++s) {
                        const int gk = b * C - P + s;
                        float v;
                        if (gq < N && gk >= 0 && gk < N && gk <= gq && (gq - gk) < P) {
                            float ac = 0; const float* qv = Q.data() + (size_t)gq * d + h * hd; const float* kv = K.data() + (size_t)gk * d + h * hd;
                            for (int e = 0; e < hd; ++e) ac += qv[e] * kv[e];
                            const float bd = bdflat[(size_t)q * S + s];                      // flat index q*S + s of the padded/flattened matrix
                            v = 50.0f * std::tanh((ac + bd) / 50.0f);                          // softcap
                        } else v = -1e9f;                                                      // blocked causal mask
                        sc[(size_t)q * S + s] = v; mx = std::max(mx, v);
                    }
                    if (gq >= N) continue;
                    float den = 0; for (int s = 0; s < S; ++s) { sc[(size_t)q * S + s] = std::exp(sc[(size_t)q * S + s] - mx); den += sc[(size_t)q * S + s]; }
                    float* ov = O.data() + (size_t)gq * d + h * hd;
                    for (int s = 0; s < S; ++s) {
                        const float wgt = sc[(size_t)q * S + s] / den; if (wgt == 0.0f) continue;
                        const int gk = b * C - P + s; if (gk < 0 || gk >= N) continue;
                        const float* vv = V.data() + (size_t)gk * d + h * hd;
                        for (int e = 0; e < hd; ++e) ov[e] += wgt * vv[e];
                    }
                }
            }
        });
        capply(L.o, O.data(), N, Y.data(), scratch);
        rms_rows(Y.data(), N, d, L.attn_post_norm.data(), eps, Y.data());
        for (size_t i = 0; i < (size_t)N * d; ++i) X[i] += Y[i];
        lap(2);
        // ---- convolution module: RMS norm -> pw1 -> GLU -> causal depthwise conv (k = 5) -> RMS norm -> SiLU -> pw2
        rms_rows(X.data(), N, d, L.norm_conv.data(), eps, xn.data());
        std::vector<float> G2((size_t)N * 2 * d), Gl((size_t)N * d), Dw((size_t)N * d);
        capply(L.pw1, xn.data(), N, G2.data(), scratch);
        for (int t = 0; t < N; ++t) for (int c = 0; c < d; ++c) Gl[(size_t)t * d + c] = G2[(size_t)t * 2 * d + c] * sigmoidf(G2[(size_t)t * 2 * d + d + c]);
        const int KS = 5;
        for (int t = 0; t < N; ++t) for (int c = 0; c < d; ++c) {
            float acc = 0;
            for (int k = 0; k < KS; ++k) { const int src = t - (KS - 1) + k; if (src >= 0) acc += L.conv_dw[(size_t)c * KS + k] * Gl[(size_t)src * d + c]; }
            Dw[(size_t)t * d + c] = acc;
        }
        rms_rows(Dw.data(), N, d, L.conv_norm.data(), eps, Dw.data());
        for (size_t i = 0; i < Dw.size(); ++i) Dw[i] = silu(Dw[i]);
        capply(L.pw2, Dw.data(), N, Y.data(), scratch);
        for (size_t i = 0; i < (size_t)N * d; ++i) X[i] += Y[i];
        lap(3);
        // ---- second half-step FFN, then the layer output norm
        ffn_block(L.ffn_up1, L.ffn_down1, L.ffn_norm1, L.ffn_post_norm1);
        rms_rows(X.data(), N, d, L.ln2.data(), eps, X.data());
        lap(1);
    }
    // ---- tail: output projection (+bias) -> weightless RMSNorm -> mm.a.input_projection
    const int d2 = I.a_out_proj.w.rows;
    std::vector<float> T1((size_t)N * d2);
    capply(I.a_out_proj, X.data(), N, T1.data(), scratch);
    for (int t = 0; t < N; ++t) for (int i = 0; i < d2; ++i) T1[(size_t)t * d2 + i] += I.a_out_bias[i];
    rms_rows(T1.data(), N, d2, nullptr, eps, T1.data());
    std::vector<float> out((size_t)N * I.a_mm_proj.w.rows);
    capply(I.a_mm_proj, T1.data(), N, out.data(), scratch);
    lap(4);
    if (tm) fprintf(stderr, "[g4a-time] N=%d  sscp %.0f  ffn %.0f  attention %.0f  conv module %.0f  tail %.0f ms\n", N, tp[0], tp[1], tp[2], tp[3], tp[4]);
    return out;
}

}  // namespace ferro
