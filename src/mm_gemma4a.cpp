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
    static const bool prof = getenv("G4A_TIME") != nullptr;   // SSCP_PROF
    double ph[6] = {0, 0, 0, 0, 0, 0}; auto tq = std::chrono::steady_clock::now();
    auto pl = [&](int k) { if (!prof) return; auto t = std::chrono::steady_clock::now(); ph[k] += std::chrono::duration<double, std::milli>(t - tq).count(); tq = t; };
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
        pl(0);
        gemm_f32(I.a_conv_w[li].data(), Co, K, col.data(), NP, gem.data(), Co);
        pl(1);
        std::vector<float> out((size_t)Co * NP);                 // back to [Co][Ho][Wo]
        Pool::get().parallel_for(Co, [&](int lo, int hi) {
            for (int o = lo; o < hi; ++o) for (int p = 0; p < NP; ++p) out[(size_t)o * NP + p] = gem[(size_t)p * Co + o];
        });
        pl(2);
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
        pl(3);
        cur = std::move(out); C = Co; H = Ho; W = Wo;
    }
    n_pos = H;                                                   // time
    const int feat = W * C;                                      // freq-major flatten: feature = freq * C + ch
    std::vector<float> flat((size_t)n_pos * feat);
    for (int t = 0; t < n_pos; ++t) for (int f = 0; f < W; ++f) for (int c = 0; c < C; ++c)
        flat[(size_t)t * feat + (size_t)f * C + c] = cur[((size_t)c * H + t) * W + f];
    std::vector<float> proj((size_t)n_pos * I.a_d);
    pl(4);
    gemm_f32(I.a_inp_proj.data(), I.a_d, feat, flat.data(), n_pos, proj.data(), I.a_d);
    pl(5);
    if (prof) fprintf(stderr, "[sscp] im2col %.0f  conv gemm %.0f  transpose %.0f  layernorm+relu %.0f  flatten %.0f  input proj %.0f ms\n", ph[0], ph[1], ph[2], ph[3], ph[4], ph[5]);
    return proj;
}


namespace {
constexpr int AC_ = 12, AP_ = 12, AR_ = AP_ + 1;
// sinusoidal relative positions [R][d]: position = P - p, sin | cos halves
std::vector<float> audio_pos_emb(int d) {
    std::vector<float> pe((size_t)AR_ * d);
    const int nts = d / 2; const float inc = std::log(10000.0f) / std::max(nts - 1, 1);
    for (int p = 0; p < AR_; ++p) for (int i = 0; i < nts; ++i) {
        const float sc = (float)(AP_ - p) * std::exp(-(float)i * inc);
        pe[(size_t)p * d + i] = std::sin(sc); pe[(size_t)p * d + i + nts] = std::cos(sc);
    }
    return pe;
}
inline float attn_q_scale(int hd) { return (1.0f / std::sqrt((float)hd)) / std::log(2.0f); }
inline float attn_k_scale() { return std::log(1.0f + std::exp(1.0f)) / std::log(2.0f); }
}  // namespace

// Conformer blocks on the device: weights, constants (norm vectors, per_dim_scale, depthwise kernel, Pk = k_rel(pos_emb)) are
// uploaded once; per call only X goes up and comes back. Returns false (host path stays in charge) on anything unexpected.
bool upload_audio_dev(Gemma4Proj::Impl& I) {
    using gpu::Buf;
    if (I.a_hd != 128 || I.a_d != I.a_heads * 128 || I.a_d % 128 || I.a_d > 2048 || I.a_ffn % 128 || I.a_ffn > 4096) return false;
    const int d = I.a_d;
    static const bool pds_sp = getenv("FERRO_G4A_PDS") && !strcmp(getenv("FERRO_G4A_PDS"), "softplus");
    const std::vector<float> pe = audio_pos_emb(d);
    std::vector<float> scratch;
    I.agl.assign(I.a_layers, Gemma4Proj::Impl::AGL());
    auto up = [&](const std::vector<float>& x) { return gpu::upload(x.data(), x.size() * 4); };
    for (int l = 0; l < I.a_layers; ++l) {
        auto& L = I.al[l]; auto& g = I.agl[l];
        // q, k, v must share their clamps (they do in every block of the shipped file) so one fused GEMM + one clamp serves all three
        if (L.q.imin != L.k.imin || L.q.imin != L.v.imin || L.q.imax != L.k.imax || L.q.imax != L.v.imax ||
            L.q.omin != L.k.omin || L.q.omin != L.v.omin || L.q.omax != L.k.omax || L.q.omax != L.v.omax) return false;
        if (L.q.w.q.empty() || L.k.w.q.empty() || L.v.w.q.empty()) return false;
        g.qkv = pack_concat({&L.q.w.q, &L.k.w.q, &L.v.w.q}, "g4a.qkv");
        g.qkv.gpu_slot = gpu::upload_q8(g.qkv);
        if (g.qkv.gpu_slot < 0) return false;
        g.n_ffn = up(L.ffn_norm); g.p_ffn = up(L.ffn_post_norm); g.n_ffn1 = up(L.ffn_norm1); g.p_ffn1 = up(L.ffn_post_norm1);
        g.n_att = up(L.attn_pre_norm); g.p_att = up(L.attn_post_norm); g.n_conv = up(L.norm_conv); g.w_conv = up(L.conv_norm); g.ln2 = up(L.ln2);
        std::vector<float> pds(I.a_hd);
        for (int e = 0; e < I.a_hd; ++e) pds[e] = pds_sp ? std::log1p(std::exp(L.per_dim_scale[e])) : L.per_dim_scale[e];
        g.pds = up(pds); g.dw = up(L.conv_dw);
        std::vector<float> Pk((size_t)AR_ * d);
        capply(L.k_rel, pe.data(), AR_, Pk.data(), scratch);
        g.pk = up(Pk);
        if (!g.n_ffn || !g.p_ffn || !g.n_ffn1 || !g.p_ffn1 || !g.n_att || !g.p_att || !g.n_conv || !g.w_conv || !g.ln2 || !g.pds || !g.dw || !g.pk) return false;
    }
    return true;
}

static bool audio_blocks_gpu(Gemma4Proj::Impl& I, std::vector<float>& X, int N) {
    const int d = I.a_d, ffn = I.a_ffn, H = I.a_heads, hd = I.a_hd;
    const float eps = 1e-6f, qs = attn_q_scale(hd), ks = attn_k_scale();
    const size_t wide = (size_t)std::max({ffn, 3 * d, 2 * d});
    if ((size_t)N > I.a_cap) {
        gpu::release(I.aX); gpu::release(I.aY); gpu::release(I.aO);
        I.a_cap = (size_t)N;
        I.aX = gpu::alloc(I.a_cap * d * 4); I.aY = gpu::alloc(I.a_cap * wide * 4); I.aO = gpu::alloc(I.a_cap * d * 4);
        if (!I.aX || !I.aY || !I.aO) { I.aX = I.aY = I.aO = nullptr; I.a_cap = 0; return false; }
    }
    if (!gpu::ensure_qscratch(N, (int)wide)) return false;
    static const bool verbose = getenv("FERRO_VERBOSE") != nullptr;
    int stage = 0;
    bool ok = gpu::write(I.aX, X.data(), (size_t)N * d * 4);
    auto ffn_half = [&](const CLin& up, const CLin& dn, gpu::Buf nw, gpu::Buf pw, gpu::Buf ln2) {
        return gpu::op_a_norm_quant(I.aX, nw, true, up.imin, up.imax, N, d, eps) &&
               gpu::op_gemm_q(up.w.q.gpu_slot, N, I.aY) &&
               gpu::op_a_act_quant(I.aY, 1, up.omin, up.omax, dn.imin, dn.imax, N, ffn) &&
               gpu::op_gemm_q(dn.w.q.gpu_slot, N, I.aY) &&
               gpu::op_a_resid(I.aX, I.aY, dn.omin, dn.omax, true, pw, 0.5f, ln2, N, d, eps);
    };
    for (int l = 0; l < I.a_layers && ok; ++l) {
        auto& L = I.al[l]; auto& g = I.agl[l];
        stage = 1; ok = ffn_half(L.ffn_up, L.ffn_down, g.n_ffn, g.p_ffn, nullptr);
        if (!ok) break;
        stage = 2;
        ok = gpu::op_a_norm_quant(I.aX, g.n_att, true, L.q.imin, L.q.imax, N, d, eps) &&
             gpu::op_gemm_q(g.qkv.gpu_slot, N, I.aY) &&
             gpu::op_a_attn(I.aY, 3 * d, g.pk, g.pds, L.q.omin, L.q.omax, qs, ks, N, H, d, I.aO) &&
             gpu::op_a_act_quant(I.aO, 0, -FLT_MAX, FLT_MAX, L.o.imin, L.o.imax, N, d) &&
             gpu::op_gemm_q(L.o.w.q.gpu_slot, N, I.aY) &&
             gpu::op_a_resid(I.aX, I.aY, L.o.omin, L.o.omax, true, g.p_att, 1.0f, nullptr, N, d, eps);
        if (!ok) break;
        stage = 3;
        ok = gpu::op_a_norm_quant(I.aX, g.n_conv, true, L.pw1.imin, L.pw1.imax, N, d, eps) &&
             gpu::op_gemm_q(L.pw1.w.q.gpu_slot, N, I.aY) &&
             gpu::op_a_conv_mid(I.aY, d, L.pw1.omin, L.pw1.omax, g.dw, g.w_conv, L.pw2.imin, L.pw2.imax, N, eps) &&
             gpu::op_gemm_q(L.pw2.w.q.gpu_slot, N, I.aY) &&
             gpu::op_a_resid(I.aX, I.aY, L.pw2.omin, L.pw2.omax, false, nullptr, 1.0f, nullptr, N, d, eps);
        if (!ok) break;
        stage = 4; ok = ffn_half(L.ffn_up1, L.ffn_down1, g.n_ffn1, g.p_ffn1, g.ln2);
        if (!ok) break;
        stage = 5; ok = gpu::finish();                      // one submission per layer (i915 watchdog), also surfaces errors
    }
    if (ok) { stage = 6; ok = gpu::read(I.aX, X.data(), (size_t)N * d * 4); }
    if (!ok && verbose) fprintf(stderr, "[g4a] on-device conformer failed at stage %d (1 ffn, 2 attention, 3 conv, 4 ffn2, 5 finish, 6 read)\n", stage);
    return ok;
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
    const std::vector<float> pos_emb = audio_pos_emb(d);
    std::vector<float> scratch, xn((size_t)N * d), F((size_t)N * ffn), Y((size_t)N * d), Q((size_t)N * d), K((size_t)N * d), V((size_t)N * d), O((size_t)N * d);
    auto ffn_block = [&](const CLin& up, const CLin& down, const std::vector<float>& nw, const std::vector<float>& pw) {
        rms_rows(X.data(), N, d, nw.data(), eps, xn.data());
        capply(up, xn.data(), N, F.data(), scratch);
        for (size_t i = 0; i < F.size(); ++i) F[i] = silu(F[i]);
        capply(down, F.data(), N, Y.data(), scratch);
        rms_rows(Y.data(), N, d, pw.data(), eps, Y.data());
        for (size_t i = 0; i < (size_t)N * d; ++i) X[i] += 0.5f * Y[i];            // half-step residual
    };
    bool dev_done = false;
    if (I.a_dev_ok) { dev_done = audio_blocks_gpu(I, X, N); lap(1); }
    const int n_host_layers = dev_done ? 0 : I.a_layers;
    for (int l = 0; l < n_host_layers; ++l) {
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
