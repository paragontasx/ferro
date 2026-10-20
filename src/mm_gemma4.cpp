// mm_gemma4.cpp — see mm_gemma4.h. CPU reference forward (GEMMs route to the iGPU when weights are resident).
#include "mm_gemma4.h"
#include "mm_util.h"
#include "threadpool.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ferro {

namespace {
// Gemma4ClippableLinear: clamp(x, in) -> W x -> clamp(., out); limits are 1-element tensors next to the weight
struct CLin {
    Lin w;
    float imin = -FLT_MAX, imax = FLT_MAX, omin = -FLT_MAX, omax = FLT_MAX;
};
float scalar_or(const GgufFile& g, const std::string& name, float dflt) {
    if (!g.tensor(name)) return dflt;
    return tf32(g, name).at(0);
}
CLin clin(const GgufFile& g, const std::string& wname) {
    CLin c;
    c.w = lin(g, wname);
    const std::string base = wname.substr(0, wname.size() - strlen(".weight"));
    c.imin = scalar_or(g, base + ".input_min", -FLT_MAX);
    c.imax = scalar_or(g, base + ".input_max", FLT_MAX);
    c.omin = scalar_or(g, base + ".output_min", -FLT_MAX);
    c.omax = scalar_or(g, base + ".output_max", FLT_MAX);
    return c;
}
void clamp_buf(float* x, size_t n, float lo, float hi) {
    if (lo <= -FLT_MAX && hi >= FLT_MAX) return;
    Pool::get().parallel_for((int)((n + 8191) / 8192), [&](int a, int b) {
        const size_t i1 = std::min(n, (size_t)b * 8192);
        for (size_t i = (size_t)a * 8192; i < i1; ++i) x[i] = std::min(hi, std::max(lo, x[i]));
    });
}
// Y[T][rows]; X is clamped in place only when limits exist (callers pass a scratch copy)
void capply(const CLin& c, const float* X, int T, float* Y, std::vector<float>& scratch) {
    const float* xin = X;
    if (c.imin > -FLT_MAX || c.imax < FLT_MAX) {
        scratch.assign(X, X + (size_t)T * c.w.cols);
        clamp_buf(scratch.data(), scratch.size(), c.imin, c.imax);
        xin = scratch.data();
    }
    apply(c.w, xin, T, Y);
    clamp_buf(Y, (size_t)T * c.w.rows, c.omin, c.omax);
}
struct VLayer {
    std::vector<float> ln1, ln2, qn, kn, post_attn, post_ffn;
    CLin q, k, v, o, gate, up, down;
};
}  // namespace

struct Gemma4Proj::Impl {
    int d = 768, layers = 16, heads = 12, hd = 64, ps = 16, merge = 3, ffn = 3072, proj = 512;
    float eps = 1e-6f, theta = 100.0f;
    std::vector<float> patch_w, pos_tbl;   // patch_w [768][3*16*16]; pos_tbl [2][10240][768]
    int pos_size = 10240;
    std::vector<VLayer> vl;
    CLin mm_proj;                          // 768 -> 512
};

bool Gemma4Proj::load(const std::string& path, std::string* err) {
    try {
        if (!g_.open_file(path.c_str(), err)) return false;
        if (g_.arch != "clip") { *err = "not an mmproj (clip) GGUF"; return false; }
        impl_ = std::make_shared<Impl>();
        Impl& I = *impl_;
        if (g_.get_bool("clip.has_vision_encoder") && g_.get_str("clip.vision.projector_type") == "gemma4v") {
            I.d = (int)g_.get_int("clip.vision.embedding_length");
            I.layers = (int)g_.get_int("clip.vision.block_count");
            I.heads = (int)g_.get_int("clip.vision.attention.head_count");
            I.ffn = (int)g_.get_int("clip.vision.feed_forward_length");
            I.ps = (int)g_.get_int("clip.vision.patch_size");
            I.proj = (int)g_.get_int("clip.vision.projection_dim");
            I.hd = I.d / I.heads;
            I.eps = (float)g_.get_f64("clip.vision.attention.layer_norm_epsilon", 1e-6);
            vp_.patch_size = I.ps;
            vp_.n_merge = (int)g_.get_int("clip.vision.projector.scale_factor", 3);
            I.merge = vp_.n_merge;
            I.patch_w = tf32(g_, "v.patch_embd.weight");
            I.pos_tbl = tf32(g_, "v.position_embd.weight");
            I.pos_size = (int)(I.pos_tbl.size() / (2 * (size_t)I.d));
            I.vl.resize(I.layers);
            for (int i = 0; i < I.layers; ++i) {
                const std::string b = "v.blk." + std::to_string(i) + ".";
                VLayer& L = I.vl[i];
                L.ln1 = tf32(g_, b + "ln1.weight"); L.ln2 = tf32(g_, b + "ln2.weight");
                L.qn = tf32(g_, b + "attn_q_norm.weight"); L.kn = tf32(g_, b + "attn_k_norm.weight");
                L.post_attn = tf32(g_, b + "attn_post_norm.weight"); L.post_ffn = tf32(g_, b + "ffn_post_norm.weight");
                L.q = clin(g_, b + "attn_q.weight"); L.k = clin(g_, b + "attn_k.weight");
                L.v = clin(g_, b + "attn_v.weight"); L.o = clin(g_, b + "attn_out.weight");
                L.gate = clin(g_, b + "ffn_gate.weight"); L.up = clin(g_, b + "ffn_up.weight");
                L.down = clin(g_, b + "ffn_down.weight");
            }
            I.mm_proj = clin(g_, "mm.input_projection.weight");
            n_out_ = I.mm_proj.w.rows;
            v_ok_ = true;
        }
        return true;
    } catch (const std::exception& e) { *err = e.what(); return false; }
}

// neox RoPE on one half of each head: dims [off, off+hn*2) pair (i, i+hn); angle = pos * theta^(-i/hn)
static void rope_half(float* x, int T, int heads, int hd, int off, int hn, const int* pos, float theta) {
    std::vector<float> inv(hn);
    for (int i = 0; i < hn; ++i) inv[i] = std::pow(theta, -(float)i / (float)hn);
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            for (int h = 0; h < heads; ++h) {
                float* v = x + ((size_t)t * heads + h) * hd + off;
                for (int i = 0; i < hn; ++i) {
                    const float a = (float)pos[t] * inv[i], c = std::cos(a), s = std::sin(a);
                    const float x0 = v[i], x1 = v[i + hn];
                    v[i] = x0 * c - x1 * s;
                    v[i + hn] = x0 * s + x1 * c;
                }
            }
        }
    });
}

std::vector<float> Gemma4Proj::encode_image(const VisionInput& in) {
    if (!v_ok_) throw std::runtime_error("mmproj has no gemma4v vision encoder");
    Impl& I = *impl_;
    const int wp = in.nx / I.ps, hp = in.ny / I.ps, T = wp * hp, d = I.d, H = I.heads, hd = I.hd;
    if (in.nx % (I.ps * I.merge) || in.ny % (I.ps * I.merge)) throw std::runtime_error("image size not aligned to patch*merge");
    if (wp > I.pos_size || hp > I.pos_size) throw std::runtime_error("image too large for the position table");

    // patches [T][3*ps*ps] (planar c,y,x as the conv kernel), pixels scaled 2x-1
    const int pd = 3 * I.ps * I.ps;
    std::vector<float> patches((size_t)T * pd);
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const int px = t % wp, py = t / wp;
            float* o = patches.data() + (size_t)t * pd;
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < I.ps; ++y) {
                    const float* src = in.chw.data() + ((size_t)c * in.ny + (size_t)py * I.ps + y) * in.nx + (size_t)px * I.ps;
                    for (int x = 0; x < I.ps; ++x) *o++ = src[x] * 2.0f - 1.0f;
                }
        }
    });
    std::vector<float> X((size_t)T * d);
    gemm_f32(I.patch_w.data(), d, pd, patches.data(), T, X.data(), d);
    std::vector<int> pos_x(T), pos_y(T);
    for (int t = 0; t < T; ++t) { pos_x[t] = t % wp; pos_y[t] = t / wp; }
    const float* tbl_x = I.pos_tbl.data();
    const float* tbl_y = I.pos_tbl.data() + (size_t)I.pos_size * d;
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            float* x = X.data() + (size_t)t * d;
            const float* ex = tbl_x + (size_t)pos_x[t] * d;
            const float* ey = tbl_y + (size_t)pos_y[t] * d;
            for (int i = 0; i < d; ++i) x[i] += ex[i] + ey[i];
        }
    });

    std::vector<float> xn((size_t)T * d), Q((size_t)T * d), K((size_t)T * d), V((size_t)T * d), O((size_t)T * d), tmp((size_t)T * d);
    std::vector<float> G((size_t)T * I.ffn), U((size_t)T * I.ffn), scratch;
    std::vector<int> seg(T, 0);
    for (int l = 0; l < I.layers; ++l) {
        const VLayer& L = I.vl[l];
        rmsnorm_q(X.data(), T, d, L.ln1.data(), I.eps, xn.data());
        capply(L.q, xn.data(), T, Q.data(), scratch);
        capply(L.k, xn.data(), T, K.data(), scratch);
        capply(L.v, xn.data(), T, V.data(), scratch);
        rmsnorm_q(Q.data(), T * H, hd, L.qn.data(), I.eps, Q.data());
        rmsnorm_q(K.data(), T * H, hd, L.kn.data(), I.eps, K.data());
        rmsnorm_w1(V.data(), T * H, hd, I.eps, V.data());
        for (float* x : {Q.data(), K.data()}) {
            rope_half(x, T, H, hd, 0, hd / 4, pos_x.data(), I.theta);
            rope_half(x, T, H, hd, hd / 2, hd / 4, pos_y.data(), I.theta);
        }
        attention(Q.data(), K.data(), V.data(), O.data(), T, H, H, hd, 1.0f, seg.data(), -1, d, d, d);
        capply(L.o, O.data(), T, tmp.data(), scratch);
        rmsnorm_q(tmp.data(), T, d, L.post_attn.data(), I.eps, tmp.data());
        add_inplace(X.data(), tmp.data(), (size_t)T * d);
        rmsnorm_q(X.data(), T, d, L.ln2.data(), I.eps, xn.data());
        capply(L.gate, xn.data(), T, G.data(), scratch);
        capply(L.up, xn.data(), T, U.data(), scratch);
        act(G.data(), G.size(), GELU_TANH);
        for (size_t i = 0; i < G.size(); ++i) G[i] *= U[i];
        capply(L.down, G.data(), T, tmp.data(), scratch);
        rmsnorm_q(tmp.data(), T, d, L.post_ffn.data(), I.eps, tmp.data());
        add_inplace(X.data(), tmp.data(), (size_t)T * d);
    }

    // Gemma4VisionPooler: 3x3 average pool over the patch grid, x sqrt(d), RMSNorm (no weight), projector
    const int ow = wp / I.merge, oh = hp / I.merge, nt = ow * oh, k2 = I.merge * I.merge;
    std::vector<float> pooled((size_t)nt * d);
    const float sc = std::sqrt((float)d);
    Pool::get().parallel_for(nt, [&](int lo, int hi) {
        for (int o = lo; o < hi; ++o) {
            const int ox = o % ow, oy = o / ow;
            float* p = pooled.data() + (size_t)o * d;
            for (int i = 0; i < d; ++i) p[i] = 0;
            for (int dy = 0; dy < I.merge; ++dy)
                for (int dx = 0; dx < I.merge; ++dx) {
                    const float* x = X.data() + ((size_t)(oy * I.merge + dy) * wp + ox * I.merge + dx) * d;
                    for (int i = 0; i < d; ++i) p[i] += x[i];
                }
            for (int i = 0; i < d; ++i) p[i] = p[i] / (float)k2 * sc;
        }
    });
    rmsnorm_w1(pooled.data(), nt, d, I.eps, pooled.data());
    std::vector<float> out((size_t)nt * I.mm_proj.w.rows);
    capply(I.mm_proj, pooled.data(), nt, out.data(), scratch);
    return out;
}

}  // namespace ferro
