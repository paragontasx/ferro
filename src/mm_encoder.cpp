#include <chrono>
// mm_encoder.cpp — see mm_encoder.h. CPU reference forward (GEMMs route to the iGPU when the weights are
// resident there); op order and layouts follow llama.cpp tools/mtmd (siglip.cpp, conformer.cpp, clip.cpp).
#include "mm_encoder.h"
#include "mm_util.h"
#include "gpu_cl.h"
#include "threadpool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <deque>
#include <dirent.h>
#include <stdexcept>

namespace ferro {

// ------------------------------------------------------------------ helpers
static std::deque<std::string> g_mm_names;
static const char* keepn(const std::string& s) { g_mm_names.push_back(s); return g_mm_names.back().c_str(); }

std::vector<float> tf32(const GgufFile& g, const std::string& name) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) throw std::runtime_error("mmproj: missing tensor " + name);
    size_t n = 1;
    for (auto d : t->dims) n *= d;
    std::vector<float> v(n);
    const uint8_t* p = g.tensor_data(*t);
    if (t->type == T_F32) memcpy(v.data(), p, n * 4);
    else if (t->type == T_F16) for (size_t i = 0; i < n; ++i) v[i] = fp16_to_f32(((const uint16_t*)p)[i]);
    else if (t->type == T_BF16) for (size_t i = 0; i < n; ++i) v[i] = bf16_to_f32(((const uint16_t*)p)[i]);
    else if (t->type == T_Q8_0) {
        const block_q8_0* b = (const block_q8_0*)p;
        for (size_t k = 0; k < n / QK8_0; ++k) {
            float s = fp16_to_f32((uint16_t)b[k].d);
            for (int i = 0; i < QK8_0; ++i) v[k * QK8_0 + i] = b[k].qs[i] * s;
        }
    } else throw std::runtime_error("mmproj: unsupported tensor type for " + name);
    return v;
}
static inline uint16_t f32_to_f16_bits(float x);
Lin lin(const GgufFile& g, const std::string& w, const std::string& b) {
    const GGUFTensor* t = g.tensor(w);
    if (!t) throw std::runtime_error("mmproj: missing tensor " + w);
    Lin L;
    L.cols = (int)t->dims[0];
    L.rows = 1;
    for (size_t i = 1; i < t->dims.size(); ++i) L.rows *= (int)t->dims[i];
    if (t->type == T_Q8_0) L.q = pack_q8(g.tensor_data(*t), L.rows, L.cols, (L.cols / QK8_0) * (int)sizeof(block_q8_0), keepn(w));
    else L.f = tf32(g, w);
    if (!b.empty()) L.b = tf32(g, b);
    return L;
}
// Y[T][rows] = X[T][cols] W^T + b
void apply(const Lin& L, const float* X, int T, float* Y) {
    if (!L.f.empty()) gemm_f32(L.f.data(), L.rows, L.cols, X, T, Y, L.rows);
    else if (getenv("FERRO_MM_QACT")) {
        // emulate ggml's CPU mul_mat: activations quantized to Q8_0 (32-blocks, fp16 scale) before the int8 dot
        std::vector<float> xq((size_t)T * L.cols);
        for (size_t r = 0; r < (size_t)T; ++r)
            for (int b = 0; b < L.cols / 32; ++b) {
                const float* x = X + r * L.cols + b * 32; float* o = xq.data() + r * L.cols + b * 32;
                float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
                const float df = amax / 127.0f, id = df != 0.0f ? 1.0f / df : 0.0f, d = fp16_to_f32(f32_to_f16_bits(df));
                for (int i = 0; i < 32; ++i) o[i] = std::round(x[i] * id) * d;
            }
        gemm_q8_fma(L.q, xq.data(), T, nullptr, nullptr, Y, L.rows);
    }
    else gemm_q8_fma(L.q, X, T, nullptr, nullptr, Y, L.rows);
    if (!L.b.empty()) {
        const float* b = L.b.data();
        const int r = L.rows;
        Pool::get().parallel_for(T, [&](int lo, int hi) {
            for (int t = lo; t < hi; ++t) { float* y = Y + (size_t)t * r; for (int i = 0; i < r; ++i) y[i] += b[i]; }
        });
    }
}
static void layernorm(const float* x, int T, int d, const float* w, const float* b, float eps, float* y) {
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const float* xr = x + (size_t)t * d;
            float* yr = y + (size_t)t * d;
            float mean = 0;
            for (int i = 0; i < d; ++i) mean += xr[i];
            mean /= d;
            float var = 0;
            for (int i = 0; i < d; ++i) { float v = xr[i] - mean; var += v * v; }
            var /= d;
            const float inv = 1.0f / std::sqrt(var + eps);
            for (int i = 0; i < d; ++i) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
        }
    });
}
// round-to-nearest-even f32 -> f16 bits (F16C)
__attribute__((target("f16c"))) static inline uint16_t f32_to_f16_bits(float x) { return (uint16_t)_cvtss_sh(x, 0); }
__attribute__((target("f16c"))) static inline float f16_bits_to_f32(uint16_t h) { return _cvtsh_ss(h); }
// FERRO_MM_GELU16=1: emulate ggml's CPU GELU (fp16 lookup table) for bit-closer parity checks
__attribute__((target("f16c"))) static float gelu_tanh_f16table(float v) {
    if (v <= -10.0f) return 0.0f;
    if (v >= 10.0f) return v;
    const float x = f16_bits_to_f32(f32_to_f16_bits(v));
    const float g = 0.5f * x * (1.0f + std::tanh(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
    return f16_bits_to_f32(f32_to_f16_bits(g));
}
void act(float* x, size_t n, Act a) {
    static const bool gelu16 = getenv("FERRO_MM_GELU16") != nullptr;
    Pool::get().parallel_for((int)((n + 4095) / 4096), [&](int lo, int hi) {
        const size_t i1 = std::min(n, (size_t)hi * 4096);
        for (size_t i = (size_t)lo * 4096; i < i1; ++i) {
            const float v = x[i];
            switch (a) {
                case GELU_TANH: x[i] = gelu16 ? gelu_tanh_f16table(v) : 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * v * (1.0f + 0.044715f * v * v))); break;
                case GELU_ERF:  x[i] = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f)); break;
                case SILU:      x[i] = v / (1.0f + std::exp(-v)); break;
                case RELU:      x[i] = v > 0 ? v : 0; break;
            }
        }
    });
}
void add_inplace(float* y, const float* x, size_t n, float s) {
    Pool::get().parallel_for((int)((n + 8191) / 8192), [&](int lo, int hi) {
        const size_t i1 = std::min(n, (size_t)hi * 8192);
        for (size_t i = (size_t)lo * 8192; i < i1; ++i) y[i] += s * x[i];
    });
}

// ------------------------------------------------------------------ weights
struct VLayer { std::vector<float> ln1w, ln1b, ln2w, ln2b; Lin qkv, o, up, down; };
struct ALayer {
    std::vector<float> ffn_norm_w, ffn_norm_b, ffn_norm1_w, ffn_norm1_b, ln1w, ln1b, ln2w, ln2b, norm_conv_w, norm_conv_b;
    std::vector<float> pos_bias_u, pos_bias_v, conv_dw_w, conv_dw_b, conv_norm_w, conv_norm_b;   // dw [512][9]
    Lin ffn_up, ffn_down, ffn_up1, ffn_down1, q, k, v, o, linear_pos, pw1, pw2;
};
struct MmProj::Impl {
    // vision
    int v_d = 768, v_layers = 12, v_heads = 12, v_ps = 16, v_merge = 2, v_side = 16;
    float v_eps = 1e-6f;
    std::vector<float> patch_w, patch_b, pos_embd;   // patch_w [768][3*16*16]
    std::vector<VLayer> vl;
    std::vector<float> post_ln_w, post_ln_b;
    Lin mm1, mm2;
    // audio
    int a_d = 512, a_layers = 17, a_heads = 8;
    float a_eps = 1e-5f;
    std::vector<float> c0w, c0b, c2w, c2b, c3w, c3b, c5w, c5b, c6w, c6b;   // pre-encode convs
    Lin pre_out;
    std::vector<ALayer> al;
    std::vector<float> mm0w, mm0b, mm4w, mm4b;
    Lin mma1, mma3, mma5, mma6;
    // ---- iGPU vision (weights resident, whole encoder on-device)
    bool vgpu = false;
    WMat patch_q;                                    // patch embedding, Q8 (int8 DP4A path)
    struct GV { gpu::Buf ln1w = nullptr, ln1b = nullptr, ln2w = nullptr, ln2b = nullptr, qkvb = nullptr, ob = nullptr, upb = nullptr, downb = nullptr; };
    std::vector<GV> gv;
    gpu::Buf postw = nullptr, postb = nullptr, mm1b = nullptr, mm2b = nullptr;
    gpu::Buf bX = nullptr, bC = nullptr, bD = nullptr, bG = nullptr, blo = nullptr, bhi = nullptr, bsrc = nullptr, bS = nullptr;
    size_t cap = 0, capm = 0, capS = 0;
};

bool MmProj::load(const std::string& path, std::string* err) {
    try {
        if (!g_.open_file(path.c_str(), err)) return false;
        if (g_.arch != "clip") { *err = "not an mmproj (clip) GGUF"; return false; }
        impl_ = std::make_shared<Impl>();
        Impl& I = *impl_;
        const std::string vproj = g_.get_str("clip.vision.projector_type"), aproj = g_.get_str("clip.audio.projector_type");
        if (g_.get_bool("clip.has_vision_encoder") && vproj == "d1omni_v") {
            I.v_d = (int)g_.get_int("clip.vision.embedding_length");
            I.v_layers = (int)g_.get_int("clip.vision.block_count");
            I.v_heads = (int)g_.get_int("clip.vision.attention.head_count");
            I.v_ps = (int)g_.get_int("clip.vision.patch_size");
            I.v_merge = (int)g_.get_int("clip.vision.projector.scale_factor", 2);
            I.v_eps = (float)g_.get_f64("clip.vision.attention.layer_norm_epsilon", 1e-6);
            auto mean = g_.get_f64arr("clip.vision.image_mean"), stdv = g_.get_f64arr("clip.vision.image_std");
            vp_.patch_size = I.v_ps; vp_.n_merge = I.v_merge;
            vp_.min_pixels = 64 * I.v_ps * I.v_ps * I.v_merge * I.v_merge;
            vp_.max_pixels = 256 * I.v_ps * I.v_ps * I.v_merge * I.v_merge;
            for (int c = 0; c < 3; ++c) { vp_.mean[c] = (float)mean.at(c); vp_.stdv[c] = (float)stdv.at(c); }
            I.patch_w = tf32(g_, "v.patch_embd.weight");
            I.patch_b = tf32(g_, "v.patch_embd.bias");
            I.pos_embd = tf32(g_, "v.position_embd.weight");
            I.v_side = (int)std::lround(std::sqrt((double)(I.pos_embd.size() / I.v_d)));
            I.vl.resize(I.v_layers);
            for (int i = 0; i < I.v_layers; ++i) {
                const std::string b = "v.blk." + std::to_string(i) + ".";
                VLayer& L = I.vl[i];
                L.ln1w = tf32(g_, b + "ln1.weight"); L.ln1b = tf32(g_, b + "ln1.bias");
                L.ln2w = tf32(g_, b + "ln2.weight"); L.ln2b = tf32(g_, b + "ln2.bias");
                Lin q = lin(g_, b + "attn_q.weight", b + "attn_q.bias"), k = lin(g_, b + "attn_k.weight", b + "attn_k.bias"),
                    v = lin(g_, b + "attn_v.weight", b + "attn_v.bias");
                if (q.q.empty() || k.q.empty() || v.q.empty()) throw std::runtime_error("mmproj: expected Q8_0 vision attention weights");
                L.qkv.q = pack_concat({&q.q, &k.q, &v.q}, keepn(b + "attn_qkv"));
                L.qkv.rows = 3 * I.v_d; L.qkv.cols = I.v_d;
                L.qkv.b = q.b; L.qkv.b.insert(L.qkv.b.end(), k.b.begin(), k.b.end()); L.qkv.b.insert(L.qkv.b.end(), v.b.begin(), v.b.end());
                L.o = lin(g_, b + "attn_out.weight", b + "attn_out.bias");
                L.up = lin(g_, b + "ffn_up.weight", b + "ffn_up.bias");
                L.down = lin(g_, b + "ffn_down.weight", b + "ffn_down.bias");
            }
            I.post_ln_w = tf32(g_, "v.post_ln.weight"); I.post_ln_b = tf32(g_, "v.post_ln.bias");
            if (g_.tensor("mm.input_norm.weight")) throw std::runtime_error("mmproj: projector input norm not supported");
            I.mm1 = lin(g_, "mm.1.weight", "mm.1.bias");
            I.mm2 = lin(g_, "mm.2.weight", "mm.2.bias");
            n_out_ = I.mm2.rows;
            v_ok_ = true;
        }
        if (g_.get_bool("clip.has_audio_encoder") && aproj == "d1omni_a") {
            I.a_d = (int)g_.get_int("clip.audio.embedding_length");
            I.a_layers = (int)g_.get_int("clip.audio.block_count");
            I.a_heads = (int)g_.get_int("clip.audio.attention.head_count");
            I.a_eps = (float)g_.get_f64("clip.audio.attention.layer_norm_epsilon", 1e-5);
            a_n_mel_ = (int)g_.get_int("clip.audio.num_mel_bins", 128);
            I.c0w = tf32(g_, "a.conv1d.0.weight"); I.c0b = tf32(g_, "a.conv1d.0.bias");
            I.c2w = tf32(g_, "a.conv1d.2.weight"); I.c2b = tf32(g_, "a.conv1d.2.bias");
            I.c3w = tf32(g_, "a.conv1d.3.weight"); I.c3b = tf32(g_, "a.conv1d.3.bias");
            I.c5w = tf32(g_, "a.conv1d.5.weight"); I.c5b = tf32(g_, "a.conv1d.5.bias");
            I.c6w = tf32(g_, "a.conv1d.6.weight"); I.c6b = tf32(g_, "a.conv1d.6.bias");
            I.pre_out = lin(g_, "a.pre_encode.out.weight", "a.pre_encode.out.bias");
            I.al.resize(I.a_layers);
            for (int i = 0; i < I.a_layers; ++i) {
                const std::string b = "a.blk." + std::to_string(i) + ".";
                ALayer& L = I.al[i];
                L.ffn_norm_w = tf32(g_, b + "ffn_norm.weight"); L.ffn_norm_b = tf32(g_, b + "ffn_norm.bias");
                L.ffn_norm1_w = tf32(g_, b + "ffn_norm_1.weight"); L.ffn_norm1_b = tf32(g_, b + "ffn_norm_1.bias");
                L.ln1w = tf32(g_, b + "ln1.weight"); L.ln1b = tf32(g_, b + "ln1.bias");
                L.ln2w = tf32(g_, b + "ln2.weight"); L.ln2b = tf32(g_, b + "ln2.bias");
                L.norm_conv_w = tf32(g_, b + "norm_conv.weight"); L.norm_conv_b = tf32(g_, b + "norm_conv.bias");
                L.pos_bias_u = tf32(g_, b + "pos_bias_u"); L.pos_bias_v = tf32(g_, b + "pos_bias_v");
                L.conv_dw_w = tf32(g_, b + "conv_dw.weight"); L.conv_dw_b = tf32(g_, b + "conv_dw.bias");
                L.conv_norm_w = tf32(g_, b + "conv_norm.weight"); L.conv_norm_b = tf32(g_, b + "conv_norm.bias");
                L.ffn_up = lin(g_, b + "ffn_up.weight", b + "ffn_up.bias");
                L.ffn_down = lin(g_, b + "ffn_down.weight", b + "ffn_down.bias");
                L.ffn_up1 = lin(g_, b + "ffn_up_1.weight", b + "ffn_up_1.bias");
                L.ffn_down1 = lin(g_, b + "ffn_down_1.weight", b + "ffn_down_1.bias");
                L.q = lin(g_, b + "attn_q.weight", b + "attn_q.bias");
                L.k = lin(g_, b + "attn_k.weight", b + "attn_k.bias");
                L.v = lin(g_, b + "attn_v.weight", b + "attn_v.bias");
                L.o = lin(g_, b + "attn_out.weight", b + "attn_out.bias");
                L.linear_pos = lin(g_, b + "linear_pos.weight");
                L.pw1 = lin(g_, b + "conv_pw1.weight", b + "conv_pw1.bias");
                L.pw2 = lin(g_, b + "conv_pw2.weight", b + "conv_pw2.bias");
            }
            I.mm0w = tf32(g_, "mm.a.mlp.0.weight"); I.mm0b = tf32(g_, "mm.a.mlp.0.bias");
            I.mma1 = lin(g_, "mm.a.mlp.1.weight", "mm.a.mlp.1.bias");
            I.mma3 = lin(g_, "mm.a.mlp.3.weight", "mm.a.mlp.3.bias");
            I.mm4w = tf32(g_, "mm.a.mlp.4.weight"); I.mm4b = tf32(g_, "mm.a.mlp.4.bias");
            I.mma5 = lin(g_, "mm.a.mlp.5.weight", "mm.a.mlp.5.bias");
            I.mma6 = lin(g_, "mm.a.mlp.6.weight", "mm.a.mlp.6.bias");
            if (n_out_ && n_out_ != I.mma6.rows) throw std::runtime_error("mmproj: vision and audio output widths differ");
            n_out_ = I.mma6.rows;
            a_ok_ = true;
        }
        if (!v_ok_ && !a_ok_) { *err = "mmproj has no d1omni_v vision or d1omni_a audio encoder"; return false; }
    } catch (const std::exception& e) {
        *err = e.what();
        return false;
    }
    return true;
}

void MmProj::upload_gpu() {
    Impl& I = *impl_;
    auto slot = [](Lin& L) { if (!L.q.empty() && L.q.gpu_slot < 0) L.q.gpu_slot = gpu::upload_q8(L.q); };
    for (auto& L : I.vl) { slot(L.qkv); slot(L.o); slot(L.up); slot(L.down); }
    if (v_ok_) { slot(I.mm1); slot(I.mm2); }
    const char* mg_env = getenv("FERRO_MM_GPU");
    if (v_ok_ && (!mg_env || atoi(mg_env) != 0)) {
        // whole vision encoder on the iGPU: needs every GEMM resident + the small vectors
        bool ok = !I.mm1.q.empty() && !I.mm2.q.empty() && I.mm1.q.gpu_slot >= 0 && I.mm2.q.gpu_slot >= 0;
        for (auto& L : I.vl) ok = ok && L.qkv.q.gpu_slot >= 0 && L.o.q.gpu_slot >= 0 && L.up.q.gpu_slot >= 0 && L.down.q.gpu_slot >= 0;
        auto up = [&](const std::vector<float>& v) { gpu::Buf b = gpu::upload(v.data(), v.size() * 4); ok = ok && b; return b; };
        if (ok) {
            I.patch_q = pack_from_f32(I.patch_w.data(), I.v_d, 3 * I.v_ps * I.v_ps, 3 * I.v_ps * I.v_ps, "v.patch_embd");
            I.patch_q.gpu_slot = gpu::upload_q8(I.patch_q);
            ok = I.patch_q.gpu_slot >= 0;
            I.gv.resize(I.vl.size());
            for (size_t i = 0; ok && i < I.vl.size(); ++i) {
                const VLayer& L = I.vl[i]; auto& g = I.gv[i];
                g.ln1w = up(L.ln1w); g.ln1b = up(L.ln1b); g.ln2w = up(L.ln2w); g.ln2b = up(L.ln2b);
                g.qkvb = up(L.qkv.b); g.ob = up(L.o.b); g.upb = up(L.up.b); g.downb = up(L.down.b);
            }
            I.postw = up(I.post_ln_w); I.postb = up(I.post_ln_b); I.mm1b = up(I.mm1.b); I.mm2b = up(I.mm2.b);
        }
        I.vgpu = ok;
        if (!ok && getenv("FERRO_VERBOSE")) {
            int miss = 0;
            for (auto& L : I.vl) miss += (L.qkv.q.gpu_slot < 0) + (L.o.q.gpu_slot < 0) + (L.up.q.gpu_slot < 0) + (L.down.q.gpu_slot < 0);
            fprintf(stderr, "[mm] vision iGPU upload incomplete: %d layer GEMMs missing, mm1 %d mm2 %d patch %d\n", miss,
                    I.mm1.q.gpu_slot, I.mm2.q.gpu_slot, I.patch_q.gpu_slot);
        }
        if (getenv("FERRO_VERBOSE")) fprintf(stderr, "[mm] vision encoder on %s\n", ok ? "iGPU" : "CPU (+iGPU GEMMs)");
    }
    for (auto& L : I.al) for (Lin* p : {&L.ffn_up, &L.ffn_down, &L.ffn_up1, &L.ffn_down1, &L.q, &L.k, &L.v, &L.o, &L.linear_pos}) slot(*p);
    if (a_ok_) { slot(I.pre_out); slot(I.mma1); slot(I.mma3); slot(I.mma5); slot(I.mma6); }
}

// ------------------------------------------------------------------ vision
// ggml_interpolate(bilinear | antialias), align_corners = false: src [sw][sh] grid of rows d -> [dw][dh]
static std::vector<float> resize_pos_embd(const std::vector<float>& pe, int d, int side, int dw, int dh) {
    if (dw == side && dh == side) return pe;
    std::vector<float> out((size_t)dw * dh * d);
    const float sf0 = (float)dw / side, sf1 = (float)dh / side, po = 0.5f;
    const float support1 = std::max(1.0f, 1.0f / sf1), invscale1 = 1.0f / support1;
    const float support0 = std::max(1.0f, 1.0f / sf0), invscale0 = 1.0f / support0;
    auto tri = [](float x) { return std::max(1.0f - std::fabs(x), 0.0f); };
    Pool::get().parallel_for(dh, [&](int lo, int hi) {
        for (int i1 = lo; i1 < hi; ++i1) {
            const float y = ((float)i1 + po) / sf1;
            for (int i0 = 0; i0 < dw; ++i0) {
                const float x = ((float)i0 + po) / sf0;
                const int64_t x_min = std::max<int64_t>((int64_t)(x - support0 + po), 0), x_max = std::min<int64_t>((int64_t)(x + support0 + po), side);
                const int64_t y_min = std::max<int64_t>((int64_t)(y - support1 + po), 0), y_max = std::min<int64_t>((int64_t)(y + support1 + po), side);
                float* o = out.data() + ((size_t)i1 * dw + i0) * d;
                for (int e = 0; e < d; ++e) {
                    float val = 0.0f, tw = 0.0f;
                    for (int64_t sy = y_min; sy < y_max; sy++) {
                        const float wy = tri((sy - y + po) * invscale1);
                        for (int64_t sx = x_min; sx < x_max; sx++) {
                            const float wx = tri((sx - x + po) * invscale0);
                            const float w = wx * wy;
                            if (w <= 0.0f) continue;
                            val += pe[((size_t)sy * side + sx) * d + e] * w;
                            tw += w;
                        }
                    }
                    if (tw > 0.0f) val /= tw;
                    o[e] = val;
                }
            }
        }
    });
    return out;
}

// whole vision encoder on the iGPU; false -> caller falls back to the CPU path
static bool encode_vision_gpu(MmProj::Impl& I, const std::vector<VisionInput>& in, const std::vector<float>& P,
                              const std::vector<float>& posb, const std::vector<int>& seg, int T,
                              const std::vector<int>& src, int Tm, std::vector<float>& out) {
    const int d = I.v_d, nh = I.v_heads, hd = d / nh, kk = 3 * I.v_ps * I.v_ps, ffv = I.vl[0].up.rows;
    const int dm = d * I.v_merge * I.v_merge, dp = I.mm1.rows, dout = I.mm2.rows;
    if (kk != d && kk > 3 * d) return false;
    if ((size_t)T > I.cap) {
        for (gpu::Buf* b : {&I.bX, &I.bC, &I.bD, &I.bG, &I.blo, &I.bhi}) { gpu::release(*b); *b = nullptr; }
        const size_t t = (size_t)T;
        I.bX = gpu::alloc(t * d * 4); I.bC = gpu::alloc(t * std::max(3 * d, kk) * 4); I.bD = gpu::alloc(t * std::max(d, dout) * 4);
        I.bG = gpu::alloc(t * std::max(ffv, dp) * 4); I.blo = gpu::alloc(t * 4); I.bhi = gpu::alloc(t * 4);
        for (gpu::Buf b : {I.bX, I.bC, I.bD, I.bG, I.blo, I.bhi}) if (!b) { I.cap = 0; return false; }
        I.cap = t;
    }
    if ((size_t)Tm * I.v_merge * I.v_merge > I.capm) {
        gpu::release(I.bsrc);
        I.capm = (size_t)Tm * I.v_merge * I.v_merge;
        I.bsrc = gpu::alloc(I.capm * 4);
        if (!I.bsrc) { I.capm = 0; return false; }
    }
    if (!gpu::ensure_qscratch(T, std::max({kk, 3 * d, ffv, dm, dp}))) return false;
    std::vector<int> lo(T), hi(T);
    long work = 0;
    for (int t = 0; t < T;) { int u = t; while (u + 1 < T && seg[u + 1] == seg[t]) ++u; for (int k = t; k <= u; ++k) { lo[k] = t; hi[k] = u; } work += (long)(u - t + 1) * (u - t + 1); t = u + 1; }
    const gpu::Buf X = I.bX, C = I.bC, D = I.bD, G = I.bG;
    static const bool prof = getenv("FERRO_TIME") != nullptr;
    double tq[8] = {0};
    auto T0 = std::chrono::steady_clock::now();
    auto P_ = [&](int k, bool r) { if (prof) { gpu::finish(); auto t = std::chrono::steady_clock::now(); tq[k] += std::chrono::duration<double>(t - T0).count(); T0 = t; } return r; };
    bool ok = gpu::write(C, P.data(), P.size() * 4) && gpu::write(X, posb.data(), posb.size() * 4) &&
              gpu::write(I.blo, lo.data(), (size_t)T * 4) && gpu::write(I.bhi, hi.data(), (size_t)T * 4) &&
              gpu::write(I.bsrc, src.data(), src.size() * 4);
    // patch embedding: D = P @ Wp^T; the first residual adds it to X = patch bias + position embedding
    ok = ok && gpu::op_quant_q(C, T, kk) && gpu::op_gemm_q(I.patch_q.gpu_slot, T, D);
    gpu::set_attn_work(work * nh);
    const float scale = 1.0f / std::sqrt((float)hd);
    // segments (one per tile / image); dense materialized attention unless FERRO_MM_DENSE=0
    std::vector<std::pair<int, int>> segs;
    for (int t = 0; t < T; t = hi[t] + 1) segs.push_back({t, hi[t] - t + 1});
    int nmax = 0; for (auto& sg : segs) nmax = std::max(nmax, sg.second);
    static const bool dense = !getenv("FERRO_MM_DENSE") || atoi(getenv("FERRO_MM_DENSE")) != 0;
    bool use_dense = dense && hd == 64;
    for (auto& sg : segs) if (sg.second & 3) use_dense = false;   // the dense kernels use float4 rows
    if (use_dense && (size_t)nh * nmax * nmax > I.capS) {
        gpu::release(I.bS);
        I.capS = (size_t)nh * nmax * nmax;
        I.bS = gpu::alloc(I.capS * 4);
        if (!I.bS) { I.capS = 0; use_dense = false; }
    }
    auto attention = [&]() {
        if (!use_dense) return gpu::op_attention_q(C, 3 * d, 0, d, 2 * d, T, nh, nh, hd, I.blo, I.bhi, 0, scale);
        for (auto& sg : segs) {
            if (!gpu::op_dense_attention(C, 3 * d, 0, d, 2 * d, sg.first, sg.second, nh, scale, I.bS, D, d)) return false;
            if (segs.size() > 1 && !gpu::finish()) return false;   // bounded submissions (i915 watchdog)
        }
        return gpu::op_quant_q(D, T, d);
    };
    for (int il = 0; ok && il < I.v_layers; ++il) {
        const VLayer& L = I.vl[il];
        const auto& g = I.gv[il];
        ok = P_(0, gpu::op_resid_norm_q(X, D, il == 0 ? nullptr : I.gv[il - 1].downb, g.ln1w, g.ln1b, true, T, d, I.v_eps))
          && P_(1, gpu::op_gemm_q(L.qkv.q.gpu_slot, T, C)) && P_(0, gpu::op_bias_add(C, g.qkvb, T, 3 * d))
          && P_(2, attention())
          && P_(1, gpu::op_gemm_q(L.o.q.gpu_slot, T, D))
          && P_(0, gpu::op_resid_norm_q(X, D, g.ob, g.ln2w, g.ln2b, true, T, d, I.v_eps))
          && P_(3, gpu::op_gemm_q(L.up.q.gpu_slot, T, G)) && P_(0, gpu::op_bias_act_mode_q(G, g.upb, 2, T, ffv))
          && P_(4, gpu::op_gemm_q(L.down.q.gpu_slot, T, D));
    }
    // last residual (its quantized LN output is unused), then post-LN + 2x2 unshuffle + quantize, projector
    ok = ok && gpu::op_resid_norm_q(X, D, I.gv.back().downb, I.postw, I.postb, true, T, d, I.v_eps)
            && gpu::op_ln_gather_q(X, I.bsrc, I.postw, I.postb, Tm * I.v_merge * I.v_merge, d, I.v_eps)
            && gpu::op_gemm_q(I.mm1.q.gpu_slot, Tm, G) && gpu::op_bias_act_mode_q(G, I.mm1b, 3, Tm, dp)
            && gpu::op_gemm_q(I.mm2.q.gpu_slot, Tm, D) && gpu::op_bias_add(D, I.mm2b, Tm, dout);
    out.resize((size_t)Tm * dout);
    ok = ok && gpu::read(D, out.data(), out.size() * 4);
    if (prof) { P_(5, true); fprintf(stderr, "[mm-time] T=%d  norms/bias/act %.0f  qkv+o gemm %.0f  attention %.0f  up gemm %.0f  down gemm %.0f  tail %.0f ms\n", T, tq[0]*1e3, tq[1]*1e3, tq[2]*1e3, tq[3]*1e3, tq[4]*1e3, tq[5]*1e3); }
    (void)in;
    return ok;
}

std::vector<float> MmProj::encode_vision(const std::vector<VisionInput>& in) {
    if (!v_ok_) throw std::runtime_error("this mmproj has no vision encoder");
    Impl& I = *impl_;
    const int d = I.v_d, ps = I.v_ps, kk = 3 * ps * ps, nh = I.v_heads, hd = d / nh, mg = I.v_merge;
    std::vector<int> off, wps, hps;
    int T = 0;
    for (const auto& v : in) {
        if (v.nx % ps || v.ny % ps) throw std::runtime_error("vision input not a multiple of the patch size");
        off.push_back(T); wps.push_back(v.nx / ps); hps.push_back(v.ny / ps);
        T += (v.nx / ps) * (v.ny / ps);
    }
    std::vector<int> seg(T);
    for (size_t i = 0; i < in.size(); ++i) for (int t = 0; t < wps[i] * hps[i]; ++t) seg[off[i] + t] = (int)i;
    // patch embedding (conv 16x16 / 16 as GEMM over the unfolded patches; weight cols = c*ps*ps + kh*ps + kw)
    std::vector<float> P((size_t)T * kk), X((size_t)T * d);
    for (size_t i = 0; i < in.size(); ++i) {
        const VisionInput& v = in[i];
        const size_t plane = (size_t)v.nx * v.ny;
        Pool::get().parallel_for(hps[i], [&](int lo, int hi) {
            for (int py = lo; py < hi; ++py)
                for (int px = 0; px < wps[i]; ++px) {
                    float* p = P.data() + (size_t)(off[i] + py * wps[i] + px) * kk;
                    for (int c = 0; c < 3; ++c)
                        for (int kh = 0; kh < ps; ++kh) {
                            const float* src = v.chw.data() + c * plane + (size_t)(py * ps + kh) * v.nx + px * ps;
                            memcpy(p + c * ps * ps + kh * ps, src, ps * 4);
                        }
                }
        });
    }
    if (I.vgpu) {
        // X0 = patch bias + resized position embeddings; src: merged token row -> source patch row (-1 = pad)
        std::vector<float> posb((size_t)T * d);
        std::vector<int> src;
        int Tm = 0;
        for (size_t i = 0; i < in.size(); ++i) {
            std::vector<float> pe = resize_pos_embd(I.pos_embd, d, I.v_side, wps[i], hps[i]);
            float* x = posb.data() + (size_t)off[i] * d;
            const size_t n = (size_t)wps[i] * hps[i];
            for (size_t t = 0; t < n; ++t) for (int e = 0; e < d; ++e) x[t * d + e] = I.patch_b[e] + pe[t * d + e];
            const int w2 = (wps[i] + mg - 1) / mg, h2 = (hps[i] + mg - 1) / mg;
            for (int Y = 0; Y < h2; ++Y)
                for (int Xc = 0; Xc < w2; ++Xc)
                    for (int dy = 0; dy < mg; ++dy)
                        for (int dx = 0; dx < mg; ++dx) {
                            const int py = Y * mg + dy, px = Xc * mg + dx;
                            src.push_back(py < hps[i] && px < wps[i] ? off[i] + py * wps[i] + px : -1);
                        }
            Tm += w2 * h2;
        }
        std::vector<float> out;
        if (encode_vision_gpu(I, in, P, posb, seg, T, src, Tm, out)) return out;
        fprintf(stderr, "[gpu] vision encoder failed -> CPU\n");
    }
    gemm_f32(I.patch_w.data(), d, kk, P.data(), T, X.data(), d);
    P.clear(); P.shrink_to_fit();
    for (size_t i = 0; i < in.size(); ++i) {
        std::vector<float> pe = resize_pos_embd(I.pos_embd, d, I.v_side, wps[i], hps[i]);
        float* x = X.data() + (size_t)off[i] * d;
        const size_t n = (size_t)wps[i] * hps[i];
        for (size_t t = 0; t < n; ++t) for (int e = 0; e < d; ++e) x[t * d + e] += I.patch_b[e] + pe[t * d + e];
    }
    // transformer (pre-LN)
    std::vector<float> H((size_t)T * d), C((size_t)T * 3 * d), O((size_t)T * d), F((size_t)T * I.vl[0].up.rows);
    const float scale = 1.0f / std::sqrt((float)hd);
    for (int il = 0; il < I.v_layers; ++il) {
        const VLayer& L = I.vl[il];
        layernorm(X.data(), T, d, L.ln1w.data(), L.ln1b.data(), I.v_eps, H.data());
        apply(L.qkv, H.data(), T, C.data());
        attention(C.data(), C.data() + d, C.data() + 2 * d, O.data(), T, nh, nh, hd, scale, seg.data(), -1, 3 * d, 3 * d, d);
        apply(L.o, O.data(), T, H.data());
        add_inplace(X.data(), H.data(), (size_t)T * d);
        layernorm(X.data(), T, d, L.ln2w.data(), L.ln2b.data(), I.v_eps, H.data());
        apply(L.up, H.data(), T, F.data());
        act(F.data(), (size_t)T * L.up.rows, GELU_TANH);
        apply(L.down, F.data(), T, H.data());
        add_inplace(X.data(), H.data(), (size_t)T * d);
    }
    layernorm(X.data(), T, d, I.post_ln_w.data(), I.post_ln_b.data(), I.v_eps, X.data());
    // 2x2 pixel unshuffle: token (Y, X) <- patches (mg*Y+dy, mg*X+dx), features [dy][dx][d]
    int Tm = 0;
    std::vector<int> moff;
    for (size_t i = 0; i < in.size(); ++i) { moff.push_back(Tm); Tm += ((wps[i] + mg - 1) / mg) * ((hps[i] + mg - 1) / mg); }
    const int dm = d * mg * mg;
    std::vector<float> M((size_t)Tm * dm, 0.0f);
    for (size_t i = 0; i < in.size(); ++i) {
        const int w2 = (wps[i] + mg - 1) / mg, h2 = (hps[i] + mg - 1) / mg;
        for (int Y = 0; Y < h2; ++Y)
            for (int Xc = 0; Xc < w2; ++Xc) {
                float* m = M.data() + (size_t)(moff[i] + Y * w2 + Xc) * dm;
                for (int dy = 0; dy < mg; ++dy)
                    for (int dx = 0; dx < mg; ++dx) {
                        const int py = Y * mg + dy, px = Xc * mg + dx;
                        if (py >= hps[i] || px >= wps[i]) continue;   // zero padding
                        memcpy(m + (dy * mg + dx) * d, X.data() + (size_t)(off[i] + py * wps[i] + px) * d, d * 4);
                    }
            }
    }
    std::vector<float> Hm((size_t)Tm * I.mm1.rows), out((size_t)Tm * I.mm2.rows);
    apply(I.mm1, M.data(), Tm, Hm.data());
    act(Hm.data(), Hm.size(), GELU_ERF);
    apply(I.mm2, Hm.data(), Tm, out.data());
    return out;
}

// ------------------------------------------------------------------ audio
// 2D conv over [C][H][W] maps (H = time, W = mel): 3x3, stride 2, pad 1
static void conv3x3_s2(const float* in, int C, int H, int W, const float* w, const float* b, int Cout, bool depthwise,
                       float* out, int& Ho, int& Wo) {
    Ho = (H - 1) / 2 + 1; Wo = (W - 1) / 2 + 1;
    const int ho = Ho, wo = Wo;
    Pool::get().parallel_for(Cout, [&](int lo, int hi) {
        for (int o = lo; o < hi; ++o) {
            float* op = out + (size_t)o * ho * wo;
            const float* wk = w + (size_t)o * 9;
            for (int y = 0; y < ho; ++y)
                for (int x = 0; x < wo; ++x) {
                    float acc = b[o];
                    const int cin0 = depthwise ? o : 0, cin1 = depthwise ? o + 1 : C;
                    for (int c = cin0; c < cin1; ++c) {
                        const float* ip = in + (size_t)c * H * W;
                        const float* wc = depthwise ? wk : w + ((size_t)o * C + c) * 9;
                        for (int kh = 0; kh < 3; ++kh) {
                            const int iy = 2 * y - 1 + kh;
                            if (iy < 0 || iy >= H) continue;
                            for (int kw = 0; kw < 3; ++kw) {
                                const int ix = 2 * x - 1 + kw;
                                if (ix < 0 || ix >= W) continue;
                                acc += ip[(size_t)iy * W + ix] * wc[kh * 3 + kw];
                            }
                        }
                    }
                    op[(size_t)y * wo + x] = acc;
                }
        }
    });
}
// pointwise conv [C][N] -> [Cout][N] (+bias), optional relu.
// AVX2/FMA: 4 output channels x 16 pixels per tile (8 accumulators); pixel tile outer so the 16-pixel input
// slab of all C channels stays in L1 across the output-channel groups.
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static void conv1x1(const float* in, int C, int N, const float* w, const float* b, int Cout, bool relu, float* out) {
    const int NT = N / 16;
    Pool::get().parallel_for(NT, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const int n0 = t * 16;
            int o = 0;
            for (; o + 4 <= Cout; o += 4) {
                __m256 a00 = _mm256_set1_ps(b[o]), a01 = a00, a10 = _mm256_set1_ps(b[o + 1]), a11 = a10;
                __m256 a20 = _mm256_set1_ps(b[o + 2]), a21 = a20, a30 = _mm256_set1_ps(b[o + 3]), a31 = a30;
                const float* w0 = w + (size_t)o * C;
                const float* ip = in + n0;
                for (int c = 0; c < C; ++c, ip += N) {
                    const __m256 x0 = _mm256_loadu_ps(ip), x1 = _mm256_loadu_ps(ip + 8);
                    __m256 wv = _mm256_broadcast_ss(w0 + c);
                    a00 = _mm256_fmadd_ps(wv, x0, a00); a01 = _mm256_fmadd_ps(wv, x1, a01);
                    wv = _mm256_broadcast_ss(w0 + C + c);
                    a10 = _mm256_fmadd_ps(wv, x0, a10); a11 = _mm256_fmadd_ps(wv, x1, a11);
                    wv = _mm256_broadcast_ss(w0 + 2 * (size_t)C + c);
                    a20 = _mm256_fmadd_ps(wv, x0, a20); a21 = _mm256_fmadd_ps(wv, x1, a21);
                    wv = _mm256_broadcast_ss(w0 + 3 * (size_t)C + c);
                    a30 = _mm256_fmadd_ps(wv, x0, a30); a31 = _mm256_fmadd_ps(wv, x1, a31);
                }
                if (relu) {
                    const __m256 z = _mm256_setzero_ps();
                    a00 = _mm256_max_ps(a00, z); a01 = _mm256_max_ps(a01, z); a10 = _mm256_max_ps(a10, z); a11 = _mm256_max_ps(a11, z);
                    a20 = _mm256_max_ps(a20, z); a21 = _mm256_max_ps(a21, z); a30 = _mm256_max_ps(a30, z); a31 = _mm256_max_ps(a31, z);
                }
                float* op = out + (size_t)o * N + n0;
                _mm256_storeu_ps(op, a00); _mm256_storeu_ps(op + 8, a01);
                _mm256_storeu_ps(op + N, a10); _mm256_storeu_ps(op + N + 8, a11);
                _mm256_storeu_ps(op + 2 * (size_t)N, a20); _mm256_storeu_ps(op + 2 * (size_t)N + 8, a21);
                _mm256_storeu_ps(op + 3 * (size_t)N, a30); _mm256_storeu_ps(op + 3 * (size_t)N + 8, a31);
            }
            for (; o < Cout; ++o)                              // leftover output channels
                for (int n = n0; n < n0 + 16; ++n) {
                    float acc = b[o];
                    for (int c = 0; c < C; ++c) acc += w[(size_t)o * C + c] * in[(size_t)c * N + n];
                    out[(size_t)o * N + n] = relu && acc < 0 ? 0.f : acc;
                }
        }
    });
    for (int n = NT * 16; n < N; ++n)                          // leftover pixels
        for (int o = 0; o < Cout; ++o) {
            float acc = b[o];
            for (int c = 0; c < C; ++c) acc += w[(size_t)o * C + c] * in[(size_t)c * N + n];
            out[(size_t)o * N + n] = relu && acc < 0 ? 0.f : acc;
        }
}

// first subsampling conv (1 -> Cout channels, 3x3 stride 2 pad 1) + ReLU: one shared im2col [9][pixels], then every
// output channel is 9 FMAs per 8 pixels
static void conv0_relu(const float* in, int H, int W, const float* w, const float* b, int Cout, float* out, int& Ho, int& Wo) {
    Ho = (H - 1) / 2 + 1; Wo = (W - 1) / 2 + 1;
    const int NP = Ho * Wo;
    std::vector<float> col((size_t)9 * NP);
    Pool::get().parallel_for(Ho, [&](int lo, int hi) {
        for (int y = lo; y < hi; ++y)
            for (int x = 0; x < Wo; ++x)
                for (int kh = 0; kh < 3; ++kh)
                    for (int kw = 0; kw < 3; ++kw) {
                        const int iy = 2 * y - 1 + kh, ix = 2 * x - 1 + kw;
                        col[(size_t)(kh * 3 + kw) * NP + (size_t)y * Wo + x] = (iy < 0 || iy >= H || ix < 0 || ix >= W) ? 0.f : in[(size_t)iy * W + ix];
                    }
    });
    Pool::get().parallel_for(Cout, [&](int lo, int hi) {
        for (int o = lo; o < hi; ++o) {
            __m256 wv[9];
            for (int k = 0; k < 9; ++k) wv[k] = _mm256_set1_ps(w[(size_t)o * 9 + k]);
            const __m256 bv = _mm256_set1_ps(b[o]), z = _mm256_setzero_ps();
            float* op = out + (size_t)o * NP;
            int p = 0;
            for (; p + 8 <= NP; p += 8) {
                __m256 acc = bv;
                for (int k = 0; k < 9; ++k) acc = _mm256_fmadd_ps(wv[k], _mm256_loadu_ps(col.data() + (size_t)k * NP + p), acc);
                _mm256_storeu_ps(op + p, _mm256_max_ps(acc, z));
            }
            for (; p < NP; ++p) {
                float acc = b[o];
                for (int k = 0; k < 9; ++k) acc += w[(size_t)o * 9 + k] * col[(size_t)k * NP + p];
                op[p] = acc > 0 ? acc : 0.f;
            }
        }
    });
}


// Transformer-XL rel-pos attention for head_dim 64: sc[j] = ((q+u).k_j + (q+v).p_{T-1-i+j}) * scale, softmax, O = P V.
// AVX2/FMA: 4 keys per step share the q loads; P V keeps the 64-wide output row in 8 registers across all keys.
static void rel_attention_hd64(const float* Q, const float* K, const float* V, const float* Pp, const float* ub, const float* vb,
                               float* O, int T, int nh, int d, float scale) {
    Pool::get().parallel_for(nh * T, [&](int lo, int hi) {
        std::vector<float> sc((size_t)T + 8);
        alignas(32) float qu[64], qv[64];
        for (int idx = lo; idx < hi; ++idx) {
            const int h = idx / T, i = idx % T;
            const float* q = Q + (size_t)i * d + h * 64;
            for (int e = 0; e < 64; ++e) { qu[e] = q[e] + ub[h * 64 + e]; qv[e] = q[e] + vb[h * 64 + e]; }
            const float* kb = K + h * 64;
            const float* pb = Pp + (size_t)(T - 1 - i) * d + h * 64;     // row for key j is pb + j * d
            int j = 0;
            float mx = -1e30f;
            for (; j + 4 <= T; j += 4) {
                __m256 ac0 = _mm256_setzero_ps(), ac1 = ac0, ac2 = ac0, ac3 = ac0, bd0 = ac0, bd1 = ac0, bd2 = ac0, bd3 = ac0;
                const float* k0 = kb + (size_t)j * d;
                const float* p0 = pb + (size_t)j * d;
                for (int e = 0; e < 64; e += 8) {
                    const __m256 a = _mm256_load_ps(qu + e), b = _mm256_load_ps(qv + e);
                    ac0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(k0 + e), ac0);
                    ac1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(k0 + d + e), ac1);
                    ac2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(k0 + 2 * (size_t)d + e), ac2);
                    ac3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(k0 + 3 * (size_t)d + e), ac3);
                    bd0 = _mm256_fmadd_ps(b, _mm256_loadu_ps(p0 + e), bd0);
                    bd1 = _mm256_fmadd_ps(b, _mm256_loadu_ps(p0 + d + e), bd1);
                    bd2 = _mm256_fmadd_ps(b, _mm256_loadu_ps(p0 + 2 * (size_t)d + e), bd2);
                    bd3 = _mm256_fmadd_ps(b, _mm256_loadu_ps(p0 + 3 * (size_t)d + e), bd3);
                }
                auto hsum = [](__m256 v) {
                    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
                    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
                    x = _mm_add_ss(x, _mm_shuffle_ps(x, x, 1));
                    return _mm_cvtss_f32(x);
                };
                sc[j] = (hsum(ac0) + hsum(bd0)) * scale; sc[j + 1] = (hsum(ac1) + hsum(bd1)) * scale;
                sc[j + 2] = (hsum(ac2) + hsum(bd2)) * scale; sc[j + 3] = (hsum(ac3) + hsum(bd3)) * scale;
                mx = std::max(std::max(mx, sc[j]), std::max(std::max(sc[j + 1], sc[j + 2]), sc[j + 3]));
            }
            for (; j < T; ++j) {
                float ac = 0, bd = 0;
                const float* k = kb + (size_t)j * d; const float* p = pb + (size_t)j * d;
                for (int e = 0; e < 64; ++e) { ac += qu[e] * k[e]; bd += qv[e] * p[e]; }
                sc[j] = (ac + bd) * scale;
                mx = std::max(mx, sc[j]);
            }
            float sum = 0;
            for (int jj = 0; jj < T; ++jj) { sc[jj] = std::exp(sc[jj] - mx); sum += sc[jj]; }
            const float inv = 1.0f / sum;
            __m256 o0 = _mm256_setzero_ps(), o1 = o0, o2 = o0, o3 = o0, o4 = o0, o5 = o0, o6 = o0, o7 = o0;
            const float* vj = V + h * 64;
            for (int jj = 0; jj < T; ++jj, vj += d) {
                const __m256 pj = _mm256_set1_ps(sc[jj] * inv);
                o0 = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj), o0);      o1 = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + 8), o1);
                o2 = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + 16), o2); o3 = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + 24), o3);
                o4 = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + 32), o4); o5 = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + 40), o5);
                o6 = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + 48), o6); o7 = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + 56), o7);
            }
            float* o = O + (size_t)i * d + h * 64;
            _mm256_storeu_ps(o, o0); _mm256_storeu_ps(o + 8, o1); _mm256_storeu_ps(o + 16, o2); _mm256_storeu_ps(o + 24, o3);
            _mm256_storeu_ps(o + 32, o4); _mm256_storeu_ps(o + 40, o5); _mm256_storeu_ps(o + 48, o6); _mm256_storeu_ps(o + 56, o7);
        }
    });
}

#pragma GCC pop_options

std::vector<float> MmProj::encode_audio(const Mel& mel) {
    if (!a_ok_) throw std::runtime_error("this mmproj has no audio encoder");
    Impl& I = *impl_;
    static const bool prof = getenv("FERRO_TIME") != nullptr;
    double tm[8] = {0};
    auto tl = std::chrono::steady_clock::now();
    auto lap = [&](int k) { if (!prof) return; auto t = std::chrono::steady_clock::now(); tm[k] += std::chrono::duration<double>(t - tl).count(); tl = t; };
    const int Tf = mel.n_len, W0 = mel.n_mel, ch = (int)I.c0b.size();
    // input map [1][H = frames][W = mel]
    std::vector<float> in((size_t)Tf * W0);
    for (int t = 0; t < Tf; ++t) for (int m = 0; m < W0; ++m) in[(size_t)t * W0 + m] = mel.data[(size_t)m * Tf + t];
    int H1, W1, H2, W2, H3, W3;
    std::vector<float> a((size_t)ch * ((Tf + 1) / 2 + 1) * (W0 / 2 + 1)), b2;
    double tc[6] = {0}; auto tcl = std::chrono::steady_clock::now();
    auto clap = [&](int k) { auto t = std::chrono::steady_clock::now(); tc[k] += std::chrono::duration<double>(t - tcl).count(); tcl = t; };
    conv0_relu(in.data(), Tf, W0, I.c0w.data(), I.c0b.data(), ch, a.data(), H1, W1);
    clap(0);
    b2.resize((size_t)ch * ((H1 + 1) / 2 + 1) * (W1 / 2 + 1));
    conv3x3_s2(a.data(), ch, H1, W1, I.c2w.data(), I.c2b.data(), ch, true, b2.data(), H2, W2);
    clap(1);
    conv1x1(b2.data(), ch, H2 * W2, I.c3w.data(), I.c3b.data(), ch, true, a.data());
    clap(2);
    conv3x3_s2(a.data(), ch, H2, W2, I.c5w.data(), I.c5b.data(), ch, true, b2.data(), H3, W3);
    clap(3);
    conv1x1(b2.data(), ch, H3 * W3, I.c6w.data(), I.c6b.data(), ch, true, a.data());
    clap(4);
    if (prof) fprintf(stderr, "[mm-time] audio convs: c0 %.0f  dw2 %.0f  pw3 %.0f  dw5 %.0f  pw6 %.0f ms (ch=%d, %dx%d -> %dx%d -> %dx%d)\n", tc[0]*1e3, tc[1]*1e3, tc[2]*1e3, tc[3]*1e3, tc[4]*1e3, ch, H1, W1, H2, W2, H3, W3);
    lap(0);
    // flatten (channel, freq) per time step: feature = w + W3 * c
    const int T = H3, fin = ch * W3, d = I.a_d;
    if (fin != I.pre_out.cols) throw std::runtime_error("audio: unexpected pre-encode width");
    std::vector<float> flat((size_t)T * fin);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < ch; ++c)
            for (int w = 0; w < W3; ++w) flat[(size_t)t * fin + w + W3 * c] = a[((size_t)c * H3 + t) * W3 + w];
    std::vector<float> X((size_t)T * d);
    apply(I.pre_out, flat.data(), T, X.data());
    // relative positional encodings: row p <-> relative distance (T - 1 - p)
    const int P = 2 * T - 1;
    std::vector<float> pe((size_t)P * d);
    {
        std::vector<double> inv_freq(d / 2);
        for (size_t i = 0; i < inv_freq.size(); ++i) inv_freq[i] = std::exp(-(std::log(10000.0) / (float)d) * (2.0f * (float)i));
        for (int p = 0; p < P; ++p)
            for (size_t i = 0; i < inv_freq.size(); ++i) {
                const float ang = (float)((T - p - 1) * inv_freq[i]);
                pe[(size_t)p * d + 2 * i] = sinf(ang);
                pe[(size_t)p * d + 2 * i + 1] = cosf(ang);
            }
    }
    const int nh = I.a_heads, hd = d / nh;
    std::vector<float> R((size_t)T * d), Hn((size_t)T * d), F((size_t)T * I.al[0].ffn_up.rows), Q((size_t)T * d), K((size_t)T * d),
        V((size_t)T * d), O((size_t)T * d), Pp((size_t)P * d), G((size_t)T * 2 * d);
    const float scale = 1.0f / std::sqrt((float)hd);
    lap(1);
    for (int il = 0; il < I.a_layers; ++il) {
        const ALayer& L = I.al[il];
        // macaron FFN 1 (half step)
        layernorm(X.data(), T, d, L.ffn_norm_w.data(), L.ffn_norm_b.data(), I.a_eps, Hn.data());
        apply(L.ffn_up, Hn.data(), T, F.data());
        act(F.data(), (size_t)T * L.ffn_up.rows, SILU);
        apply(L.ffn_down, F.data(), T, R.data());
        for (size_t i = 0; i < (size_t)T * d; ++i) R[i] = X[i] + 0.5f * R[i];
        lap(2);
        // rel-pos self-attention (Transformer-XL: (q+u)k + (q+v)p_rel)
        layernorm(R.data(), T, d, L.ln1w.data(), L.ln1b.data(), I.a_eps, Hn.data());
        apply(L.q, Hn.data(), T, Q.data());
        apply(L.k, Hn.data(), T, K.data());
        apply(L.v, Hn.data(), T, V.data());
        apply(L.linear_pos, pe.data(), P, Pp.data());
        lap(3);
        if (hd == 64) rel_attention_hd64(Q.data(), K.data(), V.data(), Pp.data(), L.pos_bias_u.data(), L.pos_bias_v.data(), O.data(), T, nh, d, scale);
        else
        Pool::get().parallel_for(nh * T, [&](int lo, int hi) {
            std::vector<float> sc(T), qu(hd), qv(hd);
            for (int idx = lo; idx < hi; ++idx) {
                const int h = idx / T, i = idx % T;
                const float* q = Q.data() + (size_t)i * d + h * hd;
                for (int e = 0; e < hd; ++e) { qu[e] = q[e] + L.pos_bias_u[h * hd + e]; qv[e] = q[e] + L.pos_bias_v[h * hd + e]; }
                float mx = -1e30f;
                for (int j = 0; j < T; ++j) {
                    const float* k = K.data() + (size_t)j * d + h * hd;
                    const float* p = Pp.data() + (size_t)(T - 1 - i + j) * d + h * hd;
                    float ac = 0, bd = 0;
                    for (int e = 0; e < hd; ++e) { ac += qu[e] * k[e]; bd += qv[e] * p[e]; }
                    sc[j] = (ac + bd) * scale;
                    mx = std::max(mx, sc[j]);
                }
                float sum = 0;
                for (int j = 0; j < T; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                const float inv = 1.0f / sum;
                float* o = O.data() + (size_t)i * d + h * hd;
                for (int e = 0; e < hd; ++e) o[e] = 0;
                for (int j = 0; j < T; ++j) {
                    const float pj = sc[j] * inv;
                    const float* v = V.data() + (size_t)j * d + h * hd;
                    for (int e = 0; e < hd; ++e) o[e] += pj * v[e];
                }
            }
        });
        lap(4);
        apply(L.o, O.data(), T, Hn.data());
        add_inplace(R.data(), Hn.data(), (size_t)T * d);
        lap(3);
        // conv module: pw1 -> GLU -> depthwise(9) -> affine norm -> SiLU -> pw2
        layernorm(R.data(), T, d, L.norm_conv_w.data(), L.norm_conv_b.data(), I.a_eps, Hn.data());
        apply(L.pw1, Hn.data(), T, G.data());
        for (int t = 0; t < T; ++t) {
            float* g = G.data() + (size_t)t * 2 * d;
            for (int c = 0; c < d; ++c) g[c] = g[c] * (1.0f / (1.0f + std::exp(-g[d + c])));
        }
        const int kz = (int)(L.conv_dw_w.size() / d), half = (kz - 1) / 2;
        Pool::get().parallel_for(T, [&](int lo, int hi) {
            for (int t = lo; t < hi; ++t) {
                float* o = O.data() + (size_t)t * d;
                for (int c = 0; c < d; ++c) {
                    float acc = 0;
                    for (int k = 0; k < kz; ++k) {
                        const int s = t + k - half;
                        if (s < 0 || s >= T) continue;
                        acc += G[(size_t)s * 2 * d + c] * L.conv_dw_w[(size_t)c * kz + k];
                    }
                    acc += L.conv_dw_b[c];
                    acc = acc * L.conv_norm_w[c] + L.conv_norm_b[c];
                    o[c] = acc / (1.0f + std::exp(-acc));
                }
            }
        });
        lap(5);
        apply(L.pw2, O.data(), T, Hn.data());
        add_inplace(R.data(), Hn.data(), (size_t)T * d);
        lap(5);
        // macaron FFN 2 (half step) + output norm
        layernorm(R.data(), T, d, L.ffn_norm1_w.data(), L.ffn_norm1_b.data(), I.a_eps, Hn.data());
        apply(L.ffn_up1, Hn.data(), T, F.data());
        act(F.data(), (size_t)T * L.ffn_up1.rows, SILU);
        apply(L.ffn_down1, F.data(), T, Hn.data());
        add_inplace(R.data(), Hn.data(), (size_t)T * d, 0.5f);
        layernorm(R.data(), T, d, L.ln2w.data(), L.ln2b.data(), I.a_eps, X.data());
        lap(2);
    }
    // adapter: LN -> mlp.1 -> GELU(erf) -> mlp.3 ; residual block LN -> mlp.5 -> GELU -> mlp.6
    layernorm(X.data(), T, d, I.mm0w.data(), I.mm0b.data(), 1e-5f, Hn.data());
    std::vector<float> A((size_t)T * I.mma1.rows), out((size_t)T * I.mma3.rows);
    apply(I.mma1, Hn.data(), T, A.data());
    act(A.data(), A.size(), GELU_ERF);
    apply(I.mma3, A.data(), T, out.data());
    const int dout = I.mma3.rows;
    std::vector<float> Ln((size_t)T * dout), B((size_t)T * I.mma5.rows), Rb((size_t)T * dout);
    layernorm(out.data(), T, dout, I.mm4w.data(), I.mm4b.data(), 1e-5f, Ln.data());
    apply(I.mma5, Ln.data(), T, B.data());
    act(B.data(), B.size(), GELU_ERF);
    apply(I.mma6, B.data(), T, Rb.data());
    add_inplace(out.data(), Rb.data(), out.size());
    lap(6);
    if (prof) fprintf(stderr, "[mm-time] audio T=%d: conv stack %.0f  pre-out %.0f  ffn %.0f  qkv+pos+o %.0f  rel-pos attention %.0f  conv module %.0f  adapter %.0f ms\n",
                      T, tm[0] * 1e3, tm[1] * 1e3, tm[2] * 1e3, tm[3] * 1e3, tm[4] * 1e3, tm[5] * 1e3, tm[6] * 1e3);
    return out;
}

std::string find_mmproj_near(const std::string& model_path) {
    const size_t slash = model_path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : model_path.substr(0, slash);
    DIR* dp = opendir(dir.c_str());
    if (!dp) return "";
    std::string best;
    while (dirent* e = readdir(dp)) {
        const std::string n = e->d_name;
        if (n.rfind("mmproj", 0) == 0 && n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0)
            if (best.empty() || n < best) best = n;
    }
    closedir(dp);
    return best.empty() ? "" : dir + "/" + best;
}

}  // namespace ferro
