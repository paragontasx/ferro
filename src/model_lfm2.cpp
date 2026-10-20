// model_lfm2.cpp — LFM2 decision model (d1-omni): loader + CPU forward. See model_lfm2.h.
// Reference: llama.cpp src/models/lfm2.cpp (graph_decision + build_decision_head).
#include "model_lfm2.h"
#include "gpu_cl.h"
#include "threadpool.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
#include <functional>
#include <stdexcept>

namespace ferro {

// ------------------------------------------------------------------ loading helpers
static std::deque<std::string> g_names;   // WMat::name is a const char* — keep storage alive
static const char* keep(const std::string& s) { g_names.push_back(s); return g_names.back().c_str(); }

static std::vector<float> to_f32(const GgufFile& g, const GGUFTensor& t) {
    size_t n = 1;
    for (auto d : t.dims) n *= d;
    std::vector<float> v(n);
    const uint8_t* p = g.tensor_data(t);
    if (t.type == T_F32) memcpy(v.data(), p, n * 4);
    else if (t.type == T_F16) for (size_t i = 0; i < n; ++i) v[i] = fp16_to_f32(((const uint16_t*)p)[i]);
    else if (t.type == T_BF16) for (size_t i = 0; i < n; ++i) v[i] = bf16_to_f32(((const uint16_t*)p)[i]);
    else if (t.type == T_Q8_0) {
        const block_q8_0* b = (const block_q8_0*)p;
        for (size_t k = 0; k < n / QK8_0; ++k) {
            float s = fp16_to_f32((uint16_t)b[k].d);
            for (int i = 0; i < QK8_0; ++i) v[k * QK8_0 + i] = b[k].qs[i] * s;
        }
    } else throw std::runtime_error(t.name + ": unsupported tensor type " + std::to_string(t.type));
    return v;
}
static std::vector<float> vec(const GgufFile& g, const std::string& name) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return to_f32(g, *t);
}
// GEMM operand [rows = dims[1]][cols = dims[0]]; Q8_0 is kept bit-exact, other types are quantized
static WMat mat(const GgufFile& g, const std::string& name) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    int cols = (int)t->dims[0], rows = t->dims.size() > 1 ? (int)t->dims[1] : 1;
    if (t->type == T_Q8_0)
        return pack_q8(g.tensor_data(*t), rows, cols, (cols / QK8_0) * (int)sizeof(block_q8_0), keep(name));
    std::vector<float> f = to_f32(g, *t);
    return pack_from_f32(f.data(), rows, cols, cols, keep(name));
}

bool is_lfm2_decision(const GgufFile& g) {
    return g.arch == "lfm2" && g.get_int("lfm2.decision.block_count", 0) > 0;
}

bool LfmModel::load(const std::string& path, std::string* err) {
    try {
        if (!gguf.open_file(path.c_str(), err)) return false;
        if (!is_lfm2_decision(gguf)) { *err = "not an LFM2 decision model (lfm2.decision.block_count missing)"; return false; }
        const std::string P = "lfm2.";
        const int n_layer = (int)gguf.get_int(P + "block_count");
        n_head_layers = (int)gguf.get_int(P + "decision.block_count");
        n_trunk = n_layer - n_head_layers;
        n_embd = (int)gguf.get_int(P + "embedding_length");
        n_head = (int)gguf.get_int(P + "attention.head_count");
        head_dim = n_embd / n_head;
        rms_eps = (float)gguf.get_f64(P + "attention.layer_norm_rms_epsilon", 1e-5);
        ln_eps = (float)gguf.get_f64(P + "attention.layer_norm_epsilon", 1e-5);
        rope_base = gguf.get_f64(P + "rope.freq_base", 1e6);
        if (gguf.get_int(P + "shortconv.l_cache", 3) != 3) { *err = "only the 3-tap short conv is supported"; return false; }
        if (gguf.get_bool(P + "attention.causal", true)) { *err = "decision head requires a non-causal trunk"; return false; }
        auto ffs = gguf.get_iarr(P + "feed_forward_length");
        auto kvs = gguf.get_iarr(P + "attention.head_count_kv");
        if ((int)ffs.size() != n_layer || (int)kvs.size() != n_layer) { *err = "per-layer arrays have the wrong length"; return false; }
        n_ff = (int)ffs[0];
        n_vocab = (int)gguf.get_strarr("tokenizer.ggml.tokens").size();

        tok_embd = mat(gguf, "token_embd.weight");
        out_norm = vec(gguf, "token_embd_norm.weight");     // LLM_TENSOR_OUTPUT_NORM_LFM2
        type_embd = vec(gguf, "token_types.weight");
        if ((int)type_embd.size() != 3 * n_embd) { *err = "decision model needs 3 token types"; return false; }
        cls_norm_w = vec(gguf, "cls.norm.weight");  cls_norm_b = vec(gguf, "cls.norm.bias");
        cls = mat(gguf, "cls.weight");               cls_b = vec(gguf, "cls.bias");
        cls_out_w = vec(gguf, "cls.output.weight");  cls_out_b = vec(gguf, "cls.output.bias")[0];

        L.resize(n_trunk);
        for (int i = 0; i < n_trunk; ++i) {
            auto& l = L[i];
            const std::string b = "blk." + std::to_string(i) + ".";
            l.attn_norm = vec(gguf, b + "attn_norm.weight");
            l.ffn_norm = vec(gguf, b + "ffn_norm.weight");
            l.n_head_kv = (int)kvs[i];
            l.conv = l.n_head_kv == 0;
            if (l.conv) {
                l.in_proj = mat(gguf, b + "shortconv.in_proj.weight");
                l.out_proj = mat(gguf, b + "shortconv.out_proj.weight");
                std::vector<float> w = vec(gguf, b + "shortconv.conv.weight");   // [d][3]
                l.taps.resize(3 * n_embd);
                for (int c = 0; c < n_embd; ++c) for (int k = 0; k < 3; ++k) l.taps[k * n_embd + c] = w[c * 3 + k];
            } else {
                WMat q = mat(gguf, b + "attn_q.weight"), k = mat(gguf, b + "attn_k.weight"), v = mat(gguf, b + "attn_v.weight");
                l.wqkv = pack_concat({&q, &k, &v}, keep(b + "attn_qkv"));
                l.wo = mat(gguf, b + "attn_output.weight");
                l.q_norm = vec(gguf, b + "attn_q_norm.weight");
                l.k_norm = vec(gguf, b + "attn_k_norm.weight");
            }
            WMat gt = mat(gguf, b + "ffn_gate.weight"), up = mat(gguf, b + "ffn_up.weight");
            l.gateup = pack_concat({&gt, &up}, keep(b + "ffn_gateup"));
            l.down = mat(gguf, b + "ffn_down.weight");
            if ((int)ffs[i] != l.down.cols) { *err = "ffn size mismatch at layer " + std::to_string(i); return false; }
        }
        H.resize(n_head_layers);
        for (int j = 0; j < n_head_layers; ++j) {
            auto& h = H[j];
            const int i = n_trunk + j;
            const std::string b = "blk." + std::to_string(i) + ".";
            h.n_head = n_head;
            h.ln1_w = vec(gguf, b + "attn_norm.weight"); h.ln1_b = vec(gguf, b + "attn_norm.bias");
            h.wqkv = mat(gguf, b + "attn_qkv.weight");   h.qkv_b = vec(gguf, b + "attn_qkv.bias");
            h.wo = mat(gguf, b + "attn_output.weight");  h.wo_b = vec(gguf, b + "attn_output.bias");
            h.ln2_w = vec(gguf, b + "ffn_norm.weight");  h.ln2_b = vec(gguf, b + "ffn_norm.bias");
            h.up = mat(gguf, b + "ffn_up.weight");       h.up_b = vec(gguf, b + "ffn_up.bias");
            h.down = mat(gguf, b + "ffn_down.weight");   h.down_b = vec(gguf, b + "ffn_down.bias");
            h.n_ff = h.up.rows;
            if (kvs[i] != n_head || h.wqkv.rows != 3 * n_embd) { *err = "unexpected decision head geometry"; return false; }
        }
        rope.build(rope_base, head_dim, 16384 + 8);
    } catch (const std::exception& e) {
        *err = e.what();
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ elementwise kernels
__attribute__((target("avx2,fma"))) static inline __m256 exp256(__m256 x) {
    // exp(x) = 2^n * p(r), r in [-ln2/2, ln2/2]; ~1.5e-7 rel. error
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(88.0f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
// F[t][i] = silu(G[t][i]) * G[t][ff + i]
__attribute__((target("avx2,fma"))) static void silu_mul_row_avx2(const float* g, const float* u, float* f, int ff) {
    int i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= ff; i += 8) {
        __m256 x = _mm256_loadu_ps(g + i);
        __m256 s = _mm256_div_ps(x, _mm256_add_ps(one, exp256(_mm256_sub_ps(_mm256_setzero_ps(), x))));
        _mm256_storeu_ps(f + i, _mm256_mul_ps(s, _mm256_loadu_ps(u + i)));
    }
    for (; i < ff; ++i) f[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
}
static void silu_mul(const float* G, int T, int ff, float* F) {
    const bool avx = isa_level() >= 1;
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const float* g = G + (size_t)t * 2 * ff;
            float* f = F + (size_t)t * ff;
            if (avx) silu_mul_row_avx2(g, g + ff, f, ff);
            else for (int i = 0; i < ff; ++i) f[i] = g[i] / (1.0f + std::exp(-g[i])) * g[ff + i];
        }
    });
}
// y = layernorm(x) * w + b  (rows of d)
static void layernorm(const float* x, int T, int d, const float* w, const float* b, float eps, float* y, int ldx, int ldy) {
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const float* xr = x + (size_t)t * ldx;
            float* yr = y + (size_t)t * ldy;
            double mean = 0;
            for (int i = 0; i < d; ++i) mean += xr[i];
            mean /= d;
            double var = 0;
            for (int i = 0; i < d; ++i) { double v = xr[i] - mean; var += v * v; }
            var /= d;
            const float m = (float)mean, inv = (float)(1.0 / std::sqrt(var + eps));
            for (int i = 0; i < d; ++i) yr[i] = (xr[i] - m) * inv * w[i] + b[i];
        }
    });
}

// ------------------------------------------------------------------ debug
static void dbg_stats(const char* tag, int il, const float* x, size_t n) {
    double ss = 0; size_t nan = 0;
    for (size_t i = 0; i < n; ++i) { if (std::isnan(x[i]) || std::isinf(x[i])) ++nan; else ss += (double)x[i] * x[i]; }
    fprintf(stderr, "[dbg %s] layer %2d rms %.6g nan %zu first %.6g\n", tag, il, std::sqrt(ss / std::max<size_t>(1, n)), nan, x[0]);
}
static bool dbg_on() { static const bool v = getenv("FERRO_DEBUG_LAYERS") != nullptr; return v; }
// FERRO_TIME=1: per-stage wall time of the CPU forward
struct StageTimer {
    const bool on = getenv("FERRO_TIME") != nullptr;
    double acc[12] = {0};
    const char* names[12] = {"embd", "conv.in", "conv.gate", "conv.out", "attn.qkv", "attn.norm+rope", "attention",
                             "attn.wo", "ffn.gateup", "ffn.silu", "ffn.down", "head+scorer"};
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void mark(int k) { if (!on) return; auto t = std::chrono::steady_clock::now(); acc[k] += std::chrono::duration<double>(t - t0).count(); t0 = t; }
    void report() { if (!on) return; double tot = 0; for (double a : acc) tot += a;
        for (int k = 0; k < 12; ++k) fprintf(stderr, "[time] %-16s %8.1f ms (%4.1f%%)\n", names[k], acc[k] * 1e3, 100 * acc[k] / tot); }
};

// ------------------------------------------------------------------ engine
struct DecisionEngine::Impl {
    std::vector<float> X, X2, Hn, C, BX, O, G, F, Xm, Xm2, Om;
    // ---- iGPU state (weights resident, whole forward on-device) ----
    bool gpu = false;
    struct GL { gpu::Buf attn_norm = nullptr, ffn_norm = nullptr, taps = nullptr, q_norm = nullptr, k_norm = nullptr; };
    struct GH { gpu::Buf ln1w = nullptr, ln1b = nullptr, qkv_b = nullptr, wo_b = nullptr, ln2w = nullptr, ln2b = nullptr, up_b = nullptr, down_b = nullptr; };
    std::vector<GL> gl;
    std::vector<GH> gh;
    gpu::Buf cosb = nullptr, sinb = nullptr, out_norm = nullptr, type_embd = nullptr, cls_nw = nullptr, cls_nb = nullptr,
             cls_b = nullptr, cls_ow = nullptr;
    gpu::Buf bX = nullptr, bC = nullptr, bD = nullptr, bG = nullptr, bpos = nullptr, blo = nullptr, bhi = nullptr,
             btype = nullptr, bmarks = nullptr, bscores = nullptr, bplo = nullptr, bphi = nullptr, bprev = nullptr,
             bnext = nullptr, bhlo = nullptr, bhhi = nullptr;
    size_t Tcap = 0, Mcap = 0;
    void ensure(int T, int d, int ff, int ffh, int qkv) {
        auto grow = [](std::vector<float>& v, size_t n) { if (v.size() < n) v.resize(n); };
        grow(X, (size_t)T * d); grow(X2, (size_t)T * d); grow(Hn, (size_t)T * d); grow(C, (size_t)T * std::max(3 * d, qkv));
        grow(BX, (size_t)T * d); grow(O, (size_t)T * d);
        grow(G, (size_t)T * std::max(2 * ff, ffh)); grow(F, (size_t)T * std::max(ff, ffh));
    }
};

static gpu::Buf up(const std::vector<float>& v) { return gpu::upload(v.data(), v.size() * 4); }

bool DecisionEngine::init(const std::string& path, int n_threads, std::string* err) {
    Pool::get().init(n_threads);
    if (!m_.load(path, err)) return false;
    impl_ = std::make_shared<Impl>();
    Impl& in = *impl_;
    {
        // iGPU by default when an OpenCL GPU with int8 dot product is present; FERRO_GPU=0 / --cpu disables
        const char* ge = getenv("FERRO_GPU");
        if (!ge || atoi(ge) != 0) in.gpu = gpu::init(ge != nullptr || getenv("FERRO_VERBOSE"));
    }
    if (in.gpu) {
        bool ok = true;
        auto slot = [&](WMat& W) { W.gpu_slot = gpu::upload_q8(W); ok = ok && W.gpu_slot >= 0; };
        for (auto& l : m_.L) {
            if (l.conv) { slot(l.in_proj); slot(l.out_proj); } else { slot(l.wqkv); slot(l.wo); }
            slot(l.gateup); slot(l.down);
        }
        for (auto& h : m_.H) { slot(h.wqkv); slot(h.wo); slot(h.up); slot(h.down); }
        slot(m_.cls);
        in.gl.resize(m_.L.size()); in.gh.resize(m_.H.size());
        for (size_t i = 0; i < m_.L.size(); ++i) {
            const auto& l = m_.L[i]; auto& g = in.gl[i];
            g.attn_norm = up(l.attn_norm); g.ffn_norm = up(l.ffn_norm);
            if (l.conv) g.taps = up(l.taps); else { g.q_norm = up(l.q_norm); g.k_norm = up(l.k_norm); }
            ok = ok && g.attn_norm && g.ffn_norm && (l.conv ? g.taps != nullptr : (g.q_norm && g.k_norm));
        }
        for (size_t j = 0; j < m_.H.size(); ++j) {
            const auto& h = m_.H[j]; auto& g = in.gh[j];
            g.ln1w = up(h.ln1_w); g.ln1b = up(h.ln1_b); g.qkv_b = up(h.qkv_b); g.wo_b = up(h.wo_b);
            g.ln2w = up(h.ln2_w); g.ln2b = up(h.ln2_b); g.up_b = up(h.up_b); g.down_b = up(h.down_b);
            ok = ok && g.ln1w && g.ln1b && g.qkv_b && g.wo_b && g.ln2w && g.ln2b && g.up_b && g.down_b;
        }
        in.cosb = up(m_.rope.cosb); in.sinb = up(m_.rope.sinb);
        in.out_norm = up(m_.out_norm); in.type_embd = up(m_.type_embd);
        in.cls_nw = up(m_.cls_norm_w); in.cls_nb = up(m_.cls_norm_b); in.cls_b = up(m_.cls_b); in.cls_ow = up(m_.cls_out_w);
        ok = ok && in.cosb && in.sinb && in.out_norm && in.type_embd && in.cls_nw && in.cls_nb && in.cls_b && in.cls_ow;
        if (!ok) {
            fprintf(stderr, "[gpu] decision model upload incomplete -> CPU\n");
            in.gpu = false;
            for (auto& l : m_.L) for (WMat* W : {&l.in_proj, &l.out_proj, &l.wqkv, &l.wo, &l.gateup, &l.down}) W->gpu_slot = -1;
            for (auto& h : m_.H) for (WMat* W : {&h.wqkv, &h.wo, &h.up, &h.down}) W->gpu_slot = -1;
            m_.cls.gpu_slot = -1;
        } else if (getenv("FERRO_VERBOSE")) {
            fprintf(stderr, "[gpu] decision model resident on %s (%.1f MB)\n", gpu::device_name(), gpu::resident_bytes() / 1e6);
        }
    }
    const bool low_mem = getenv("FERRO_LOW_MEM") != nullptr || in.gpu;   // GPU-resident: no f32 planes (CPU fallback dequantizes on the fly)
    if (!low_mem) {
        auto f32 = [](WMat& W) { pack_q8_to_f32(W); };
        for (auto& l : m_.L) {
            if (l.conv) { f32(l.in_proj); f32(l.out_proj); } else { f32(l.wqkv); f32(l.wo); }
            f32(l.gateup); f32(l.down);
        }
        for (auto& h : m_.H) { f32(h.wqkv); f32(h.wo); f32(h.up); f32(h.down); }
        f32(m_.cls);
    }
    return true;
}

bool DecisionEngine::on_gpu() const { return impl_ && impl_->gpu; }

void DecisionEngine::score(const std::vector<DecisionSeq>& seqs, std::vector<std::vector<float>>& scores,
                           const std::vector<DecisionMediaEmbd>& media) {
    Impl& in = *impl_;
    const int d = m_.n_embd;
    scores.assign(seqs.size(), {});
    for (const auto& q : seqs)
        if (q.media >= (int)media.size() || (q.media >= 0 && media[q.media].n_tok != q.n_media))
            throw std::runtime_error("decision sequence refers to a missing media prefix");
    // sequences are bidirectional (each must be whole); pack them into batches of ~budget rows. A media prefix
    // is computed once per batch for all the question sequences of its request.
    static const int budget = getenv("FERRO_DECISION_BATCH") ? atoi(getenv("FERRO_DECISION_BATCH")) : 4096;
    size_t s0 = 0;
    while (s0 < seqs.size()) {
        size_t s1 = s0, nrow = 0;
        std::vector<int> mids;
        auto has = [&](int mi) { return std::find(mids.begin(), mids.end(), mi) != mids.end(); };
        while (s1 < seqs.size()) {
            const int mi = seqs[s1].media;
            const size_t c = seqs[s1].tokens.size() + (mi >= 0 && !has(mi) ? (size_t)media[mi].n_tok : 0);
            if (s1 > s0 && nrow + c > (size_t)budget) break;
            nrow += c;
            if (mi >= 0 && !has(mi)) mids.push_back(mi);
            ++s1;
        }
        DecisionBatch b;
        std::vector<int> mstart(media.size(), -1);
        auto row = [&](int pos, int lo, int hi, int plo, int phi, int prev, int next) {
            b.pos.push_back(pos); b.lo.push_back(lo); b.hi.push_back(hi); b.plo.push_back(plo); b.phi.push_back(phi);
            b.prev.push_back(prev); b.next.push_back(next);
        };
        for (int mi : mids) {
            const int n = media[mi].n_tok, base = b.Tm;
            mstart[mi] = base;
            b.membd.insert(b.membd.end(), media[mi].embd, media[mi].embd + (size_t)n * d);
            for (int i = 0; i < n; ++i) row(i, base, base + n - 1, 1, 0, i > 0 ? base + i - 1 : -1, i + 1 < n ? base + i + 1 : -1);
            b.Tm += n;
        }
        int tb = 0;   // text-row index
        for (size_t s = s0; s < s1; ++s) {
            const DecisionSeq& q = seqs[s];
            const int n = (int)q.tokens.size(), base = b.Tm + tb, mi = q.media;
            const int nm = mi >= 0 ? media[mi].n_tok : 0, ms = mi >= 0 ? mstart[mi] : -1;
            for (int i = 0; i < n; ++i) {
                b.ids.push_back(q.tokens[i]);
                row(nm + i, base, base + n - 1, mi >= 0 ? ms : 1, mi >= 0 ? ms + nm - 1 : 0,
                    i > 0 ? base + i - 1 : (mi >= 0 ? ms + nm - 1 : -1), i + 1 < n ? base + i + 1 : -1);
                b.type.push_back((int)q.type); b.hlo.push_back(tb); b.hhi.push_back(tb + n - 1);
            }
            for (int mk : q.markers) b.marks.push_back(tb + mk);
            tb += n;
        }
        b.T = b.Tm + tb;
        std::vector<float> out(b.marks.size());
        if (tb > 0) {
            bool done = in.gpu && forward_gpu(b, out.data());
            if (in.gpu && !done) fprintf(stderr, "[gpu] decision forward failed -> CPU\n");
            if (!done) forward_cpu(b, out.data());
        }
        size_t k = 0;
        for (size_t s = s0; s < s1; ++s)
            for (size_t j = 0; j < seqs[s].markers.size(); ++j) scores[s].push_back(out[k++]);
        s0 = s1;
    }
}

bool DecisionEngine::forward_gpu(const DecisionBatch& bt, float* out) {
    const LfmModel& m = m_;
    Impl& in = *impl_;
    const int T = bt.T, Tm = bt.Tm, Tt = T - Tm;
    const int d = m.n_embd, hd = m.head_dim, nq = m.n_head, ff = m.n_ff, M = (int)bt.marks.size();
    const int ffh = m.H.empty() ? 0 : m.H[0].n_ff;
    const float scale = 1.0f / std::sqrt((float)hd);
    if ((size_t)T > in.Tcap) {
        gpu::Buf* all[] = {&in.bX, &in.bC, &in.bD, &in.bG, &in.bpos, &in.blo, &in.bhi, &in.btype, &in.bplo, &in.bphi,
                           &in.bprev, &in.bnext, &in.bhlo, &in.bhhi};
        for (gpu::Buf* b : all) { gpu::release(*b); *b = nullptr; }
        const size_t t = (size_t)T;
        in.bX = gpu::alloc(t * d * 4); in.bC = gpu::alloc(t * 3 * d * 4); in.bD = gpu::alloc(t * d * 4);
        in.bG = gpu::alloc(t * std::max(2 * ff, ffh) * 4);
        for (gpu::Buf* b : {&in.bpos, &in.blo, &in.bhi, &in.btype, &in.bplo, &in.bphi, &in.bprev, &in.bnext, &in.bhlo, &in.bhhi})
            *b = gpu::alloc(t * 4);
        for (gpu::Buf* b : all) if (!*b) { in.Tcap = 0; return false; }
        in.Tcap = t;
    }
    if ((size_t)M > in.Mcap || !in.bmarks) {
        gpu::release(in.bmarks); gpu::release(in.bscores);
        in.Mcap = std::max<size_t>(M, 64);
        in.bmarks = gpu::alloc(in.Mcap * 4); in.bscores = gpu::alloc(in.Mcap * 4);
        if (!in.bmarks || !in.bscores) { in.Mcap = 0; return false; }
    }
    if (!gpu::ensure_qscratch(T, std::max({3 * d, 2 * ff, ffh, d}))) return false;
    // host side: media rows + text embedding gather
    in.X.resize((size_t)T * d);
    if (Tm) memcpy(in.X.data(), bt.membd.data(), (size_t)Tm * d * 4);
    gather_embd(m.tok_embd, bt.ids.data(), Tt, 1.0f, in.X.data() + (size_t)Tm * d);
    long w_trunk = 0, w_head = 0;
    for (int t = 0; t < T; ++t) w_trunk += bt.hi[t] - bt.lo[t] + 1 + (bt.phi[t] >= bt.plo[t] ? bt.phi[t] - bt.plo[t] + 1 : 0);
    for (int t = 0; t < Tt; ++t) w_head += bt.hhi[t] - bt.hlo[t] + 1;
    const bool pre = bt.has_prefix();
    bool ok = gpu::write(in.bX, in.X.data(), (size_t)T * d * 4) && gpu::write(in.bpos, bt.pos.data(), (size_t)T * 4) &&
              gpu::write(in.blo, bt.lo.data(), (size_t)T * 4) && gpu::write(in.bhi, bt.hi.data(), (size_t)T * 4) &&
              gpu::write(in.bprev, bt.prev.data(), (size_t)T * 4) && gpu::write(in.bnext, bt.next.data(), (size_t)T * 4) &&
              gpu::write(in.btype, bt.type.data(), (size_t)Tt * 4) && gpu::write(in.bmarks, bt.marks.data(), (size_t)M * 4) &&
              gpu::write(in.bhlo, bt.hlo.data(), (size_t)Tt * 4) && gpu::write(in.bhhi, bt.hhi.data(), (size_t)Tt * 4);
    if (pre) ok = ok && gpu::write(in.bplo, bt.plo.data(), (size_t)T * 4) && gpu::write(in.bphi, bt.phi.data(), (size_t)T * 4);
    const gpu::Buf PLO = pre ? in.bplo : nullptr, PHI = pre ? in.bphi : nullptr;
    static const int sync_mode = getenv("FERRO_GPU_SYNC") ? atoi(getenv("FERRO_GPU_SYNC")) : 0;   // debug: 1 = after uploads, 2 = every layer
    if (sync_mode >= 1) ok = ok && gpu::finish();
    const gpu::Buf X = in.bX, C = in.bC, D = in.bD, G = in.bG;
    gpu::set_attn_work(w_trunk * m.n_head);
    ok = ok && gpu::op_resid_norm_q(X, nullptr, nullptr, in.gl[0].attn_norm, nullptr, false, T, d, m.rms_eps);
    for (int il = 0; ok && il < m.n_trunk; ++il) {
        const LfmTrunkLayer& l = m.L[il];
        const Impl::GL& g = in.gl[il];
        if (l.conv) {
            ok = gpu::op_gemm_q(l.in_proj.gpu_slot, T, C)
              && gpu::op_conv_gate_q(C, in.bprev, in.bnext, g.taps, T, d)
              && gpu::op_gemm_q(l.out_proj.gpu_slot, T, D);
        } else {
            const int nkv = l.n_head_kv, rowc = l.wqkv.rows;
            ok = gpu::op_gemm_q(l.wqkv.gpu_slot, T, C)
              && gpu::op_qkv_post(C, rowc, T, nq, nkv, hd, g.q_norm, g.k_norm, in.cosb, in.sinb, in.bpos, hd / 2, m.rms_eps, false)
              && gpu::op_attention_q(C, rowc, 0, nq * hd, nq * hd + nkv * hd, T, nq, nkv, hd, in.blo, in.bhi, 0, scale, PLO, PHI)
              && (sync_mode != 3 || gpu::finish())
              && gpu::op_gemm_q(l.wo.gpu_slot, T, D);
        }
        ok = ok && gpu::op_resid_norm_q(X, D, nullptr, g.ffn_norm, nullptr, false, T, d, m.rms_eps)
                && gpu::op_gemm_q(l.gateup.gpu_slot, T, G)
                && gpu::op_silu_mul_q(G, T, ff)
                && gpu::op_gemm_q(l.down.gpu_slot, T, D);
        if (!ok) break;
        if (il + 1 < m.n_trunk) {
            ok = gpu::op_resid_norm_q(X, D, nullptr, in.gl[il + 1].attn_norm, nullptr, false, T, d, m.rms_eps);
        } else {
            if (Tm) {
                // the head reads text rows only: move rows [Tm, T) of X and D to the front (via free scratch)
                const size_t rb = (size_t)d * 4;
                ok = gpu::copy(G, 0, X, (size_t)Tm * rb, (size_t)Tt * rb) && gpu::copy(C, 0, D, (size_t)Tm * rb, (size_t)Tt * rb)
                  && gpu::copy(X, 0, G, 0, (size_t)Tt * rb) && gpu::copy(D, 0, C, 0, (size_t)Tt * rb);
            }
            ok = ok && gpu::op_head_in_q(X, D, in.out_norm, in.type_embd, in.btype, in.gh[0].ln1w, in.gh[0].ln1b, Tt, d, m.rms_eps, m.ln_eps);
        }
        if (ok && sync_mode >= 2) ok = gpu::finish();
        if (ok && dbg_on() && il + 1 < m.n_trunk) {
            std::vector<float> h((size_t)T * d);
            ok = gpu::read(X, h.data(), h.size() * 4);
            dbg_stats("gpu", il, h.data(), h.size());
        }
    }
    gpu::set_attn_work(w_head * m.n_head);
    for (int j = 0; ok && j < m.n_head_layers; ++j) {
        const LfmHeadLayer& h = m.H[j];
        const Impl::GH& g = in.gh[j];
        const bool last = j + 1 == m.n_head_layers;
        ok = gpu::op_gemm_q(h.wqkv.gpu_slot, Tt, C)
          && gpu::op_bias_add(C, g.qkv_b, Tt, 3 * d)
          && gpu::op_attention_q(C, 3 * d, 0, d, 2 * d, Tt, h.n_head, h.n_head, hd, in.bhlo, in.bhhi, 0, scale)
          && (sync_mode != 3 || gpu::finish())
          && gpu::op_gemm_q(h.wo.gpu_slot, Tt, D)
          && gpu::op_resid_norm_q(X, D, g.wo_b, g.ln2w, g.ln2b, true, Tt, d, m.ln_eps)
          && gpu::op_gemm_q(h.up.gpu_slot, Tt, G)
          && gpu::op_bias_act_q(G, g.up_b, true, Tt, h.n_ff)
          && gpu::op_gemm_q(h.down.gpu_slot, Tt, D)
          && gpu::op_resid_norm_q(X, D, g.down_b, last ? in.cls_nw : in.gh[j + 1].ln1w, last ? in.cls_nb : in.gh[j + 1].ln1b,
                                  true, Tt, d, m.ln_eps);
    }
    ok = ok && gpu::op_gemm_q(m.cls.gpu_slot, Tt, C)
            && gpu::op_score_out(C, in.cls_b, in.cls_ow, m.cls_out_b, in.bmarks, in.bscores, M, d)
            && gpu::read(in.bscores, out, (size_t)M * 4);
    if (ok && getenv("FERRO_GPU_KBENCH")) {
        // re-launch each op on the live buffers (rusticl event timing is unreliable); ms per forward
        const int R = 10;
        double tot = 0;
        auto tb = [&](const char* name, int per_fwd, const std::function<void()>& f) {
            gpu::finish();
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < R; ++r) f();
            gpu::finish();
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / R;
            tot += ms * per_fwd;
            fprintf(stderr, "[kbench] %-22s %7.3f ms x%2d = %8.2f ms/fwd\n", name, ms, per_fwd, ms * per_fwd);
        };
        int nconv = 0, nattn = 0, ic = -1, ia = -1;
        for (int il = 0; il < m.n_trunk; ++il) { if (m.L[il].conv) { ++nconv; if (ic < 0) ic = il; } else { ++nattn; if (ia < 0) ia = il; } }
        const auto& Lc = m.L[ic]; const auto& La = m.L[ia]; const auto& H0 = m.H[0];
        const int rowc = La.wqkv.rows, nkv = La.n_head_kv;
        gpu::set_attn_work(w_trunk * m.n_head);
        tb("gemm conv.in", nconv, [&] { gpu::op_gemm_q(Lc.in_proj.gpu_slot, T, C); });
        tb("conv gate", nconv, [&] { gpu::op_conv_gate_q(C, in.bprev, in.bnext, in.gl[ic].taps, T, d); });
        tb("gemm conv.out", nconv, [&] { gpu::op_gemm_q(Lc.out_proj.gpu_slot, T, D); });
        tb("gemm attn.qkv", nattn, [&] { gpu::op_gemm_q(La.wqkv.gpu_slot, T, C); });
        tb("qkv norm+rope", nattn, [&] { gpu::op_qkv_post(C, rowc, T, nq, nkv, hd, in.gl[ia].q_norm, in.gl[ia].k_norm, in.cosb, in.sinb, in.bpos, hd / 2, m.rms_eps, false); });
        tb("attention (trunk)", nattn, [&] { gpu::op_attention_q(C, rowc, 0, nq * hd, nq * hd + nkv * hd, T, nq, nkv, hd, in.blo, in.bhi, 0, scale, PLO, PHI); });
        tb("gemm attn.wo", nattn, [&] { gpu::op_gemm_q(La.wo.gpu_slot, T, D); });
        tb("resid+rmsnorm+q", 2 * m.n_trunk, [&] { gpu::op_resid_norm_q(X, nullptr, nullptr, in.gl[0].ffn_norm, nullptr, false, T, d, m.rms_eps); });
        tb("gemm ffn.gateup", m.n_trunk, [&] { gpu::op_gemm_q(Lc.gateup.gpu_slot, T, G); });
        tb("silu*up+q", m.n_trunk, [&] { gpu::op_silu_mul_q(G, T, ff); });
        tb("gemm ffn.down", m.n_trunk, [&] { gpu::op_gemm_q(Lc.down.gpu_slot, T, D); });
        gpu::set_attn_work(w_head * m.n_head);
        tb("head gemm qkv", m.n_head_layers, [&] { gpu::op_gemm_q(H0.wqkv.gpu_slot, Tt, C); });
        tb("head attention", m.n_head_layers, [&] { gpu::op_attention_q(C, 3 * d, 0, d, 2 * d, Tt, H0.n_head, H0.n_head, hd, in.bhlo, in.bhhi, 0, scale); });
        tb("head gemm wo", m.n_head_layers, [&] { gpu::op_gemm_q(H0.wo.gpu_slot, Tt, D); });
        tb("head gemm up", m.n_head_layers, [&] { gpu::op_gemm_q(H0.up.gpu_slot, Tt, G); });
        tb("head bias+relu+q", m.n_head_layers, [&] { gpu::op_bias_act_q(G, in.gh[0].up_b, true, Tt, H0.n_ff); });
        tb("head gemm down", m.n_head_layers, [&] { gpu::op_gemm_q(H0.down.gpu_slot, Tt, D); });
        tb("cls gemm", 1, [&] { gpu::op_gemm_q(m.cls.gpu_slot, Tt, C); });
        fprintf(stderr, "[kbench] total %.1f ms/fwd for T=%d (%.0f tok/s device-bound)\n", tot, T, T / tot * 1e3);
    }
    return ok;
}

// CPU attention with per-query key ranges: prefix [plo, phi] (if plo <= phi) then own [lo, hi]
static void attention_ranges(const float* C, int ldc, int qoff, int koff, int voff, float* O, int ldo, int T, int nq,
                             int nkv, int hd, float scale, const int* lo, const int* hi, const int* plo, const int* phi) {
    const int g = nq / nkv;
    Pool::get().parallel_for(nq * T, [&](int a, int bnd) {
        std::vector<float> sc;
        std::vector<int> keys;
        for (int idx = a; idx < bnd; ++idx) {
            const int t = idx / nq, h = idx % nq, kh = h / g;
            keys.clear();
            if (plo[t] <= phi[t]) for (int j = plo[t]; j <= phi[t]; ++j) keys.push_back(j);
            for (int j = lo[t]; j <= hi[t]; ++j) keys.push_back(j);
            sc.resize(keys.size());
            const float* q = C + (size_t)t * ldc + qoff + h * hd;
            float mx = -1e30f;
            for (size_t k = 0; k < keys.size(); ++k) {
                const float* kr = C + (size_t)keys[k] * ldc + koff + kh * hd;
                float acc = 0;
                for (int e = 0; e < hd; ++e) acc += q[e] * kr[e];
                sc[k] = acc * scale;
                mx = std::max(mx, sc[k]);
            }
            float sum = 0;
            for (auto& v : sc) { v = std::exp(v - mx); sum += v; }
            const float inv = 1.0f / sum;
            float* o = O + (size_t)t * ldo + h * hd;
            for (int e = 0; e < hd; ++e) o[e] = 0;
            for (size_t k = 0; k < keys.size(); ++k) {
                const float p = sc[k] * inv;
                const float* vr = C + (size_t)keys[k] * ldc + voff + kh * hd;
                for (int e = 0; e < hd; ++e) o[e] += p * vr[e];
            }
        }
    });
}

void DecisionEngine::forward_cpu(const DecisionBatch& bt, float* out) {
    const LfmModel& m = m_;
    Impl& in = *impl_;
    const int T = bt.T, Tm = bt.Tm, Tt = T - Tm;
    const int d = m.n_embd, hd = m.head_dim, nq = m.n_head;
    const int ffh = m.H.empty() ? 0 : m.H[0].n_ff;
    in.ensure(T, d, m.n_ff, ffh, 3 * d);
    float *X = in.X.data(), *X2 = in.X2.data(), *Hn = in.Hn.data(), *C = in.C.data(), *BX = in.BX.data(), *O = in.O.data();
    float *G = in.G.data(), *F = in.F.data();
    const float scale = 1.0f / std::sqrt((float)hd);
    const int* pos = bt.pos.data();
    const bool pre = bt.has_prefix();

    StageTimer tm;
    if (Tm) memcpy(X, bt.membd.data(), (size_t)Tm * d * 4);
    gather_embd(m.tok_embd, bt.ids.data(), Tt, 1.0f, X + (size_t)Tm * d);
    tm.mark(0);

    for (int il = 0; il < m.n_trunk; ++il) {
        const LfmTrunkLayer& l = m.L[il];
        rmsnorm_q(X, T, d, l.attn_norm.data(), m.rms_eps, Hn);
        if (l.conv) {
            gemm_q8_fma(l.in_proj, Hn, T, nullptr, nullptr, C, 3 * d);          // b | c | x
            tm.mark(1);
            Pool::get().parallel_for(T, [&](int lo, int hi) {
                for (int t = lo; t < hi; ++t) {
                    const float* b = C + (size_t)t * 3 * d; const float* x = b + 2 * d;
                    float* bx = BX + (size_t)t * d;
                    for (int i = 0; i < d; ++i) bx[i] = b[i] * x[i];
                }
            });
            const float *t0 = l.taps.data(), *t1 = t0 + d, *t2 = t1 + d;
            Pool::get().parallel_for(T, [&](int lo, int hi) {
                for (int t = lo; t < hi; ++t) {
                    const float* c = C + (size_t)t * 3 * d + d;
                    const float* bx = BX + (size_t)t * d;
                    float* o = O + (size_t)t * d;
                    for (int i = 0; i < d; ++i) o[i] = bx[i] * t1[i];
                    if (bt.prev[t] >= 0) { const float* p = BX + (size_t)bt.prev[t] * d; for (int i = 0; i < d; ++i) o[i] += p[i] * t0[i]; }
                    if (bt.next[t] >= 0) { const float* n = BX + (size_t)bt.next[t] * d; for (int i = 0; i < d; ++i) o[i] += n[i] * t2[i]; }
                    for (int i = 0; i < d; ++i) o[i] *= c[i];
                }
            });
            tm.mark(2);
            gemm_q8_fma(l.out_proj, O, T, nullptr, X, X2, d); std::swap(X, X2);
            tm.mark(3);
        } else {
            const int nkv = l.n_head_kv, rowc = l.wqkv.rows, qsz = nq * hd, ksz = nkv * hd;
            gemm_q8_fma(l.wqkv, Hn, T, nullptr, nullptr, C, rowc);
            tm.mark(4);
            for (int h = 0; h < nq; ++h) rmsnorm_q(C + h * hd, T, hd, l.q_norm.data(), m.rms_eps, C + h * hd, rowc, rowc);
            for (int h = 0; h < nkv; ++h) rmsnorm_q(C + qsz + h * hd, T, hd, l.k_norm.data(), m.rms_eps, C + qsz + h * hd, rowc, rowc);
            rope_apply(C, T, nq, hd, m.rope, pos, rowc);
            rope_apply(C + qsz, T, nkv, hd, m.rope, pos, rowc);
            tm.mark(5);
            if (pre) attention_ranges(C, rowc, 0, qsz, qsz + ksz, O, d, T, nq, nkv, hd, scale, bt.lo.data(), bt.hi.data(), bt.plo.data(), bt.phi.data());
            else attention(C, C + qsz, C + qsz + ksz, O, T, nq, nkv, hd, scale, bt.lo.data(), -1, rowc, rowc, d);   // lo = segment id
            tm.mark(6);
            gemm_q8_fma(l.wo, O, T, nullptr, X, X2, d); std::swap(X, X2);
            tm.mark(7);
        }
        rmsnorm_q(X, T, d, l.ffn_norm.data(), m.rms_eps, Hn);
        gemm_q8_fma(l.gateup, Hn, T, nullptr, nullptr, G, 2 * m.n_ff);
        tm.mark(8);
        silu_mul(G, T, m.n_ff, F);
        tm.mark(9);
        gemm_q8_fma(l.down, F, T, nullptr, X, X2, d); std::swap(X, X2);
        tm.mark(10);
        if (dbg_on()) dbg_stats("cpu", il, X, (size_t)T * d);
        if (const char* dp = getenv("FERRO_DUMP_LAYERS")) {   // debug: trunk layer outputs, T x d f32 per layer
            FILE* f = fopen(dp, il == 0 ? "wb" : "ab");
            if (f) { fwrite(X, 4, (size_t)T * d, f); fclose(f); }
        }
    }
    // ---- decision head: text rows only
    if (Tm) memmove(X, X + (size_t)Tm * d, (size_t)Tt * d * 4);
    rmsnorm_q(X, Tt, d, m.out_norm.data(), m.rms_eps, X);
    if (const char* dp = getenv("FERRO_DUMP_HEADIN")) { FILE* f = fopen(dp, "wb"); if (f) { fwrite(X, 4, (size_t)Tt * d, f); fclose(f); } }
    for (int t = 0; t < Tt; ++t) {
        const float* te = m.type_embd.data() + (size_t)bt.type[t] * d;
        float* x = X + (size_t)t * d;
        for (int i = 0; i < d; ++i) x[i] += te[i];
    }
    const std::vector<int>& marks = bt.marks;
    const int M = (int)marks.size();
    const int* hseg = bt.hlo.data();
    in.Xm.resize((size_t)M * d); in.Xm2.resize((size_t)M * d); in.Om.resize((size_t)M * d);
    float* Xm = in.Xm.data(); float* Xm2 = in.Xm2.data();
    for (int j = 0; j < m.n_head_layers; ++j) {
        const LfmHeadLayer& h = m.H[j];
        const bool last = j + 1 == m.n_head_layers;
        layernorm(X, Tt, d, h.ln1_w.data(), h.ln1_b.data(), m.ln_eps, Hn, d, d);
        gemm_q8_fma(h.wqkv, Hn, Tt, h.qkv_b.data(), nullptr, C, 3 * d);
        attention(C, C + d, C + 2 * d, O, Tt, h.n_head, h.n_head, hd, scale, hseg, -1, 3 * d, 3 * d, d);
        if (!last) {
            gemm_q8_fma(h.wo, O, Tt, h.wo_b.data(), X, X2, d); std::swap(X, X2);
            layernorm(X, Tt, d, h.ln2_w.data(), h.ln2_b.data(), m.ln_eps, Hn, d, d);
            gemm_q8_fma(h.up, Hn, Tt, h.up_b.data(), nullptr, G, h.n_ff);
            for (size_t i = 0, n = (size_t)Tt * h.n_ff; i < n; ++i) G[i] = std::max(G[i], 0.0f);
            gemm_q8_fma(h.down, G, Tt, h.down_b.data(), X, X2, d); std::swap(X, X2);
        } else {
            // the last head block only matters at the option markers
            for (int k = 0; k < M; ++k) {
                memcpy(in.Om.data() + (size_t)k * d, O + (size_t)marks[k] * d, d * 4);
                memcpy(Xm + (size_t)k * d, X + (size_t)marks[k] * d, d * 4);
            }
            float* Om = in.Om.data();
            gemm_q8_fma(h.wo, Om, M, h.wo_b.data(), Xm, Xm2, d); std::swap(Xm, Xm2);
            layernorm(Xm, M, d, h.ln2_w.data(), h.ln2_b.data(), m.ln_eps, Hn, d, d);
            gemm_q8_fma(h.up, Hn, M, h.up_b.data(), nullptr, G, h.n_ff);
            for (size_t i = 0, n = (size_t)M * h.n_ff; i < n; ++i) G[i] = std::max(G[i], 0.0f);
            gemm_q8_fma(h.down, G, M, h.down_b.data(), Xm, Xm2, d); std::swap(Xm, Xm2);
        }
    }
    if (m.n_head_layers == 0) throw std::runtime_error("no decision head");
    // ---- scorer: LayerNorm -> Linear -> GELU(erf) -> Linear(1)
    layernorm(Xm, M, d, m.cls_norm_w.data(), m.cls_norm_b.data(), m.ln_eps, Hn, d, d);
    gemm_q8_fma(m.cls, Hn, M, m.cls_b.data(), nullptr, O, d);
    for (int k = 0; k < M; ++k) {
        const float* o = O + (size_t)k * d;
        double acc = m.cls_out_b;
        for (int i = 0; i < d; ++i) {
            float v = o[i];
            acc += (double)(0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f))) * m.cls_out_w[i];
        }
        out[k] = (float)acc;
        if (getenv("FERRO_PRINT_SCORES")) fprintf(stderr, "[score] marker row %d (type %d): %.5f\n", marks[k], bt.type[marks[k]], out[k]);
    }
    tm.mark(11);
    tm.report();
}

}  // namespace ferro
