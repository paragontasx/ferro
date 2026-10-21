// model_lfm2_lm.cpp - see model_lfm2_lm.h. Reference: llama.cpp src/models/lfm2.cpp (graph<false>, causal path).
#include "model_lfm2_lm.h"
#include "gpu_cl.h"
#include "threadpool.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
#include <map>
#include <stdexcept>

namespace ferro {

// ------------------------------------------------------------------ loading (helpers mirror model_lfm2.cpp)
static std::deque<std::string> g_lm_names;
static const char* keep_name(const std::string& s) { g_lm_names.push_back(s); return g_lm_names.back().c_str(); }

static std::vector<float> lm_to_f32(const GgufFile& g, const GGUFTensor& t) {
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
            const float s = fp16_to_f32((uint16_t)b[k].d);
            for (int i = 0; i < QK8_0; ++i) v[k * QK8_0 + i] = b[k].qs[i] * s;
        }
    } else throw std::runtime_error(t.name + ": unsupported tensor type " + std::to_string(t.type) + " (this build reads Q8_0, F16, BF16 and F32 GGUFs)");
    return v;
}
static std::vector<float> lm_vec(const GgufFile& g, const std::string& name) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return lm_to_f32(g, *t);
}
static WMat lm_mat(const GgufFile& g, const std::string& name) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    const int cols = (int)t->dims[0], rows = t->dims.size() > 1 ? (int)t->dims[1] : 1;
    if (t->type == T_Q8_0) return pack_q8(g.tensor_data(*t), rows, cols, (cols / QK8_0) * (int)sizeof(block_q8_0), keep_name(name));
    std::vector<float> f = lm_to_f32(g, *t);
    return pack_from_f32(f.data(), rows, cols, cols, keep_name(name));
}

bool is_lfm2_lm(const GgufFile& g) {
    return g.arch == "lfm2" && g.get_int("lfm2.decision.block_count", 0) == 0 && g.get_bool("lfm2.attention.causal", true);
}

bool LmModel::load(const std::string& path, int max_ctx, std::string* err) {
    try {
        if (!gguf.open_file(path.c_str(), err)) return false;
        if (!is_lfm2_lm(gguf)) { *err = "not a causal LFM2 language model (arch lfm2 without a decision head expected)"; return false; }
        const std::string P = "lfm2.";
        n_layer = (int)gguf.get_int(P + "block_count");
        n_embd = (int)gguf.get_int(P + "embedding_length");
        n_head = (int)gguf.get_int(P + "attention.head_count");
        head_dim = n_embd / n_head;
        n_ctx_train = (int)gguf.get_int(P + "context_length", 32768);
        rms_eps = (float)gguf.get_f64(P + "attention.layer_norm_rms_epsilon", 1e-5);
        rope_base = gguf.get_f64(P + "rope.freq_base", 1e6);
        if (gguf.get_int(P + "shortconv.l_cache", 3) != 3) { *err = "only the 3-tap short conv is supported"; return false; }
        auto ffs = gguf.get_iarr(P + "feed_forward_length");
        auto kvs = gguf.get_iarr(P + "attention.head_count_kv");
        if ((int)kvs.size() != n_layer) { *err = "attention.head_count_kv must be a per-layer array"; return false; }
        n_ff = ffs.empty() ? (int)gguf.get_int(P + "feed_forward_length") : (int)ffs[0];
        n_vocab = (int)gguf.get_strarr("tokenizer.ggml.tokens").size();
        bos_id = (int)gguf.get_int("tokenizer.ggml.bos_token_id", 1);
        eos_id = (int)gguf.get_int("tokenizer.ggml.eos_token_id", 7);
        tok_embd = lm_mat(gguf, "token_embd.weight");
        if (tok_embd.rows != n_vocab || tok_embd.cols != n_embd) { *err = "token_embd has an unexpected shape"; return false; }
        out_norm = lm_vec(gguf, "token_embd_norm.weight");
        L.resize(n_layer);
        n_kv = 0;
        for (int i = 0; i < n_layer; ++i) {
            LmLayer& l = L[i];
            const std::string b = "blk." + std::to_string(i) + ".";
            l.attn_norm = lm_vec(gguf, b + "attn_norm.weight");
            l.ffn_norm = lm_vec(gguf, b + "ffn_norm.weight");
            l.conv = kvs[i] == 0;
            if (l.conv) {
                ++n_conv_layers;
                l.in_proj = lm_mat(gguf, b + "shortconv.in_proj.weight");
                l.out_proj = lm_mat(gguf, b + "shortconv.out_proj.weight");
                std::vector<float> w = lm_vec(gguf, b + "shortconv.conv.weight");     // ggml [3][d] (ne0 = 3 fastest)
                l.taps.resize(3 * (size_t)n_embd);
                for (int c = 0; c < n_embd; ++c) for (int k = 0; k < 3; ++k) l.taps[(size_t)k * n_embd + c] = w[(size_t)c * 3 + k];
            } else {
                ++n_attn_layers;
                if (n_kv && n_kv != (int)kvs[i]) { *err = "per-layer KV head counts differ (unsupported)"; return false; }
                n_kv = (int)kvs[i];
                WMat q = lm_mat(gguf, b + "attn_q.weight"), k = lm_mat(gguf, b + "attn_k.weight"), v = lm_mat(gguf, b + "attn_v.weight");
                l.wqkv = pack_concat({&q, &k, &v}, keep_name(b + "attn_qkv"));
                l.wo = lm_mat(gguf, b + "attn_output.weight");
                l.q_norm = lm_vec(gguf, b + "attn_q_norm.weight");
                l.k_norm = lm_vec(gguf, b + "attn_k_norm.weight");
            }
            WMat gt = lm_mat(gguf, b + "ffn_gate.weight"), up = lm_mat(gguf, b + "ffn_up.weight");
            l.gateup = pack_concat({&gt, &up}, keep_name(b + "ffn_gateup"));
            l.down = lm_mat(gguf, b + "ffn_down.weight");
            if (l.down.cols != n_ff) { *err = "feed-forward size mismatch at layer " + std::to_string(i); return false; }
        }
        rope.build(rope_base, head_dim, max_ctx + 8);
    } catch (const std::exception& e) { *err = e.what(); return false; }
    return true;
}

// ------------------------------------------------------------------ engine
struct LmEngine::Impl {
    // ---- iGPU state ----
    struct Gpu {
        bool ok = false;
        int slots = 1, cap = 0, Tmax = 0;
        std::vector<gpu::Buf> an, fn, taps, qn, kn, kv, kvs, cst;   // per layer (an/fn), per conv layer (taps, cst), per attention layer (qn, kn, kv)
        gpu::Buf out_norm = nullptr, cosb = nullptr, sinb = nullptr, zero = nullptr;
        gpu::Buf X = nullptr, X2 = nullptr, D = nullptr, C = nullptr, G = nullptr, Xl = nullptr, Dl = nullptr, logits = nullptr;
        gpu::Buf bpos = nullptr, bdst = nullptr, bsrc1 = nullptr, bsrc2 = nullptr, bslot = nullptr, br1 = nullptr, br2 = nullptr, bbase = nullptr, blen = nullptr;
        std::vector<int> free_slots;
        std::vector<float> hostX;
        std::vector<int> pos, dst, src1, src2, slot, r1, r2, base, len;      // staging (must outlive the enqueued writes)
        int max_ns = 0;
    } g;
    std::vector<float> X, X2, Hn, C, BX, O, G, F, last, lastn;
    std::vector<int> pos, row_seq;
    void ensure(int T, int d, int ff, int qkv) {
        auto grow = [](std::vector<float>& v, size_t n) { if (v.size() < n) v.resize(n); };
        grow(X, (size_t)T * d); grow(X2, (size_t)T * d); grow(Hn, (size_t)T * d); grow(O, (size_t)T * d); grow(BX, (size_t)T * d);
        grow(C, (size_t)T * std::max(3 * d, qkv)); grow(G, (size_t)T * 2 * ff); grow(F, (size_t)T * ff);
    }
};

// ------------------------------------------------------------------ iGPU path
static bool gpu_setup(LmEngine::Impl& in, LmModel& m, int max_ctx, int n_slots, bool kv_int8_) {
    auto& g = in.g;
    const int d = m.n_embd, hd = m.head_dim, ff = m.n_ff, kvw = 2 * m.n_kv * hd, rowc = m.n_head * hd + kvw;
    if (hd != 64 || d % 32 || d / 4 > 512 || ff % 32 || ff / 8 > 1024 || m.n_head % m.n_kv) return false;
    bool ok = true;
    auto slot = [&](WMat& W) { W.gpu_slot = gpu::upload_q8(W); ok = ok && W.gpu_slot >= 0; };
    for (auto& l : m.L) { if (l.conv) { slot(l.in_proj); slot(l.out_proj); } else { slot(l.wqkv); slot(l.wo); } slot(l.gateup); slot(l.down); }
    slot(m.tok_embd);
    if (!ok) return false;
    auto up = [&](const std::vector<float>& v) { gpu::Buf b = gpu::upload(v.data(), v.size() * 4); ok = ok && b; return b; };
    for (auto& l : m.L) {
        g.an.push_back(up(l.attn_norm)); g.fn.push_back(up(l.ffn_norm));
        if (l.conv) { g.taps.push_back(up(l.taps)); g.cst.push_back(gpu::alloc((size_t)n_slots * 2 * d * 4)); ok = ok && g.cst.back(); }
        else {
            g.qn.push_back(up(l.q_norm)); g.kn.push_back(up(l.k_norm));
            g.kv.push_back(gpu::alloc((size_t)n_slots * max_ctx * kvw * (kv_int8_ ? 1 : 4))); ok = ok && g.kv.back();
            if (kv_int8_) { g.kvs.push_back(gpu::alloc((size_t)n_slots * max_ctx * (kvw / 32) * 4)); ok = ok && g.kvs.back(); }
        }
    }
    g.out_norm = up(m.out_norm); g.cosb = up(m.rope.cosb); g.sinb = up(m.rope.sinb);
    std::vector<float> z(2 * (size_t)d, 0.0f);
    g.zero = up(z);
    gpu::warm_small_gemm(4);
    g.slots = n_slots; g.cap = max_ctx;
    g.Tmax = std::min(max_ctx, getenv("FERRO_LM_TMAX") ? atoi(getenv("FERRO_LM_TMAX")) : 1024);
    g.max_ns = std::max(n_slots, 1);
    const size_t R = (size_t)std::max(g.Tmax, g.max_ns);
    g.X = gpu::alloc(R * d * 4); g.X2 = gpu::alloc((size_t)4 * d * 4); g.D = gpu::alloc(R * d * 4); g.C = gpu::alloc(R * std::max(3 * d, rowc) * 4);
    g.G = gpu::alloc(R * 2 * ff * 4); g.Xl = gpu::alloc((size_t)g.max_ns * d * 4); g.Dl = gpu::alloc((size_t)g.max_ns * d * 4);
    g.logits = gpu::alloc((size_t)g.max_ns * m.n_vocab * 4);
    const size_t mi = R * 4;
    g.bpos = gpu::alloc(mi); g.bdst = gpu::alloc(mi); g.bsrc1 = gpu::alloc(mi); g.bsrc2 = gpu::alloc(mi);
    g.bslot = gpu::alloc(mi); g.br1 = gpu::alloc(mi); g.br2 = gpu::alloc(mi); g.bbase = gpu::alloc(mi); g.blen = gpu::alloc(mi);
    for (gpu::Buf b : {g.X, g.X2, g.D, g.C, g.G, g.Xl, g.Dl, g.logits, g.bpos, g.bdst, g.bsrc1, g.bsrc2, g.bslot, g.br1, g.br2, g.bbase, g.blen, g.zero}) ok = ok && b;
    if (!ok || !gpu::ensure_qscratch((int)R, std::max(ff, 2 * d))) return false;
    for (int i = n_slots - 1; i >= 0; --i) g.free_slots.push_back(i);
    in.g.hostX.resize(R * d);
    if (!getenv("FERRO_LM_KEEP_HOST")) {
        // The weights now live on the device (system RAM on an iGPU), so the host copy is pure duplication: ~1.2 GB for the 1.2B model.
        // Matters under tight memory limits (this container is capped at 4 GiB). The embedding table stays: token lookup runs on the host.
        // Without the host planes there is no CPU fallback for this engine instance (set FERRO_LM_KEEP_HOST=1 to keep it).
        auto drop = [](WMat& W) { std::vector<int8_t>().swap(W.wq); std::vector<float>().swap(W.ws); std::vector<float>().swap(W.rs); std::vector<float>().swap(W.wf); };
        for (auto& l : m.L) for (WMat* W : {&l.in_proj, &l.out_proj, &l.wqkv, &l.wo, &l.gateup, &l.down}) if (!W->empty()) drop(*W);
    }
    if (getenv("FERRO_VERBOSE"))
        fprintf(stderr, "[lm] resident on %s: %.0f MB (weights + %d slot(s) x %d ctx KV/conv state)\n", gpu::device_name(), gpu::resident_bytes() / 1e6, n_slots, max_ctx);
    return true;
}

void LmEngine::release(LmSeq& s) {
    if (impl_ && impl_->g.ok && s.gpu_slot >= 0) impl_->g.free_slots.push_back(s.gpu_slot);
    s.gpu_slot = -1;
}

static bool lm_skip(const char* name) {
    static const std::string abl = getenv("FERRO_LM_ABL") ? std::string(",") + getenv("FERRO_LM_ABL") + "," : std::string();
    return !abl.empty() && abl.find(std::string(",") + name + ",") != std::string::npos;
}

static bool lm_dup(const char* name) {            // FERRO_LM_DUP=a,b: launch these ops twice (marginal launch cost experiment; results are garbage)
    static const std::string d = getenv("FERRO_LM_DUP") ? std::string(",") + getenv("FERRO_LM_DUP") + "," : std::string();
    return !d.empty() && d.find(std::string(",") + name + ",") != std::string::npos;
}

// One forward over `T` rows. Prefill (pf): rows are consecutive tokens of one sequence (positions pos0..), decode: one row per sequence.
static bool gpu_forward(LmEngine::Impl& in, const LmModel& m, const std::vector<int>& ids, const std::vector<LmSeq*>& seqs, bool pf, int pos0, float* logits,
                        std::string* err) {
    // Decode rows always use the split-K GEMV and prefill rows always use the tile GEMM, so a row's arithmetic does not
    // depend on how many other rows share the launch (batch- and chunk-invariant; FERRO_ROW_POLICY=0 restores the old mix).
    struct Policy { Policy(int p) { static const char* e = getenv("FERRO_ROW_POLICY"); gpu::set_row_policy(e && atoi(e) == 0 ? 0 : p); } ~Policy() { gpu::set_row_policy(0); } } policy_guard(pf ? 2 : 1);
    auto& g = in.g;
    const bool kvq = !g.kvs.empty();                                       // int8 KV cache
    const int d = m.n_embd, hd = m.head_dim, nq = m.n_head, nkv = m.n_kv, ff = m.n_ff, T = (int)ids.size(), ns = (int)seqs.size();
    const int qsz = nq * hd, ksz = nkv * hd, kvw = 2 * ksz, rowc = qsz + kvw;
    const float scale = 1.0f / std::sqrt((float)hd);
    g.pos.resize(T); g.dst.resize(T); g.src1.resize(T); g.src2.resize(T);
    g.slot.resize(ns); g.r1.resize(ns); g.r2.resize(ns); g.base.resize(ns); g.len.resize(ns);
    if (pf) {
        const int sl = seqs[0]->gpu_slot;
        for (int t = 0; t < T; ++t) {
            g.pos[t] = pos0 + t; g.dst[t] = sl * g.cap + pos0 + t;
            g.src1[t] = t >= 1 ? t - 1 : -1 - (sl * 2 + 1);
            g.src2[t] = t >= 2 ? t - 2 : (t == 1 ? -1 - (sl * 2 + 1) : -1 - (sl * 2 + 0));
        }
        g.slot[0] = sl; g.r1[0] = T - 1; g.r2[0] = T >= 2 ? T - 2 : -1;
    } else {
        for (int b = 0; b < T; ++b) {
            const int sl = seqs[b]->gpu_slot, p = seqs[b]->n_past;
            g.pos[b] = p; g.dst[b] = sl * g.cap + p; g.src1[b] = -1 - (sl * 2 + 1); g.src2[b] = -1 - (sl * 2 + 0);
            g.slot[b] = sl; g.r1[b] = b; g.r2[b] = -1; g.base[b] = sl * g.cap; g.len[b] = p + 1;
        }
    }
    static const bool steptime = getenv("FERRO_LM_STEPTIME") != nullptr;
    const auto tstep0 = std::chrono::steady_clock::now();
    gather_embd(m.tok_embd, ids.data(), T, 1.0f, g.hostX.data());
    bool ok = gpu::write(g.X, g.hostX.data(), (size_t)T * d * 4) && gpu::write(g.bpos, g.pos.data(), (size_t)T * 4) && gpu::write(g.bdst, g.dst.data(), (size_t)T * 4)
              && gpu::write(g.bsrc1, g.src1.data(), (size_t)T * 4) && gpu::write(g.bsrc2, g.src2.data(), (size_t)T * 4)
              && gpu::write(g.bslot, g.slot.data(), (size_t)ns * 4) && gpu::write(g.br1, g.r1.data(), (size_t)ns * 4) && gpu::write(g.br2, g.r2.data(), (size_t)ns * 4);
    if (!pf) ok = ok && gpu::write(g.bbase, g.base.data(), (size_t)ns * 4) && gpu::write(g.blen, g.len.data(), (size_t)ns * 4);
    int ci = 0, ai = 0;
    static const bool fuse = !(getenv("FERRO_LM_FUSE") && atoi(getenv("FERRO_LM_FUSE")) == 0);      // FERRO_LM_FUSE=0: unfused launches (A/B); unset/1: conv+state and qkv+kv fusions (default); 2: + prologue-fused GEMVs (opt-in, break-even on ADL-N)
    static const bool fuse_pro = fuse && getenv("FERRO_LM_FUSE") && atoi(getenv("FERRO_LM_FUSE")) >= 2;
    // decode with <= 4 sequences: rmsnorm / silu*up run inside the consumer GEMV (one launch less each); X ping-pongs between X and X2
    static const int pro_mask = getenv("FERRO_PRO_MASK") ? atoi(getenv("FERRO_PRO_MASK")) : 15;   // sites: 1 layer-opening GEMV, 2 gateup, 4 down(silu), 8 lm_head
    bool fpro = fuse_pro && !pf && T <= 4 && ns == T && gpu::gemv_pro_ok(m.tok_embd.gpu_slot, T, 1, 0);
    for (int il = 0; fpro && il < m.n_layer; ++il) {
        const LmLayer& l = m.L[il];
        fpro = gpu::gemv_pro_ok(l.conv ? l.in_proj.gpu_slot : l.wqkv.gpu_slot, T, 1, 0) && gpu::gemv_pro_ok(l.gateup.gpu_slot, T, 1, 0) && gpu::gemv_pro_ok(l.down.gpu_slot, T, 2, ff);
    }
    const bool p_first = fpro && (pro_mask & 1), p_gate = fpro && (pro_mask & 2), p_down = fpro && (pro_mask & 4), p_head = fpro && (pro_mask & 8);
    if (!pf && getenv("FERRO_VERBOSE")) { static int last = -1; const int sig = (p_first | p_gate << 1 | p_down << 2 | p_head << 3) * 8 + T; if (last != sig) { last = sig; fprintf(stderr, "[lm] decode T=%d: fused-prologue GEMV sites first=%d gateup=%d down=%d head=%d\n", T, p_first, p_gate, p_down, p_head); } }
    gpu::Buf xcur = g.X, xoth = g.X2; bool have_d = false;      // xcur: residual stream; have_d: g.D holds a not-yet-added branch output
    auto first_gemm = [&](int slot, int il, gpu::Buf Y) {        // GEMV that opens a layer, together with the attn/conv input norm
        if (p_first) {
            const bool r = gpu::op_gemv_pro(slot, T, Y, 1, xcur, have_d ? g.D : nullptr, g.an[il], have_d ? xoth : nullptr, 0, m.rms_eps);
            if (r && have_d) std::swap(xcur, xoth);
            have_d = false;
            return r;
        }
        const bool r = gpu::op_resid_norm_q(xcur, have_d ? g.D : nullptr, nullptr, g.an[il], nullptr, false, T, d, m.rms_eps);   // in place
        have_d = false;
        return r && gpu::op_gemm_q(slot, T, Y);
    };
    static const bool prof = getenv("FERRO_LM_PROF") != nullptr;          // per-op timing (syncs after every op)
    static std::map<std::string, std::pair<double, int>> times;
    static struct Dump { ~Dump() { if (!prof) return; double tot = 0; for (auto& kv : times) tot += kv.second.first;
        for (auto& kv : times) fprintf(stderr, "[lm-prof] %-14s %8.1f ms total  %6d calls  %7.3f ms/call  %5.1f%%\n", kv.first.c_str(), kv.second.first * 1e3, kv.second.second, kv.second.first * 1e3 / kv.second.second, 100 * kv.second.first / tot); } } dump_at_exit;
    auto P_impl = [&](const char* name, bool r) {
        if (!prof || !r) return r;
        const auto t0 = std::chrono::steady_clock::now(); gpu::finish();
        auto& e = times[std::string(pf ? "pf:" : "dec:") + name]; e.first += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); e.second++;
        return r;
    };
#define P(name, expr) (lm_skip(name) ? true : (lm_dup(name) ? ((expr) && P_impl(name, (expr))) : P_impl(name, (expr))))   // FERRO_LM_ABL=a,b: skip ops by name (timing ablation; results are garbage)
    // when profiling, each op is timed from enqueue to completion (the previous op is already finished, so this is that op's own latency)
    for (int il = 0; ok && il < m.n_layer; ++il) {
        const LmLayer& l = m.L[il];
        if (l.conv) {
            ok = P("gemm", first_gemm(l.in_proj.gpu_slot, il, g.C));
            if (ok && fuse && !pf) ok = P("conv", gpu::op_lm_conv_dec_q(g.C, g.cst[ci], g.bslot, g.taps[ci], T, d));          // decode: gate + state in one launch
            else if (ok) ok = P("conv", gpu::op_lm_conv_q(g.C, g.bsrc1, g.bsrc2, g.cst[ci], g.taps[ci], T, d))
                              && P("conv_state", gpu::op_lm_conv_state(g.C, g.cst[ci], g.bslot, g.br1, g.br2, ns, d));
            ok = ok && P("gemm", gpu::op_gemm_q(l.out_proj.gpu_slot, T, g.D));
            ++ci;
        } else {
            ok = P("gemm", first_gemm(l.wqkv.gpu_slot, il, g.C));
            if (ok && kvq) ok = P("qkv_post", gpu::op_lm_qkv_post_kvq(g.C, rowc, T, nq, nkv, hd, g.qn[ai], g.kn[ai], g.cosb, g.sinb, g.bpos, hd / 2, m.rms_eps, g.kv[ai], g.kvs[ai], g.bdst, kvw));
            else if (ok && fuse) ok = P("qkv_post", gpu::op_lm_qkv_post_kv(g.C, rowc, T, nq, nkv, hd, g.qn[ai], g.kn[ai], g.cosb, g.sinb, g.bpos, hd / 2, m.rms_eps, g.kv[ai], g.bdst, kvw));
            else if (ok) ok = P("qkv_post", gpu::op_qkv_post(g.C, rowc, T, nq, nkv, hd, g.qn[ai], g.kn[ai], g.cosb, g.sinb, g.bpos, hd / 2, m.rms_eps, false))
                              && P("kv_write", gpu::op_lm_kv_write(g.C, rowc, qsz, g.kv[ai], g.bdst, T, kvw));
            ok = ok
                 && P("attn", kvq ? (pf ? gpu::op_lm_attn_pf_q(g.C, rowc, g.kv[ai], g.kvs[ai], kvw, ksz, seqs[0]->gpu_slot * g.cap, nq, nkv, pos0, scale, T)
                                        : gpu::op_lm_attn_dec_q(g.C, rowc, g.kv[ai], g.kvs[ai], kvw, ksz, g.bbase, g.blen, nq, nkv, scale, T))
                                  : (pf ? gpu::op_lm_attn_pf(g.C, rowc, g.kv[ai], kvw, ksz, seqs[0]->gpu_slot * g.cap, nq, nkv, pos0, scale, T)
                                        : gpu::op_lm_attn_dec(g.C, rowc, g.kv[ai], kvw, ksz, g.bbase, g.blen, nq, nkv, scale, T)))
                 && P("gemm", gpu::op_gemm_q(l.wo.gpu_slot, T, g.D));
            ++ai;
        }
        if (p_gate) { ok = ok && P("gemm", gpu::op_gemv_pro(l.gateup.gpu_slot, T, g.G, 1, xcur, g.D, g.fn[il], xoth, 0, m.rms_eps)); std::swap(xcur, xoth); }
        else ok = ok && P("resid_norm", gpu::op_resid_norm_q(xcur, g.D, nullptr, g.fn[il], nullptr, false, T, d, m.rms_eps)) && P("gemm", gpu::op_gemm_q(l.gateup.gpu_slot, T, g.G));
        if (p_down) ok = ok && P("gemm", gpu::op_gemv_pro(l.down.gpu_slot, T, g.D, 2, g.G, nullptr, nullptr, nullptr, ff, 0.f));
        else ok = ok && P("silu_mul", gpu::op_silu_mul_q(g.G, T, ff)) && P("gemm", gpu::op_gemm_q(l.down.gpu_slot, T, g.D));
        have_d = true;
        if (ok && pf) ok = gpu::finish();                                  // keep every submission short (i915 watchdog)
    }
    // final: X += D on the last row of every sequence, output norm, tied LM head
    for (int s = 0; ok && !p_head && s < ns; ++s) {
        const size_t row = (size_t)(pf ? T - 1 : s) * d * 4;
        ok = gpu::copy(g.Xl, (size_t)s * d * 4, xcur, row, (size_t)d * 4) && gpu::copy(g.Dl, (size_t)s * d * 4, g.D, row, (size_t)d * 4);
    }
    const auto tenq = std::chrono::steady_clock::now();
    if (p_head) ok = ok && P("lm_head", gpu::op_gemv_pro(m.tok_embd.gpu_slot, ns, g.logits, 1, xcur, g.D, g.out_norm, nullptr, 0, m.rms_eps)) && gpu::read(g.logits, logits, (size_t)ns * m.n_vocab * 4);
    else
    ok = ok && gpu::op_resid_norm_q(g.Xl, g.Dl, nullptr, g.out_norm, nullptr, false, ns, d, m.rms_eps)
         && P("lm_head", gpu::op_gemm_q(m.tok_embd.gpu_slot, ns, g.logits)) && gpu::read(g.logits, logits, (size_t)ns * m.n_vocab * 4);
    if (steptime && !pf) {
        const auto tend = std::chrono::steady_clock::now();
        fprintf(stderr, "[lm-step] enqueue %.1f ms, device wait+read %.1f ms\n", std::chrono::duration<double>(tenq - tstep0).count() * 1e3, std::chrono::duration<double>(tend - tenq).count() * 1e3);
    }
    if (!ok && err) *err = "iGPU forward failed";
    return ok;
}

bool LmEngine::on_gpu() const { return impl_ && impl_->g.ok; }

bool LmEngine::init(const std::string& path, int n_threads, int max_ctx, std::string* err, int n_slots, bool kv_int8) {
    (void)n_threads;
    max_ctx_ = max_ctx; n_slots_ = std::max(1, n_slots);
    if (const char* e = getenv("FERRO_KV")) kv_int8 = std::string(e) == "int8";
    kv_int8_ = kv_int8;
    if (!m_.load(path, max_ctx, err)) return false;
    impl_ = std::make_shared<Impl>();
    const char* ge = getenv("FERRO_GPU"); const char* gl = getenv("FERRO_LM_GPU");
    if ((!ge || atoi(ge) != 0) && (!gl || atoi(gl) != 0)) {
        const bool verbose = getenv("FERRO_VERBOSE") != nullptr;
        if (gpu::init(verbose) && gpu_setup(*impl_, m_, max_ctx_, n_slots_, kv_int8_)) impl_->g.ok = true;
        else if (verbose) fprintf(stderr, "[lm] iGPU path unavailable -> CPU\n");
    }
    return true;
}

void LmEngine::reset(LmSeq& s) const {
    s.n_past = 0;
    if (impl_ && impl_->g.ok) {                                       // device-resident state: take a slot and clear its conv state
        auto& g = impl_->g;
        if (s.gpu_slot < 0) {
            if (g.free_slots.empty()) { fprintf(stderr, "[lm] no free sequence slot\n"); return; }
            s.gpu_slot = g.free_slots.back(); g.free_slots.pop_back();
        }
        for (gpu::Buf b : g.cst) gpu::copy(b, (size_t)s.gpu_slot * 2 * m_.n_embd * 4, g.zero, 0, (size_t)2 * m_.n_embd * 4);
        return;
    }
    s.k.assign(m_.n_attn_layers, {}); s.v.assign(m_.n_attn_layers, {});
    s.conv.assign(m_.n_conv_layers, std::vector<float>(2 * (size_t)m_.n_embd, 0.0f));
}

LmPrefix::~LmPrefix() { for (void* b : kv) if (b) gpu::release(b); for (void* b : kvs) if (b) gpu::release(b); for (void* b : cst) if (b) gpu::release(b); }

size_t LmEngine::prefix_bytes(int n_tokens) const {
    const size_t kvw = (size_t)2 * m_.n_kv * m_.head_dim, per_row = kv_int8_ ? kvw + kvw / 32 * 4 : kvw * 4;
    return (size_t)n_tokens * per_row * m_.n_attn_layers + (size_t)2 * m_.n_embd * 4 * m_.n_conv_layers;
}

std::shared_ptr<LmPrefix> LmEngine::snapshot(const LmSeq& s) const {
    if (!impl_ || !impl_->g.ok || s.gpu_slot < 0 || s.n_past <= 0) return nullptr;
    auto& g = impl_->g;
    const size_t kvw = (size_t)2 * m_.n_kv * m_.head_dim, d = (size_t)m_.n_embd;
    auto p = std::make_shared<LmPrefix>();
    p->n = s.n_past;
    const size_t esz = kv_int8_ ? 1 : 4, rows = (size_t)s.n_past * kvw * esz, srows = (size_t)s.n_past * (kvw / 32) * 4;
    for (size_t l = 0; l < g.kv.size(); ++l) {
        gpu::Buf b = gpu::alloc(rows);
        if (!b) return nullptr;
        p->kv.push_back(b);
        if (!gpu::copy(b, 0, g.kv[l], (size_t)s.gpu_slot * g.cap * kvw * esz, rows)) return nullptr;
        if (kv_int8_) {
            gpu::Buf sb = gpu::alloc(srows);
            if (!sb) return nullptr;
            p->kvs.push_back(sb);
            if (!gpu::copy(sb, 0, g.kvs[l], (size_t)s.gpu_slot * g.cap * (kvw / 32) * 4, srows)) return nullptr;
        }
    }
    for (size_t l = 0; l < g.cst.size(); ++l) {
        gpu::Buf b = gpu::alloc(2 * d * 4);
        if (!b) return nullptr;
        p->cst.push_back(b);
        if (!gpu::copy(b, 0, g.cst[l], (size_t)s.gpu_slot * 2 * d * 4, 2 * d * 4)) return nullptr;
    }
    p->bytes = prefix_bytes(s.n_past);
    return p;
}

bool LmEngine::restore(LmSeq& s, const LmPrefix& p) const {
    if (!impl_ || !impl_->g.ok || p.n <= 0 || p.n >= max_ctx_) return false;
    auto& g = impl_->g;
    if (p.kv.size() != g.kv.size() || p.cst.size() != g.cst.size() || p.kvs.size() != g.kvs.size()) return false;
    const size_t kvw = (size_t)2 * m_.n_kv * m_.head_dim, d = (size_t)m_.n_embd;
    if (s.gpu_slot < 0) {
        if (g.free_slots.empty()) return false;
        s.gpu_slot = g.free_slots.back(); g.free_slots.pop_back();
    }
    bool ok = true;
    const size_t esz = kv_int8_ ? 1 : 4;
    for (size_t l = 0; l < g.kv.size(); ++l) {
        ok = ok && gpu::copy(g.kv[l], (size_t)s.gpu_slot * g.cap * kvw * esz, p.kv[l], 0, (size_t)p.n * kvw * esz);
        if (kv_int8_) ok = ok && gpu::copy(g.kvs[l], (size_t)s.gpu_slot * g.cap * (kvw / 32) * 4, p.kvs[l], 0, (size_t)p.n * (kvw / 32) * 4);
    }
    for (size_t l = 0; l < g.cst.size(); ++l) ok = ok && gpu::copy(g.cst[l], (size_t)s.gpu_slot * 2 * d * 4, p.cst[l], 0, 2 * d * 4);
    if (ok) s.n_past = p.n;
    return ok;
}

bool LmEngine::step(const std::vector<LmSeq*>& seqs, const std::vector<const int*>& toks, const std::vector<int>& n, float* logits, std::string* err,
                    bool force_prefill) {
    int T = 0;
    for (size_t i = 0; i < seqs.size(); ++i) {
        if (n[i] <= 0) { if (err) *err = "empty token span"; return false; }
        if (seqs[i]->n_past + n[i] > max_ctx_) { if (err) *err = "context length exceeded (" + std::to_string(max_ctx_) + ")"; return false; }
        T += n[i];
    }
    if (impl_->g.ok) {
        for (LmSeq* s : seqs) if (s->gpu_slot < 0) { if (err) *err = "sequence has no device slot (call reset first)"; return false; }
        if (forward_gpu(seqs, toks, n, logits, err, force_prefill)) { for (size_t i = 0; i < seqs.size(); ++i) seqs[i]->n_past += n[i]; return true; }
        return false;
    }
    for (LmSeq* s : seqs) if ((int)s->conv.size() != m_.n_conv_layers) reset(*s);
    forward_cpu(seqs, toks, n, logits);
    for (size_t i = 0; i < seqs.size(); ++i) seqs[i]->n_past += n[i];
    return true;
}

bool LmEngine::forward_gpu(const std::vector<LmSeq*>& seqs, const std::vector<const int*>& toks, const std::vector<int>& n, float* logits, std::string* err,
                           bool force_prefill) {
    Impl& in = *impl_;
    const int V = m_.n_vocab;
    std::vector<size_t> dec;                                           // single-token sequences are batched into one decode pass
    for (size_t i = 0; i < seqs.size(); ++i) {
        if (n[i] == 1 && !force_prefill) { dec.push_back(i); continue; }
        for (int off = 0; off < n[i]; off += in.g.Tmax) {              // prefill in chunks (each attends over the KV written so far)
            const int c = std::min(in.g.Tmax, n[i] - off);
            std::vector<int> ids(toks[i] + off, toks[i] + off + c);
            const bool last = off + c == n[i];
            std::vector<float> tmp(last ? 0 : (size_t)V);
            LmSeq* sp = seqs[i];
            const int pos0 = sp->n_past + off;
            if (!gpu_forward(in, m_, ids, {sp}, true, pos0, last ? logits + i * (size_t)V : tmp.data(), err)) return false;
        }
    }
    for (size_t b0 = 0; b0 < dec.size(); b0 += (size_t)in.g.max_ns) {
        const size_t b1 = std::min(dec.size(), b0 + (size_t)in.g.max_ns);
        std::vector<int> ids; std::vector<LmSeq*> ss; std::vector<float> tmp((b1 - b0) * (size_t)V);
        for (size_t k = b0; k < b1; ++k) { ids.push_back(toks[dec[k]][0]); ss.push_back(seqs[dec[k]]); }
        if (!gpu_forward(in, m_, ids, ss, false, 0, tmp.data(), err)) return false;
        for (size_t k = b0; k < b1; ++k) memcpy(logits + dec[k] * (size_t)V, tmp.data() + (k - b0) * (size_t)V, (size_t)V * 4);
    }
    return true;
}

// silu(G[:ff]) * G[ff:]
static void silu_mul(const float* G, int T, int ff, float* F) {
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int t = lo; t < hi; ++t) {
            const float* g = G + (size_t)t * 2 * ff; float* f = F + (size_t)t * ff;
            for (int i = 0; i < ff; ++i) f[i] = g[i] / (1.0f + std::exp(-g[i])) * g[ff + i];
        }
    });
}

void LmEngine::forward_cpu(const std::vector<LmSeq*>& seqs, const std::vector<const int*>& toks, const std::vector<int>& n, float* logits) {
    const LmModel& m = m_;
    Impl& in = *impl_;
    const int d = m.n_embd, hd = m.head_dim, nq = m.n_head, nkv = m.n_kv, ff = m.n_ff;
    const int qsz = nq * hd, ksz = nkv * hd, rowc = qsz + 2 * ksz;
    const int ns = (int)seqs.size();
    int T = 0;
    std::vector<int> row0(ns);
    for (int i = 0; i < ns; ++i) { row0[i] = T; T += n[i]; }
    in.ensure(T, d, ff, rowc);
    std::vector<int> ids(T), pos(T);
    for (int i = 0; i < ns; ++i) for (int j = 0; j < n[i]; ++j) { ids[row0[i] + j] = toks[i][j]; pos[row0[i] + j] = seqs[i]->n_past + j; }
    float *X = in.X.data(), *X2 = in.X2.data(), *Hn = in.Hn.data(), *C = in.C.data(), *BX = in.BX.data(), *O = in.O.data();
    float *G = in.G.data(), *F = in.F.data();
    const float scale = 1.0f / std::sqrt((float)hd);
    gather_embd(m.tok_embd, ids.data(), T, 1.0f, X);
    int ci = 0, ai = 0;
    const int nl_lim = getenv("FERRO_LM_NLAYER") ? std::min(m.n_layer, atoi(getenv("FERRO_LM_NLAYER"))) : m.n_layer;   // debug: run only the first N layers
    for (int il = 0; il < nl_lim; ++il) {
        const LmLayer& l = m.L[il];
        rmsnorm_q(X, T, d, l.attn_norm.data(), m.rms_eps, Hn);
        if (l.conv) {
            gemm_q8_fma(l.in_proj, Hn, T, nullptr, nullptr, C, 3 * d);                    // b | c | x
            Pool::get().parallel_for(T, [&](int lo, int hi) {
                for (int t = lo; t < hi; ++t) {
                    const float* b = C + (size_t)t * 3 * d; const float* x = b + 2 * d; float* bx = BX + (size_t)t * d;
                    for (int i = 0; i < d; ++i) bx[i] = b[i] * x[i];
                }
            });
            const float *w0 = l.taps.data(), *w1 = w0 + d, *w2 = w1 + d;
            for (int s = 0; s < ns; ++s) {                                                // causal depthwise conv over the sequence + its state
                float* st = seqs[s]->conv[ci].data();                                     // [0,d) = bx(t-2), [d,2d) = bx(t-1) of the first new token
                const int r0 = row0[s], cnt = n[s];
                Pool::get().parallel_for(cnt, [&](int lo, int hi) {
                    for (int i = lo; i < hi; ++i) {
                        const float* cx = C + (size_t)(r0 + i) * 3 * d + d;
                        const float* p2 = i >= 2 ? BX + (size_t)(r0 + i - 2) * d : (i == 0 ? st : st + d);
                        const float* p1 = i >= 1 ? BX + (size_t)(r0 + i - 1) * d : st + d;
                        const float* p0 = BX + (size_t)(r0 + i) * d;
                        float* o = O + (size_t)(r0 + i) * d;
                        for (int c = 0; c < d; ++c) o[c] = (w0[c] * p2[c] + w1[c] * p1[c] + w2[c] * p0[c]) * cx[c];
                    }
                });
                // new state = the last two bx rows (state is read above, so update after the conv)
                std::vector<float> ns2(2 * (size_t)d);
                if (cnt >= 2) { memcpy(ns2.data(), BX + (size_t)(r0 + cnt - 2) * d, d * 4); memcpy(ns2.data() + d, BX + (size_t)(r0 + cnt - 1) * d, d * 4); }
                else { memcpy(ns2.data(), st + d, d * 4); memcpy(ns2.data() + d, BX + (size_t)r0 * d, d * 4); }
                memcpy(st, ns2.data(), 2 * (size_t)d * 4);
            }
            ++ci;
            gemm_q8_fma(l.out_proj, O, T, nullptr, X, X2, d); std::swap(X, X2);
        } else {
            gemm_q8_fma(l.wqkv, Hn, T, nullptr, nullptr, C, rowc);
            for (int h = 0; h < nq; ++h) rmsnorm_q(C + h * hd, T, hd, l.q_norm.data(), m.rms_eps, C + h * hd, rowc, rowc);
            for (int h = 0; h < nkv; ++h) rmsnorm_q(C + qsz + h * hd, T, hd, l.k_norm.data(), m.rms_eps, C + qsz + h * hd, rowc, rowc);
            rope_apply(C, T, nq, hd, m.rope, pos.data(), rowc);
            rope_apply(C + qsz, T, nkv, hd, m.rope, pos.data(), rowc);
            for (int s = 0; s < ns; ++s) {                                                // append K/V of the new tokens to the cache
                std::vector<float>& K = seqs[s]->k[ai]; std::vector<float>& V = seqs[s]->v[ai];
                const size_t old = K.size() / ksz;
                K.resize((old + n[s]) * (size_t)ksz); V.resize((old + n[s]) * (size_t)ksz);
                for (int j = 0; j < n[s]; ++j) {
                    memcpy(K.data() + (old + j) * ksz, C + (size_t)(row0[s] + j) * rowc + qsz, ksz * 4);
                    memcpy(V.data() + (old + j) * ksz, C + (size_t)(row0[s] + j) * rowc + qsz + ksz, ksz * 4);
                }
            }
            const int grp = nq / nkv;
            Pool::get().parallel_for(T * nq, [&](int a, int b) {
                std::vector<float> sc;
                for (int idx = a; idx < b; ++idx) {
                    const int t = idx / nq, h = idx % nq, kh = h / grp;
                    int s = 0; while (s + 1 < ns && row0[s + 1] <= t) ++s;
                    const int kn = pos[t] + 1;                                            // causal: keys 0 .. own position
                    const float* K = seqs[s]->k[ai].data(); const float* V = seqs[s]->v[ai].data();
                    sc.resize((size_t)kn);
                    const float* q = C + (size_t)t * rowc + h * hd;
                    float mx = -1e30f;
                    for (int j = 0; j < kn; ++j) {
                        const float* kr = K + (size_t)j * ksz + kh * hd; float acc = 0;
                        for (int e = 0; e < hd; ++e) acc += q[e] * kr[e];
                        sc[j] = acc * scale; mx = std::max(mx, sc[j]);
                    }
                    float sum = 0;
                    for (int j = 0; j < kn; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                    const float inv = 1.0f / sum;
                    float* o = O + (size_t)t * d + h * hd;
                    for (int e = 0; e < hd; ++e) o[e] = 0;
                    for (int j = 0; j < kn; ++j) { const float p = sc[j] * inv; const float* vr = V + (size_t)j * ksz + kh * hd; for (int e = 0; e < hd; ++e) o[e] += p * vr[e]; }
                }
            });
            ++ai;
            gemm_q8_fma(l.wo, O, T, nullptr, X, X2, d); std::swap(X, X2);
        }
        rmsnorm_q(X, T, d, l.ffn_norm.data(), m.rms_eps, Hn);
        gemm_q8_fma(l.gateup, Hn, T, nullptr, nullptr, G, 2 * ff);
        silu_mul(G, T, ff, F);
        gemm_q8_fma(l.down, F, T, nullptr, X, X2, d); std::swap(X, X2);
    }
    if (const char* dp = getenv("FERRO_LM_DUMP_HIDDEN")) {          // debug: post-output-norm hidden state of every row (what llama.cpp returns as embeddings)
        std::vector<float> hn((size_t)T * d);
        rmsnorm_q(X, T, d, m.out_norm.data(), m.rms_eps, hn.data());
        FILE* f = fopen(dp, "wb"); if (f) { fwrite(hn.data(), 4, hn.size(), f); fclose(f); }
    }
    // logits of the last token of every sequence
    in.last.resize((size_t)ns * d); in.lastn.resize((size_t)ns * d);
    for (int s = 0; s < ns; ++s) memcpy(in.last.data() + (size_t)s * d, X + (size_t)(row0[s] + n[s] - 1) * d, d * 4);
    rmsnorm_q(in.last.data(), ns, d, m.out_norm.data(), m.rms_eps, in.lastn.data());
    gemm_q8_fma(m.tok_embd, in.lastn.data(), ns, nullptr, nullptr, logits, m.n_vocab);
}

}  // namespace ferro
