// model_lfm3_d1.cpp — LFM3-d1 (d1-2B) stateless decision forward, CPU path + iGPU path.
// The block arithmetic is the LFM3 CPU reference (see forward_cpu_lfm3 in model_lfm2_lm.cpp), with the two
// decision differences: the conv state is zero at every layer (the prompt is the whole world) and the global
// attention layers read ALL prompt rows (the SWA layers keep causal + window), per llama.cpp lfm3.cpp graph<true>.
// The iGPU path reuses the LFM3 prefill kernel set: tile GEMMs on the shared int8 scratch, op_lm_conv_q7 from a
// permanently zero conv state (no state update — the prompt is the whole world), op_lm_qkv_post_kvq128 +
// op_lm_attn_pf_q128 with the whole prompt written as KV rows (dst[t] = t); global layers use the bidirectional
// kernel mode (every query row sees keys [0, T-1]), SWA layers keep causal + window.
#include "model_lfm3_d1.h"
#include "gpu_cl.h"
#include "kernels.h"
#include "threadpool.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ferro {

static double now_ms_d1() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// FERRO_D1_DENSE: bit 1 = dense attention on the global (bidirectional) layers, bit 2 = on the SWA layers; default 3.
// 0 = the per-tile online-softmax kernel (lm_attn_pf_q128, int8 KV) everywhere — one prompt per forward then.
static int d1_dense_mode(const LmModel& m) {
    if (m.head_dim % 64 || m.n_head * m.head_dim != m.n_embd) return 0;
    static const int e = getenv("FERRO_D1_DENSE") ? atoi(getenv("FERRO_D1_DENSE")) : 3;
    return e;
}

bool is_lfm3_d1(const GgufFile& g) {
    return g.arch == "lfm3" && g.get_str("lfm3.decision.type") == "lfm3-d1";
}

// ------------------------------------------------------------------ iGPU state
static constexpr int kD1MaxSeg = 64;                                 // prompts per packed forward (logits rows read back)
struct Lfm3D1Engine::Impl {
    bool ok = false;
    int Tmax = 0;
    std::vector<gpu::Buf> an, fn, pan, pfn, taps, qn, kn, kv, kvs;   // per layer / per conv / per attention layer
    gpu::Buf out_norm = nullptr, cosb = nullptr, sinb = nullptr, cosb2 = nullptr, sinb2 = nullptr;
    gpu::Buf X = nullptr, D = nullptr, C = nullptr, G = nullptr, GA = nullptr, Xl = nullptr, Dl = nullptr, logits = nullptr;
    gpu::Buf bpos = nullptr, bdst = nullptr, bsrc6 = nullptr, cst0 = nullptr;
    gpu::Buf S = nullptr; size_t capS = 0;                           // dense attention score scratch (floats); queries are tiled to fit
    std::vector<float> hostX;                                        // staging: must outlive the enqueued writes
    std::vector<int> pos, dst, src6;
    std::vector<int> seg0, segn;                                     // packed segments of the current forward
};

static bool d1_gpu_setup(Lfm3D1Engine::Impl& g, LmModel& m, int tmax, bool verbose) {
    const int d = m.n_embd, hd = m.head_dim, nq = m.n_head, nkv = m.n_kv, ff = m.n_ff;
    const int qsz = nq * hd, ksz = nkv * hd, kvw = 2 * ksz, rowc = qsz + kvw;
    if (hd != 128 || !m.sandwich || !m.has_gate || m.n_conv_state != 6 || d % 32 || d / 4 > 512 || ff % 32 || ff / 8 > 1024 || nq % nkv) return false;
    if (m.swa_window > 0 && (d % 32 || d / 4 > 512)) return false;
    bool ok = true;
    auto slot = [&](WMat& W) { W.gpu_slot = gpu::upload_q8(W); ok = ok && W.gpu_slot >= 0; };
    for (auto& l : m.L) { if (l.conv) { slot(l.in_proj); slot(l.out_proj); } else { slot(l.wqkv); slot(l.wo); slot(l.attn_gate); } slot(l.gateup); slot(l.down); }
    slot(m.tok_embd);
    if (!ok) return false;
    auto up = [&](const std::vector<float>& v) { gpu::Buf b = gpu::upload(v.data(), v.size() * 4); ok = ok && b; return b; };
    for (auto& l : m.L) {
        g.an.push_back(up(l.attn_norm)); g.fn.push_back(up(l.ffn_norm));
        g.pan.push_back(up(l.post_attn_norm)); g.pfn.push_back(up(l.post_ffn_norm));
        if (l.conv) { g.taps.push_back(up(l.taps)); }
        else {
            g.qn.push_back(up(l.q_norm)); g.kn.push_back(up(l.k_norm));
            g.kv.push_back(gpu::alloc((size_t)tmax * kvw)); ok = ok && g.kv.back();
            g.kvs.push_back(gpu::alloc((size_t)tmax * (kvw / 32) * 4)); ok = ok && g.kvs.back();
        }
    }
    g.out_norm = up(m.out_norm); g.cosb = up(m.rope.cosb); g.sinb = up(m.rope.sinb);
    g.cosb2 = up(m.rope_swa.cosb); g.sinb2 = up(m.rope_swa.sinb);
    std::vector<float> z((size_t)m.n_conv_state * d, 0.0f);          // the decision conv state is zero forever
    g.cst0 = up(z);
    g.Tmax = tmax;
    const size_t R = (size_t)tmax;
    g.X = gpu::alloc(R * d * 4); g.D = gpu::alloc(R * d * 4); g.C = gpu::alloc(R * (size_t)std::max(3 * d, rowc) * 4);
    g.G = gpu::alloc(R * 2 * ff * 4); g.GA = gpu::alloc(R * 64 * 4);
    g.Xl = gpu::alloc((size_t)kD1MaxSeg * d * 4); g.Dl = gpu::alloc((size_t)kD1MaxSeg * d * 4);
    g.logits = gpu::alloc((size_t)kD1MaxSeg * m.n_vocab * 4);
    g.bpos = gpu::alloc(R * 4); g.bdst = gpu::alloc(R * 4); g.bsrc6 = gpu::alloc(6 * R * 4);
    // dense attention scratch: nq * rows * T floats; 16M floats (64 MB) keeps a whole 1k-token prompt in one query tile
    g.capS = std::max<size_t>((size_t)16 << 20, (size_t)nq * 64 * R); g.S = gpu::alloc(g.capS * 4);
    for (gpu::Buf b : {g.X, g.D, g.C, g.G, g.GA, g.Xl, g.Dl, g.logits, g.bpos, g.bdst, g.bsrc6, g.cst0, g.S}) ok = ok && b;
    if (!ok || !gpu::ensure_qscratch((int)R, std::max(ff, 2 * d))) return false;
    g.hostX.resize(R * d);
    g.ok = true;
    if (verbose) fprintf(stderr, "[d1] resident on %s: %.0f MB, Tmax %d\n", gpu::device_name(), gpu::resident_bytes() / 1e6, tmax);
    return true;
}

bool Lfm3D1Engine::init(const std::string& path, int n_threads, std::string* err) {
    n_threads_ = n_threads;
    Pool::get().init(n_threads);                                      // the parallel_for sections below run on this pool
    max_ctx_ = 16384;                                       // decision prompts are state + one question; the rope tables cover this
    if (!m_.load(path, max_ctx_, err, /*allow_decision=*/true)) return false;
    if (!is_lfm3_d1(m_.gguf)) { *err = "not an LFM3 decision model (expected lfm3.decision.type == lfm3-d1)"; return false; }
    if (!m_.sandwich || m_.n_conv_state != 6) { *err = "unexpected LFM3 layout (sandwich norms / 7-tap conv expected)"; return false; }
    const char* ge = getenv("FERRO_GPU"); const char* gd = getenv("FERRO_D1_GPU");
    if ((!ge || atoi(ge) != 0) && (!gd || atoi(gd) != 0)) {
        const bool verbose = getenv("FERRO_VERBOSE") != nullptr;
        if (gpu::init(verbose)) {
            int tmax = std::min(max_ctx_, getenv("FERRO_D1_TMAX") ? atoi(getenv("FERRO_D1_TMAX")) : 4096);
            impl_ = std::make_unique<Impl>();
            if (!d1_gpu_setup(*impl_, m_, tmax, verbose)) impl_.reset();
            if (!impl_ && verbose) fprintf(stderr, "[d1] iGPU decision path unavailable -> CPU\n");
        } else if (verbose) fprintf(stderr, "[d1] iGPU unavailable -> CPU decision forward\n");
    }
    return true;
}

// One packed stateless forward on the iGPU over items[i0, i1): each prompt is a segment of rows [seg0, seg0 + segn)
// with positions 0..segn-1; the KV rows are the packed rows (dst[t] = t). Conv refs before a segment start hit the
// all-zero cst0 rows (zero state at every segment), attention runs per segment (dense: global bidirectional, SWA
// causal + window), everything else runs over all packed rows at once. Logits are read at each segment's last row.
bool Lfm3D1Engine::forward_gpu(const std::vector<D1Item>& items, size_t i0, size_t i1, float* logits) {
    auto& g = *impl_;
    auto& m = m_;
    const int d = m.n_embd, nq = m.n_head, nkv = m.n_kv, ff = m.n_ff;
    const int qsz = nq * m.head_dim, ksz = nkv * m.head_dim, kvw = 2 * ksz, rowc = qsz + kvw;
    const int B = (int)(i1 - i0);
    if (B <= 0 || B > kD1MaxSeg) return false;
    g.seg0.assign(B, 0); g.segn.assign(B, 0);
    int T = 0;
    for (int b = 0; b < B; ++b) { g.seg0[b] = T; g.segn[b] = (int)items[i0 + b].toks->size(); T += g.segn[b]; }
    if (T <= 0 || T > g.Tmax) return false;
    struct Policy { Policy(int p) { static const char* e = getenv("FERRO_ROW_POLICY"); gpu::set_row_policy(e && atoi(e) == 0 ? 0 : p); } ~Policy() { gpu::set_row_policy(0); } } policy_guard(2);
    g.pos.resize((size_t)T); g.dst.resize((size_t)T); g.src6.resize((size_t)T * 6);   // staging (must outlive the enqueued writes)
    for (int b = 0; b < B; ++b) {
        const D1Item& it = items[i0 + b];
        const int r0 = g.seg0[b], n = g.segn[b];
        float* hx = g.hostX.data() + (size_t)r0 * d;
        gather_embd(m.tok_embd, it.toks->data(), n, 1.0f, hx);
        if (it.media_rows)                                                       // image rows overwrite the placeholder ids
            for (size_t i = 0; i < it.media_pos.size(); ++i)
                memcpy(hx + (size_t)it.media_pos[i] * d, it.media_rows + i * d, d * 4);
        for (int tl = 0; tl < n; ++tl) {
            const int t = r0 + tl;
            g.pos[t] = tl; g.dst[t] = t;
            for (int k = 0; k < 6; ++k)                                         // zero state: refs before the segment hit cst0
                g.src6[(size_t)t * 6 + k] = tl - 6 + k >= 0 ? t - 6 + k : -1 - (tl + k);
        }
    }
    bool ok = gpu::write(g.X, g.hostX.data(), (size_t)T * d * 4) && gpu::write(g.bpos, g.pos.data(), (size_t)T * 4)
              && gpu::write(g.bdst, g.dst.data(), (size_t)T * 4) && gpu::write(g.bsrc6, g.src6.data(), (size_t)T * 6 * 4);
    gpu::Buf xcur = g.X; bool have_d = false;
    const float scale = 1.0f / std::sqrt((float)m.head_dim);
    static const bool ldbg = getenv("FERRO_D1_LDBG") != nullptr;
    static const auto ldump = [](int lay, const char* tag, gpu::Buf src, int rows, int cols) {
        std::vector<float> v((size_t)rows * cols);
        if (!gpu::read(src, v.data(), (size_t)rows * cols * 4)) return;
        char n[64]; snprintf(n, sizeof n, "/tmp/d1_gpu_%d_%s.bin", lay, tag);
        FILE* f = fopen(n, "wb"); if (f) { fwrite(v.data(), 4, (size_t)rows * cols, f); fclose(f); }
    };
    // FERRO_D1_PROF=1: finish() after every op and accumulate wall time per op class (profiling only — the
    // syncs themselves cost time, so the per-class sum is a little above the unprofiled forward).
    static const bool prof = getenv("FERRO_D1_PROF") != nullptr;
    // FERRO_D1_DENSE=0: the per-tile online-softmax kernel (int8 KV) instead of the dense GEMM-tiled attention
    // bit 1 = global (bidirectional) layers, bit 2 = SWA (causal + window) layers; default 3
    const int dmode = d1_dense_mode(m);
    if (B > 1 && dmode != 3) return false;                                       // the per-tile kernel has no segment support
    static double pacc[16]; static const char* pname[16] = {"norm+quant", "gemm conv_in", "conv7", "gemm conv_out", "gemm wqkv",
        "gemm gate", "qkv_post", "attn global", "attn swa", "gemm wo", "rms_post", "gemm gateup", "silu_mul", "gemm down", "finish", "head"};
    static int pcalls = 0;
    auto P = [&](int k, bool r) { if (prof && r) { const double t0 = now_ms_d1(); r = gpu::finish(); pacc[k] += now_ms_d1() - t0; } return r; };
    auto T0 = [&]() { if (prof) gpu::finish(); return prof ? now_ms_d1() : 0.0; };
    int ci = 0, ai = 0;
    for (int il = 0; ok && il < m.n_layer; ++il) {
        const LmLayer& l = m.L[il];
        double t0 = T0();
        ok = ok && gpu::op_resid_norm_q(xcur, have_d ? g.D : nullptr, nullptr, g.an[il], nullptr, false, T, d, m.rms_eps);
        auto acc = [&](int k) { if (prof && ok) { ok = gpu::finish(); const double t1 = now_ms_d1(); pacc[k] += t1 - t0; t0 = t1; } };
        acc(0);
        have_d = false;
        if (l.conv) {
            ok = ok && gpu::op_gemm_q(l.in_proj.gpu_slot, T, g.C); acc(1);
            ok = ok && gpu::op_lm_conv_q7(g.C, g.bsrc6, g.cst0, g.taps[ci], T, d); acc(2);
            ok = ok && gpu::op_gemm_q(l.out_proj.gpu_slot, T, g.D); acc(3);
            ++ci;
        } else {
            ok = ok && gpu::op_gemm_q(l.wqkv.gpu_slot, T, g.C); acc(4);
            ok = ok && gpu::op_gemm_q(l.attn_gate.gpu_slot, T, g.GA); acc(5);   // per-head gate logits: same quantized layer input as wqkv
            ok = ok && gpu::op_lm_qkv_post_kvq128(g.C, rowc, T, nq, nkv, g.qn[ai], g.kn[ai], l.swa ? g.cosb2 : g.cosb, l.swa ? g.sinb2 : g.sinb,
                                               g.bpos, m.rms_eps, g.kv[ai], g.kvs[ai], g.bdst, kvw); acc(6);
            const bool dense = dmode & (l.swa ? 2 : 1);
            if (dense) {                                                               // GEMM-tiled QK^T / stats / PV on the f32 rows of C, per segment
                for (int b = 0; ok && b < B; ++b)
                    ok = gpu::op_dense_attention_h(g.C, rowc, 0, qsz, qsz + ksz, g.segn[b], nq, nkv, m.head_dim, scale, g.S, g.D, qsz,
                                                   l.swa ? m.swa_window : -1, g.capS, l.swa ? 1 : 0, g.seg0[b]);
                ok = ok && gpu::op_gate_quant_h128(g.D, g.GA, 64, T, qsz);
            } else
                ok = ok && gpu::op_lm_attn_pf_q128(g.C, rowc, g.kv[ai], g.kvs[ai], kvw, ksz, 0, nq, nkv, 0, scale, g.GA, 64,
                                                   l.swa ? m.swa_window : 0, T, l.swa ? 0 : 1);
            acc(l.swa ? 8 : 7);
            ok = ok && gpu::op_gemm_q(l.wo.gpu_slot, T, g.D); acc(9);
            ++ai;
        }
        ok = ok && gpu::op_rms_inplace(g.D, g.pan[il], T, d, m.rms_eps); acc(10);       // sandwich: norm the block output before the residual add
        if (ldbg && ok) {                                                            // debug: attn block output = xcur + D
            ok = ok && gpu::op_add_scale(xcur, g.D, g.G, T * d, 1.0f) && gpu::finish();
            ldump(il, "attn", g.G, T, d);
        }
        ok = ok && gpu::op_resid_norm_q(xcur, g.D, nullptr, g.fn[il], nullptr, false, T, d, m.rms_eps); acc(0);
        ok = ok && gpu::op_gemm_q(l.gateup.gpu_slot, T, g.G); acc(11);
        ok = ok && gpu::op_silu_mul_q(g.G, T, ff); acc(12);
        ok = ok && gpu::op_gemm_q(l.down.gpu_slot, T, g.D); acc(13);
        ok = ok && gpu::op_rms_inplace(g.D, g.pfn[il], T, d, m.rms_eps); acc(10);      // sandwich: norm the FFN output before the residual add
        have_d = true;
        if (ldbg && ok) {                                                            // debug: block output = xcur + D, dumped for the CPU diff
            ok = ok && gpu::op_add_scale(xcur, g.D, g.G, T * d, 1.0f) && gpu::finish();
            ldump(il, "ffn", g.G, T, d);
        }
        if (ok) ok = gpu::finish();                                                  // keep every submission short (i915 watchdog)
        acc(14);
    }
    double th = T0();
    for (int b = 0; ok && b < B; ++b) {
        const size_t last = (size_t)(g.seg0[b] + g.segn[b] - 1) * d * 4;
        ok = gpu::copy(g.Xl, (size_t)b * d * 4, xcur, last, (size_t)d * 4) && gpu::copy(g.Dl, (size_t)b * d * 4, g.D, last, (size_t)d * 4);
    }
    ok = ok && gpu::op_resid_norm_q(g.Xl, g.Dl, nullptr, g.out_norm, nullptr, false, B, d, m.rms_eps)
         && gpu::op_gemm_q(m.tok_embd.gpu_slot, B, g.logits)
         && gpu::read(g.logits, logits, (size_t)B * m.n_vocab * 4);
    if (ok) ok = gpu::finish();
    if (prof) {
        pacc[15] += now_ms_d1() - th;
        if (++pcalls % 4 == 0 || T > 1000) {
            double tot = 0; for (double v : pacc) tot += v;
            fprintf(stderr, "[d1-prof] %d forwards, last T=%d, total %.1f ms\n", pcalls, T, tot);
            for (int k = 0; k < 16; ++k) if (pacc[k] > 0) fprintf(stderr, "[d1-prof]   %-14s %8.1f ms  %5.1f%%\n", pname[k], pacc[k], 100 * pacc[k] / tot);
        }
    }
    return ok;
}

void Lfm3D1Engine::forward(const std::vector<int>& ids, float* last_logits,
                           const float* media_rows, const std::vector<int>& media_pos) {
    const LmModel& m = m_;
    const int d = m.n_embd, hd = m.head_dim, nq = m.n_head, nkv = m.n_kv, ff = m.n_ff, Lc = m.n_conv_state + 1;
    const int qsz = nq * hd, ksz = nkv * hd, rowc = qsz + 2 * ksz;
    const int T = (int)ids.size();
    const int cw = std::max(rowc, 3 * d);                                            // C serves both wqkv (rowc cols) and conv in_proj (3*d cols)
    std::vector<float> X((size_t)T * d), X2((size_t)T * d), Hn((size_t)T * d), C((size_t)T * cw), BX((size_t)T * d), O((size_t)T * d), G((size_t)T * 2 * ff),
                       F((size_t)T * std::max(2 * ff, 64)), K((size_t)T * ksz), V((size_t)T * ksz),
                       lastn(d), zero((size_t)(Lc - 1) * d, 0.0f);
    std::vector<int> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = t;
    const float scale = 1.0f / std::sqrt((float)hd);
    const int SW = m.swa_window;

    gather_embd(m.tok_embd, ids.data(), T, 1.0f, X.data());
    if (media_rows)                                                                  // image rows overwrite the placeholder ids
        for (size_t i = 0; i < media_pos.size(); ++i)
            memcpy(X.data() + (size_t)media_pos[i] * d, media_rows + i * d, d * 4);
    static const bool ldbg = getenv("FERRO_D1_LDBG") != nullptr;
    auto ldump = [&](int lay, const char* tag, const std::vector<float>& v) {
        if (!ldbg) return;
        char n[64]; snprintf(n, sizeof n, "/tmp/d1_cpu_%d_%s.bin", lay, tag);
        FILE* f = fopen(n, "wb"); if (f) { fwrite(v.data(), 4, (size_t)T * d, f); fclose(f); }
    };

    for (int il = 0; il < m.n_layer; ++il) {
        const LmLayer& l = m.L[il];
        rmsnorm_q(X.data(), T, d, l.attn_norm.data(), m.rms_eps, Hn.data());
        if (l.conv) {
            gemm_q8_fma(l.in_proj, Hn.data(), T, nullptr, nullptr, C.data(), 3 * d);                    // b | c | x
            Pool::get().parallel_for(T, [&](int lo, int hi) {
                for (int t = lo; t < hi; ++t) {
                    const float* b = C.data() + (size_t)t * 3 * d; const float* x = b + 2 * d; float* bx = BX.data() + (size_t)t * d;
                    for (int i = 0; i < d; ++i) bx[i] = b[i] * x[i];
                }
            });
            const float* w = l.taps.data();
            Pool::get().parallel_for(T, [&](int lo, int hi) {                                          // causal from the zero state
                for (int i = lo; i < hi; ++i) {
                    const float* cx = C.data() + (size_t)i * 3 * d + d;
                    float* o = O.data() + (size_t)i * d;
                    for (int c = 0; c < d; ++c) {
                        float acc = 0;
                        for (int k = 0; k < Lc; ++k) {                                                // tap k multiplies token i-(Lc-1)+k
                            const int off = i - (Lc - 1) + k;
                            const float* p = off >= 0 ? BX.data() + (size_t)off * d : zero.data();                        // any zero row (they are all zero)
                            acc += w[(size_t)k * d + c] * p[c];
                        }
                        o[c] = acc * cx[c];
                    }
                }
            });
            gemm_q8_fma(l.out_proj, O.data(), T, nullptr, nullptr, X2.data(), d);
            rmsnorm_q(X2.data(), T, d, l.post_attn_norm.data(), m.rms_eps, X2.data());                 // sandwich: norm before the add
            add_rows(X.data(), X2.data(), T * d, X2.data()); std::swap(X, X2);
            ldump(il, "attn", X);
        } else {
            gemm_q8_fma(l.wqkv, Hn.data(), T, nullptr, nullptr, C.data(), rowc);
            for (int h = 0; h < nq; ++h) rmsnorm_q(C.data() + h * hd, T, hd, l.q_norm.data(), m.rms_eps, C.data() + h * hd, rowc, rowc);
            for (int h = 0; h < nkv; ++h) rmsnorm_q(C.data() + qsz + h * hd, T, hd, l.k_norm.data(), m.rms_eps, C.data() + qsz + h * hd, rowc, rowc);
            const RopeTable& rt = l.swa ? m.rope_swa : m.rope;
            rope_apply(C.data(), T, nq, hd, rt, pos.data(), rowc);
            rope_apply(C.data() + qsz, T, nkv, hd, rt, pos.data(), rowc);
            for (int t = 0; t < T; ++t) {                                                              // K/V of the prompt rows
                memcpy(K.data() + (size_t)t * ksz, C.data() + (size_t)t * rowc + qsz, ksz * 4);
                memcpy(V.data() + (size_t)t * ksz, C.data() + (size_t)t * rowc + qsz + ksz, ksz * 4);
            }
            const int grp = nq / nkv;
            Pool::get().parallel_for(T * nq, [&](int a, int b) {
                std::vector<float> sc;
                for (int idx = a; idx < b; ++idx) {
                    const int t = idx / nq, h = idx % nq, kh = h / grp;
                    const int kst = l.swa ? std::max(0, t - SW + 1) : 0;                               // global: bidirectional over the whole prompt
                    const int ken = l.swa ? t : T - 1;                                                 // SWA stays causal
                    const int kn = ken - kst + 1;
                    sc.resize(kn);
                    const float* q = C.data() + (size_t)t * rowc + h * hd;
                    float mx = -1e30f;
                    for (int j = 0; j < kn; ++j) {
                        const float* kr = K.data() + (size_t)(kst + j) * ksz + kh * hd; float acc = 0;
                        for (int e = 0; e < hd; ++e) acc += q[e] * kr[e];
                        sc[j] = acc * scale; mx = std::max(mx, sc[j]);
                    }
                    float sum = 0;
                    for (int j = 0; j < kn; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                    const float inv = 1.0f / sum;
                    float* o = O.data() + (size_t)t * d + h * hd;
                    for (int e = 0; e < hd; ++e) o[e] = 0;
                    for (int j = 0; j < kn; ++j) { const float p = sc[j] * inv; const float* vr = V.data() + (size_t)(kst + j) * ksz + kh * hd; for (int e = 0; e < hd; ++e) o[e] += p * vr[e]; }
                }
            });
            if (m.has_gate) {                                                                           // per-head sigmoid gate on the attention output
                gemm_q8_fma(l.attn_gate, Hn.data(), T, nullptr, nullptr, F.data(), l.attn_gate.rows);
                Pool::get().parallel_for(T, [&](int lo, int hi) {
                    for (int t = lo; t < hi; ++t) {
                        const float* gt = F.data() + (size_t)t * l.attn_gate.rows;
                        float* o = O.data() + (size_t)t * d;
                        for (int h = 0; h < nq; ++h) {
                            const float g = 1.0f / (1.0f + std::exp(-gt[h]));
                            float* oh = o + (size_t)h * hd;
                            for (int e = 0; e < hd; ++e) oh[e] *= g;
                        }
                    }
                });
            }
            gemm_q8_fma(l.wo, O.data(), T, nullptr, nullptr, X2.data(), d);
            rmsnorm_q(X2.data(), T, d, l.post_attn_norm.data(), m.rms_eps, X2.data());
            add_rows(X.data(), X2.data(), T * d, X2.data()); std::swap(X, X2);
            ldump(il, "attn", X);
        }
        rmsnorm_q(X.data(), T, d, l.ffn_norm.data(), m.rms_eps, Hn.data());
        gemm_q8_fma(l.gateup, Hn.data(), T, nullptr, nullptr, G.data(), 2 * ff);
        Pool::get().parallel_for(T, [&](int lo, int hi) {                                               // silu(gate) * up
            for (int t = lo; t < hi; ++t) {
                const float* g = G.data() + (size_t)t * 2 * ff; float* u = G.data() + (size_t)t * 2 * ff + ff; float* f = F.data() + (size_t)t * ff;
                for (int i = 0; i < ff; ++i) f[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
            }
        });
        gemm_q8_fma(l.down, F.data(), T, nullptr, nullptr, X2.data(), d);
        rmsnorm_q(X2.data(), T, d, l.post_ffn_norm.data(), m.rms_eps, X2.data());
        add_rows(X.data(), X2.data(), T * d, X2.data()); std::swap(X, X2);
        ldump(il, "ffn", X);
    }
    rmsnorm_q(X.data() + (size_t)(T - 1) * d, 1, d, m.out_norm.data(), m.rms_eps, lastn.data());
    gemm_q8_fma(m.tok_embd, lastn.data(), 1, nullptr, nullptr, last_logits, m.n_vocab);
}

bool Lfm3D1Engine::on_gpu() const { return impl_ && impl_->ok; }

int Lfm3D1Engine::gpu_tmax() const { return impl_ ? impl_->Tmax : 0; }

bool Lfm3D1Engine::score(const std::vector<int>& toks, const std::vector<std::vector<int>>& labels,
                         std::vector<float>& scores, std::string* err,
                         const float* media_rows, const std::vector<int>& media_pos) {
    std::vector<D1Item> items(1);
    items[0].toks = &toks; items[0].labels = &labels; items[0].media_rows = media_rows; items[0].media_pos = media_pos;
    std::vector<std::vector<float>> sc;
    if (!score_batch(items, sc, err)) return false;
    scores = std::move(sc[0]);
    return true;
}

bool Lfm3D1Engine::score_batch(const std::vector<D1Item>& items, std::vector<std::vector<float>>& scores, std::string* err) {
    for (const D1Item& it : items) {
        if (!it.toks || !it.labels || it.toks->empty()) { if (err) *err = "empty prompt"; return false; }
        if ((int)it.toks->size() > max_ctx_ - 8) { if (err) *err = "prompt longer than the decision context (" + std::to_string(max_ctx_ - 8) + " tokens)"; return false; }
        if (!it.media_pos.empty() && !it.media_rows) { if (err) *err = "media positions without image rows"; return false; }
        for (int p : it.media_pos) if (p < 0 || p >= (int)it.toks->size()) { if (err) *err = "media row position outside the prompt"; return false; }
    }
    const size_t V = (size_t)m_.n_vocab;
    std::vector<float> lg(items.size() * V);
    std::vector<char> done(items.size(), 0);
    if (on_gpu()) {                                                     // greedy packing: consecutive prompts while they fit Tmax rows
        const int maxb = d1_dense_mode(m_) == 3 ? kD1MaxSeg : 1;
        static const int env_b = getenv("FERRO_D1_PACK") ? std::max(1, atoi(getenv("FERRO_D1_PACK"))) : kD1MaxSeg;
        const int lim = std::min(maxb, env_b);
        size_t a = 0;
        while (a < items.size()) {
            if ((int)items[a].toks->size() > impl_->Tmax) { ++a; continue; }       // too long for the device: CPU below
            size_t b = a; int rows = 0;
            while (b < items.size() && (int)(b - a) < lim && rows + (int)items[b].toks->size() <= impl_->Tmax) rows += (int)items[b++].toks->size();
            if (forward_gpu(items, a, b, lg.data() + a * V))
                for (size_t k = a; k < b; ++k) done[k] = 1;
            a = b;
        }
    }
    static const bool dbg = getenv("FERRO_D1_DBG") != nullptr;
    std::vector<float> lc;
    for (size_t k = 0; k < items.size(); ++k) {
        const D1Item& it = items[k];
        float* lk = lg.data() + k * V;
        if (dbg && done[k]) {                                           // GPU-vs-CPU logits for this prompt
            lc.resize(V);
            forward(*it.toks, lc.data(), it.media_rows, it.media_pos);
            float tv = 0, tvg = 0; int amg = 0, amc = 0;
            for (size_t i = 1; i < V; ++i) { tv = std::max(tv, std::fabs(lk[i] - lc[i])); if (lk[i] > lk[amg]) amg = (int)i; if (lc[i] > lc[amc]) amc = (int)i; }
            for (const auto& lab : *it.labels) for (int t : lab) tvg = std::max(tvg, std::fabs(lk[t] - lc[t]));
            fprintf(stderr, "[d1-dbg] T=%d gpu=1 TV_all=%.4f TV_labels=%.4f am g=%d c=%d lgg=%.3f lgc=%.3f\n",
                    (int)it.toks->size(), tv, tvg, amg, amc, lk[amg], lc[amc]);
        }
        if (!done[k]) forward(*it.toks, lk, it.media_rows, it.media_pos);
    }
    scores.assign(items.size(), {});
    for (size_t k = 0; k < items.size(); ++k) {
        const float* lk = lg.data() + k * V;
        scores[k].assign(items[k].labels->size(), -1e30f);
        for (size_t j = 0; j < items[k].labels->size(); ++j)
            for (int t : (*items[k].labels)[j]) scores[k][j] = std::max(scores[k][j], lk[t]);
    }
    return true;
}

}  // namespace ferro
