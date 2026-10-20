// model_eg2.cpp — EmbeddingGemma-2 load + forward
#include "model_eg2.h"
#include "threadpool.h"
#include "gpu_cl.h"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <stdexcept>
#include <algorithm>

namespace ferro {

// aligned f32 arena
struct Arena {
    float* p = nullptr; size_t n = 0;
    ~Arena() { free(p); }
    void ensure(size_t count) {
        if (n >= count) return;
        free(p);
        n = (count + 15) & ~size_t(15);
        p = (float*)aligned_alloc(64, ((n * 4 + 63) / 64) * 64);
        memset(p, 0, n * 4);
    }
    float* get(size_t off) { return p + off; }
};

static std::vector<float> load_f32(const GgufFile& g, const std::string& name) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    size_t n = 1; for (auto d : t->dims) n *= d;
    std::vector<float> v(n);
    const uint8_t* src = g.tensor_data(*t);
    if (t->type == T_F32) memcpy(v.data(), src, n * 4);
    else if (t->type == T_BF16) {
        const uint16_t* s = (const uint16_t*)src;
        for (size_t i = 0; i < n; ++i) v[i] = bf16_to_f32(s[i]);
    } else if (t->type == T_F16) {
        const uint16_t* s = (const uint16_t*)src;
        for (size_t i = 0; i < n; ++i) v[i] = fp16_to_f32(s[i]);
    } else throw std::runtime_error("unsupported f32-load type for " + name);
    return v;
}

static bool g_gpu_weights = false;   // GPU initialized: skip f32 planes for GPU-resident matrices

static WMat load_q8(const GgufFile& g, const std::string& name) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    if (t->type != T_Q8_0) throw std::runtime_error(name + " is not Q8_0");
    int cols = (int)t->dims[0], rows = (int)t->dims[1];
    int nb = cols / QK8_0;
    WMat W = pack_q8(g.tensor_data(*t), rows, cols, nb * (int)sizeof(block_q8_0), name.c_str());
    static const bool hetero = getenv("FERRO_HETERO") && atoi(getenv("FERRO_HETERO")) != 0;
    if (!getenv("FERRO_LOW_MEM") && (!g_gpu_weights || hetero) && name != "token_embd.weight") pack_q8_to_f32(W);  // embd is gather-only
    if (getenv("FERRO_TIME") && W.rows >= 1024)
        fprintf(stderr, "[w] %-28s %dx%d wf=%zu\n", name.c_str(), W.rows, W.cols, W.wf.size());
    if (getenv("FERRO_DBGW") && name == "blk.0.attn_q.weight") {
        const char* d = getenv("FERRO_DBGWDIR");
        if (d) {
            std::vector<float> dq((size_t)rows * cols);
            for (int r = 0; r < rows; ++r)
                for (int b = 0; b < nb; ++b) {
                    float s = W.ws[(size_t)r * nb + b];
                    for (int i = 0; i < QK8_0; ++i) {
                        int c = b * QK8_0 + i;
                        if (c < cols) dq[(size_t)r * cols + c] = (float)W.wq[(size_t)r * cols + c] * s;
                    }
                }
            char path[512]; snprintf(path, sizeof path, "%s/load_wq.bin", d);
            FILE* fp = fopen(path, "wb"); size_t n = dq.size();
            fwrite(&n, sizeof n, 1, fp); fwrite(dq.data(), 4, n, fp); fclose(fp);
            snprintf(path, sizeof path, "%s/load_wq_plane.bin", d);
            fp = fopen(path, "wb");
            std::vector<float> wqf((size_t)rows * cols);
            for (size_t i = 0; i < wqf.size(); ++i) wqf[i] = (float)W.wq[i];
            n = wqf.size(); fwrite(&n, sizeof n, 1, fp); fwrite(wqf.data(), 4, n, fp); fclose(fp);
            snprintf(path, sizeof path, "%s/load_ws_plane.bin", d);
            fp = fopen(path, "wb"); n = W.ws.size();
            fwrite(&n, sizeof n, 1, fp); fwrite(W.ws.data(), 4, n, fp); fclose(fp);
        }
    }
    return W;
}


// ================= iGPU on-device layer loop =================
struct GpuLayerBufs { gpu::Buf attn_norm, q_norm, k_norm, attn_post_norm, ffn_norm, ffn_post_norm, ple_post_norm; };
static std::vector<GpuLayerBufs> g_gl;
static gpu::Buf g_cos_full = nullptr, g_sin_full = nullptr, g_cos_swa = nullptr, g_sin_swa = nullptr;
static bool g_gpu_layers = false;
static WMat g_plm_q8;                   // plm_proj quantized to Q8 blocks for the GPU (DP4A)
static gpu::Buf g_plm_norm = nullptr;

static void gpu_prepare_layers(const EG2Model& m) {
    for (auto& L : m.layers)
        for (const WMat* W : {&L.wqkv, &L.wo, &L.ffn_gateup, &L.ffn_down, &L.ple_gate, &L.ple_proj})
            if (W->gpu_slot < 0) { fprintf(stderr, "[gpu] not all GEMMs resident -> per-GEMM offload only\n"); return; }
    auto up = [](const std::vector<float>& v) { return gpu::upload(v.data(), v.size() * 4); };
    for (auto& L : m.layers) {
        GpuLayerBufs b{up(L.attn_norm), up(L.q_norm), up(L.k_norm), up(L.attn_post_norm),
                       up(L.ffn_norm), up(L.ffn_post_norm), up(L.ple_post_norm)};
        for (gpu::Buf x : {b.attn_norm, b.q_norm, b.k_norm, b.attn_post_norm, b.ffn_norm, b.ffn_post_norm, b.ple_post_norm})
            if (!x) { fprintf(stderr, "[gpu] norm upload failed -> per-GEMM offload only\n"); return; }
        g_gl.push_back(b);
    }
    // PLM projection [n_layer*n_pl x n_embd] -> Q8 blocks -> GPU (one DP4A GEMM per forward)
    g_plm_q8 = pack_from_f32(m.plm_proj.data(), m.n_pl * m.n_layer, m.n_embd, m.n_embd, "plm_proj");
    g_plm_q8.gpu_slot = gpu::upload_q8(g_plm_q8);
    g_plm_norm = up(m.plm_norm);
    if (g_plm_q8.gpu_slot < 0 || !g_plm_norm) { fprintf(stderr, "[gpu] plm upload failed\n"); return; }
    g_cos_full = up(m.rope_full.cosb); g_sin_full = up(m.rope_full.sinb);
    g_cos_swa  = up(m.rope_swa.cosb);  g_sin_swa  = up(m.rope_swa.sinb);
    if (!g_cos_full || !g_sin_full || !g_cos_swa || !g_sin_swa) return;
    g_gpu_layers = true;
    fprintf(stderr, "[gpu] on-device layer loop enabled (%.1f MB resident)\n", gpu::resident_bytes() / 1e6);
}

// FERRO_GPU_PROF2: sync around each op, attribute wall time to op class (timing only)
static double g_pt[16]; static int g_pcur = -1; static std::chrono::steady_clock::time_point g_pt0;
static bool prof_tick(int bit) {
    static const bool on = getenv("FERRO_GPU_PROF2") != nullptr;
    if (!on) return true;
    gpu::finish();
    auto now = std::chrono::steady_clock::now();
    if (g_pcur >= 0) g_pt[g_pcur] += std::chrono::duration<double, std::milli>(now - g_pt0).count();
    g_pcur = bit; g_pt0 = now;
    return true;
}
// Runs all transformer layers on the iGPU. In: A [T][E0] embeddings, H [L][T][n_pl] PLE inputs.
// Out: A overwritten with the final hidden state. Returns false (A untouched) on any failure.
static bool layers_gpu(const EG2Model& m, const int* seg, const int* pos, int T, float* A, const float* H) {
    if (!g_gpu_layers) return false;
    const int E0 = m.n_embd, FF = m.n_ff, NPL = m.n_pl;
    static int Tcap = 0;
    static gpu::Buf bA, bB, bC, bOB, bD, bE, bG, bF, bI, bH, bpos, blo, bhi;
    static std::vector<int> lo, hi;
    int max_rowc = 0, max_qsz = 0;
    for (auto& L : m.layers) {
        max_rowc = std::max(max_rowc, (L.n_head + 2 * L.n_head_kv) * L.head_dim);
        max_qsz = std::max(max_qsz, L.n_head * L.head_dim);
    }
    // segment bounds (contiguous segments)
    lo.resize(T); hi.resize(T);
    int maxseg = 0;
    for (int t = 0; t < T; ) {
        int u = t; while (u + 1 < T && seg[u + 1] == seg[t]) ++u;
        for (int k = t; k <= u; ++k) { lo[k] = t; hi[k] = u; }
        maxseg = std::max(maxseg, u - t + 1); t = u + 1;
    }
    if (maxseg > gpu::attn_max_seg()) return false;
    if (T > Tcap) {
        for (gpu::Buf* b : {&bA, &bB, &bC, &bOB, &bD, &bE, &bG, &bF, &bI, &bH, &bpos, &blo, &bhi}) { gpu::release(*b); *b = nullptr; }
        const size_t t = (size_t)std::max(T, 64);
        bA = gpu::alloc(t * E0 * 4); bB = gpu::alloc(t * E0 * 4); bE = gpu::alloc(t * E0 * 4);
        bD = gpu::alloc(t * E0 * 4); bC = gpu::alloc(t * max_rowc * 4); bOB = gpu::alloc(t * max_qsz * 4);
        bG = gpu::alloc(t * 2 * FF * 4); bF = gpu::alloc(t * FF * 4); bI = gpu::alloc(t * NPL * 4);
        bH = gpu::alloc(t * m.n_layer * NPL * 4);
        bpos = gpu::alloc(t * 4); blo = gpu::alloc(t * 4); bhi = gpu::alloc(t * 4);
        for (gpu::Buf b : {bA, bB, bC, bOB, bD, bE, bG, bF, bI, bH, bpos, blo, bhi}) if (!b) { Tcap = 0; return false; }
        Tcap = (int)t;
    }
    bool ok = gpu::write(bA, A, (size_t)T * E0 * 4) &&
              gpu::write(bpos, pos, (size_t)T * 4) && gpu::write(blo, lo.data(), (size_t)T * 4) &&
              gpu::write(bhi, hi.data(), (size_t)T * 4);
    // PLE inputs: GPU PLM GEMM (layout [t][l][npl], hld = L*npl) or host-provided H ([l][t][npl])
    const bool plm_gpu = (H == nullptr);
    const int hld = plm_gpu ? m.n_layer * NPL : NPL;
    auto hoff_of = [&](int il) { return plm_gpu ? il * NPL : il * T * NPL; };
    const int n = T * E0;
    const float eps = m.rms_eps;
    static const int skip = getenv("FERRO_GPU_SKIP") ? atoi(getenv("FERRO_GPU_SKIP")) : 0;  // 1 gemm 2 attn 4 norm 8 elem
    static const bool unfused = getenv("FERRO_GPU_UNFUSED") != nullptr;
    int max_cols = std::max(std::max(E0, FF), NPL);
    for (auto& L : m.layers) max_cols = std::max(max_cols, L.n_head * L.head_dim);
    ok = ok && gpu::ensure_qscratch(T, max_cols);
    if (plm_gpu)
        ok = ok && gpu::op_row_fused(bA, nullptr, nullptr, 1.0f, nullptr, nullptr, false, true, T, E0, eps)  // quant(A)
                && gpu::op_gemm_q(g_plm_q8.gpu_slot, T, bH)                                                   // [T][L*npl]
                && gpu::op_rmsnorm(bH, 0, hld, bH, 0, hld, T, m.n_layer, NPL, g_plm_norm, eps);               // per (t,l)
    else
        ok = ok && gpu::write(bH, H, (size_t)m.n_layer * T * NPL * 4);
    if (!unfused) {
        // ---- fused: 13 launches/layer; every producer emits the next GEMM's int8 input ----
        ok = ok && gpu::op_row_fused(bA, nullptr, nullptr, 1.0f, nullptr, g_gl[0].attn_norm, true, true, T, E0, eps);
        for (int il = 0; ok && il < m.n_layer; ++il) {
            const auto& L = m.layers[il];
            const auto& g = g_gl[il];
            const int nq = L.n_head, nkv = L.n_head_kv, hd = L.head_dim;
            const int qsz = nq * hd, ksz = nkv * hd, rowc = qsz + 2 * ksz;
            const RopeTable& rt = L.is_swa ? m.rope_swa : m.rope_full;
            gpu::Buf cb = L.is_swa ? g_cos_swa : g_cos_full, sb = L.is_swa ? g_sin_swa : g_sin_full;
            const bool last = (il + 1 == m.n_layer);
            ok = ok && gpu::op_gemm_q(L.wqkv.gpu_slot, T, bC)
                    && gpu::op_qkv_post(bC, rowc, T, nq, nkv, hd, g.q_norm, g.k_norm, cb, sb, bpos, rt.rot / 2, eps)
                    && gpu::op_attention_q(bC, rowc, 0, qsz, qsz + ksz, T, nq, nkv, hd, blo, bhi,
                                           L.is_swa ? m.swa_window : -1, 1.0f)
                    && gpu::op_gemm_q(L.wo.gpu_slot, T, bD)
                    && gpu::op_row_fused(bA, bD, g.attn_post_norm, 1.0f, bE, g.ffn_norm, true, true, T, E0, eps)
                    && gpu::op_gemm_q(L.ffn_gateup.gpu_slot, T, bG)
                    && gpu::op_gelu_mul_q(bG, T, FF)
                    && gpu::op_gemm_q(L.ffn_down.gpu_slot, T, bD)
                    && gpu::op_row_fused(bE, bD, g.ffn_post_norm, 1.0f, bE, nullptr, false, true, T, E0, eps)
                    && gpu::op_gemm_q(L.ple_gate.gpu_slot, T, bI)
                    && gpu::op_gelu_mulvec_q(bI, bH, hoff_of(il), hld, T, NPL)
                    && gpu::op_gemm_q(L.ple_proj.gpu_slot, T, bD)
                    && gpu::op_row_fused(bE, bD, g.ple_post_norm, L.out_scale, bA,
                                         last ? nullptr : g_gl[il + 1].attn_norm, !last, !last, T, E0, eps);
        }
    } else {
#define SKIPOK(bit) (skip & (bit)) ? true : prof_tick(bit) &&
    for (int il = 0; ok && il < m.n_layer; ++il) {
        const auto& L = m.layers[il];
        const auto& g = g_gl[il];
        const int nq = L.n_head, nkv = L.n_head_kv, hd = L.head_dim;
        const int qsz = nq * hd, ksz = nkv * hd, rowc = qsz + 2 * ksz;
        const RopeTable& rt = L.is_swa ? m.rope_swa : m.rope_full;
        gpu::Buf cb = L.is_swa ? g_cos_swa : g_cos_full, sb = L.is_swa ? g_sin_swa : g_sin_full;
        const int half = rt.rot / 2;
        // attention block
        ok = ok && SKIPOK(4) gpu::op_rmsnorm(bA, 0, E0, bB, 0, E0, T, 1, E0, g.attn_norm, eps)
                && SKIPOK(1) gpu::op_gemm(L.wqkv.gpu_slot, bB, T, bC)
                && SKIPOK(4) gpu::op_rmsnorm(bC, 0, rowc, bC, 0, rowc, T, nq, hd, g.q_norm, eps)
                && SKIPOK(4) gpu::op_rmsnorm(bC, qsz, rowc, bC, qsz, rowc, T, nkv, hd, g.k_norm, eps)
                && SKIPOK(4) gpu::op_rmsnorm(bC, qsz + ksz, rowc, bC, qsz + ksz, rowc, T, nkv, hd, nullptr, eps)
                && SKIPOK(8) gpu::op_rope(bC, 0, rowc, T, nq, hd, half, cb, sb, bpos)
                && SKIPOK(8) gpu::op_rope(bC, qsz, rowc, T, nkv, hd, half, cb, sb, bpos)
                && SKIPOK(2) gpu::op_attention(bC, rowc, 0, qsz, qsz + ksz, T, nq, nkv, hd, blo, bhi,
                                     L.is_swa ? m.swa_window : -1, 1.0f, bOB, qsz)
                && SKIPOK(1) gpu::op_gemm(L.wo.gpu_slot, bOB, T, bD)
                && SKIPOK(4) gpu::op_rmsnorm(bD, 0, E0, bD, 0, E0, T, 1, E0, g.attn_post_norm, eps)
                && SKIPOK(8) gpu::op_add_scale(bD, bA, bE, n, 1.0f);
        // FFN
        ok = ok && SKIPOK(4) gpu::op_rmsnorm(bE, 0, E0, bB, 0, E0, T, 1, E0, g.ffn_norm, eps)
                && SKIPOK(1) gpu::op_gemm(L.ffn_gateup.gpu_slot, bB, T, bG)
                && SKIPOK(8) gpu::op_gelu_mul(bG, T, FF, bF)
                && SKIPOK(1) gpu::op_gemm(L.ffn_down.gpu_slot, bF, T, bD)
                && SKIPOK(4) gpu::op_rmsnorm(bD, 0, E0, bD, 0, E0, T, 1, E0, g.ffn_post_norm, eps)
                && SKIPOK(8) gpu::op_add_scale(bD, bE, bE, n, 1.0f);
        // per-layer embedding branch
        ok = ok && SKIPOK(1) gpu::op_gemm(L.ple_gate.gpu_slot, bE, T, bI)
                && SKIPOK(8) gpu::op_gelu_mulvec(bI, bH, il * T * NPL, T * NPL)
                && SKIPOK(1) gpu::op_gemm(L.ple_proj.gpu_slot, bI, T, bD)
                && SKIPOK(4) gpu::op_rmsnorm(bD, 0, E0, bD, 0, E0, T, 1, E0, g.ple_post_norm, eps)
                && SKIPOK(8) gpu::op_add_scale(bE, bD, bA, n, L.out_scale);
    }
#undef SKIPOK
    }
    if (ok && getenv("FERRO_GPU_KBENCH")) {
        // per-kernel device time: re-launch on live buffers, N back-to-back, one finish (timing only)
        auto tb = [&](const char* name, int per_fwd, auto fn) {
            const int N = 30; gpu::finish();
            auto a = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) fn();
            gpu::finish();
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count() / N;
            fprintf(stderr, "[kbench] %-22s %7.3f ms/launch  x%-3d = %6.1f ms/fwd\n", name, ms, per_fwd, ms * per_fwd);
        };
        int nswa = 0; for (auto& L : m.layers) nswa += L.is_swa;
        int ifull = 0; while (m.layers[ifull].is_swa) ++ifull;
        for (int which : {0, ifull}) {
            const auto& L = m.layers[which]; const auto& g = g_gl[which];
            const int nq = L.n_head, nkv = L.n_head_kv, hd = L.head_dim, qsz = nq * hd, ksz = nkv * hd, rowc = qsz + 2 * ksz;
            const int cnt = L.is_swa ? nswa : m.n_layer - nswa;
            const char* tag = L.is_swa ? "swa" : "full";
            char nm[64];
            snprintf(nm, sizeof nm, "gemm qkv (%s)", tag);  tb(nm, cnt, [&]{ gpu::op_gemm_q(L.wqkv.gpu_slot, T, bC); });
            snprintf(nm, sizeof nm, "qkv_post (%s)", tag);  tb(nm, cnt, [&]{ gpu::op_qkv_post(bC, rowc, T, nq, nkv, hd, g.q_norm, g.k_norm,
                L.is_swa ? g_cos_swa : g_cos_full, L.is_swa ? g_sin_swa : g_sin_full, bpos, (L.is_swa ? m.rope_swa : m.rope_full).rot / 2, eps); });
            snprintf(nm, sizeof nm, "attention_q (%s)", tag); tb(nm, cnt, [&]{ gpu::op_attention_q(bC, rowc, 0, qsz, qsz + ksz, T, nq, nkv, hd,
                blo, bhi, L.is_swa ? m.swa_window : -1, 1.0f); });
            snprintf(nm, sizeof nm, "gemm wo (%s)", tag);   tb(nm, cnt, [&]{ gpu::op_gemm_q(L.wo.gpu_slot, T, bD); });
        }
        const auto& L = m.layers[0]; const auto& g = g_gl[0];
        tb("row_fused (norm+q)", m.n_layer * 2, [&]{ gpu::op_row_fused(bA, bD, g.attn_post_norm, 1.0f, bB, g.ffn_norm, true, true, T, E0, eps); });
        tb("row_fused (no norm q)", m.n_layer, [&]{ gpu::op_row_fused(bE, bD, g.ffn_post_norm, 1.0f, bB, nullptr, false, true, T, E0, eps); });
        tb("gemm gateup", m.n_layer, [&]{ gpu::op_gemm_q(L.ffn_gateup.gpu_slot, T, bG); });
        tb("gelu_mul_q", m.n_layer, [&]{ gpu::op_gelu_mul_q(bG, T, FF); });
        tb("gemm down", m.n_layer, [&]{ gpu::op_gemm_q(L.ffn_down.gpu_slot, T, bD); });
        tb("gemm ple_gate", m.n_layer, [&]{ gpu::op_gemm_q(L.ple_gate.gpu_slot, T, bI); });
        tb("gelu_mulvec_q", m.n_layer, [&]{ gpu::op_gelu_mulvec_q(bI, bH, 0, hld, T, NPL); });
        tb("plm gemm (q8)", 1, [&]{ gpu::op_gemm_q(g_plm_q8.gpu_slot, T, bH); });
        tb("gemm ple_proj", m.n_layer, [&]{ gpu::op_gemm_q(L.ple_proj.gpu_slot, T, bD); });
    }
    if (getenv("FERRO_GPU_PROF2")) {
        prof_tick(0); g_pcur = -1;
        static int calls = 0; ++calls;
        fprintf(stderr, "[gpu-prof2] per fwd: gemm %.1f | attn %.1f | norm %.1f | elem/rope %.1f ms\n",
                g_pt[1] / calls, g_pt[2] / calls, g_pt[4] / calls, g_pt[8] / calls);
    }
    if (!ok) { gpu::finish(); return false; }
    static std::vector<float> tmp;
    tmp.resize((size_t)n);
    if (!gpu::read(bA, tmp.data(), (size_t)n * 4)) return false;
    memcpy(A, tmp.data(), (size_t)n * 4);
    return true;
}

bool EG2Model::load(const std::string& path, std::string* err) {
    std::string a = path;
    if (!gguf.open_file(path.c_str(), err)) return false;
    const std::string P = gguf.arch + ".";
    auto I = [&](const std::string& k, int64_t d = 0) { return gguf.get_int(P + k, d); };
    auto F = [&](const std::string& k, double d = 0) { return gguf.get_f64(P + k, d); };

    {
        // iGPU by default when an OpenCL GPU with int8 dot product is present; FERRO_GPU=0 / --cpu disables
        const char* ge = getenv("FERRO_GPU");
        if (!ge || atoi(ge) != 0) g_gpu_weights = gpu::init(ge != nullptr || getenv("FERRO_VERBOSE"));
    }
    n_layer   = (int)I("block_count");
    n_embd    = (int)I("embedding_length");
    n_ff      = (int)I("feed_forward_length");
    n_pl      = (int)I("embedding_length_per_layer_input");
    n_embd_out= (int)I("embedding_length_out", n_embd);
    rms_eps   = (float)F("attention.layer_norm_rms_epsilon", 1e-6);
    swa_window= (int)I("attention.sliding_window", 0);
    rope_base = F("rope.freq_base", 1e6);
    rope_base_swa = F("rope.freq_base_swa", rope_base);
    auto toks = gguf.get_strarr("tokenizer.ggml.tokens");
    n_vocab = (int)toks.size();

    // per-layer head/kv/head-dim geometry
    auto kv_arr = gguf.get_iarr(P + "attention.head_count_kv");
    auto swa_arr = gguf.get_iarr(P + "attention.sliding_window_pattern");
    int hd_full = (int)I("attention.key_length");        // per-head k len for full layers
    int hd_swa  = (int)I("attention.key_length_swa", hd_full);
    int rot_full= (int)I("rope.dimension_count", hd_full);
    int rot_swa = (int)I("rope.dimension_count_swa", hd_swa);

    tok_embd = load_q8(gguf, "token_embd.weight");
    // per-layer model proj: BF16 [n_embd, n_pl*n_layer] -> f32 plane, pre-scaled 1/sqrt(n_embd)
    {
        auto v = load_f32(gguf, "per_layer_model_proj.weight");
        size_t total = v.size();
        float sc = 1.0f / std::sqrt((float)n_embd);
        for (size_t i = 0; i < total; ++i) v[i] *= sc;
        plm_proj = std::move(v);
    }
    plm_norm = load_f32(gguf, "per_layer_proj_norm.weight");
    output_norm = load_f32(gguf, "output_norm.weight");
    output = load_q8(gguf, "output.weight");

    layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        auto& L = layers[il];
        std::string b = "blk." + std::to_string(il) + ".";
        L.is_swa = swa_arr.empty() ? (il % 6 != 5) : (swa_arr[il] != 0);
        L.n_head = (int)I("attention.head_count", 4);
        L.n_head_kv = kv_arr.empty() ? L.n_head : (int)kv_arr[il];
        L.head_dim = L.is_swa ? hd_swa : hd_full;
        L.n_rot    = L.is_swa ? rot_swa : rot_full;

        WMat wq = load_q8(gguf, b + "attn_q.weight");
        WMat wk = load_q8(gguf, b + "attn_k.weight");
        WMat wv = load_q8(gguf, b + "attn_v.weight");
        std::vector<const WMat*> parts{&wq, &wk, &wv};
        if (il == 0 && getenv("FERRO_DBGW")) {
            const char* d = getenv("FERRO_DBGWDIR");
            if (d) {
                char path[512]; snprintf(path, sizeof path, "%s/preconcat_wq_ws.bin", d);
                FILE* fp = fopen(path, "wb"); size_t n = wq.ws.size();
                fprintf(stderr, "[dbg] preconcat wq.ws ptr=%p n=%zu rows=%d cols=%d nb=%d\n", (void*)wq.ws.data(), n, wq.rows, wq.cols, wq.nb);
                fwrite(&n, sizeof n, 1, fp); fwrite(wq.ws.data(), 4, n, fp); fclose(fp);
            }
        }
        L.wqkv = pack_concat(parts, "wqkv");
        if (il == 0 && getenv("FERRO_DBGW")) {
            const char* d = getenv("FERRO_DBGWDIR");
            if (d) {
                char pp[512]; snprintf(pp, sizeof pp, "%s/postconcat_wq_ws.bin", d);
                FILE* fp2 = fopen(pp, "wb"); size_t m = wq.ws.size();
                fwrite(&m, sizeof m, 1, fp2); fwrite(wq.ws.data(), 4, m, fp2); fclose(fp2);
                snprintf(pp, sizeof pp, "%s/preconcat_wk_ws.bin", d);
                fp2 = fopen(pp, "wb"); m = wk.ws.size();
                fwrite(&m, sizeof m, 1, fp2); fwrite(wk.ws.data(), 4, m, fp2); fclose(fp2);
                snprintf(pp, sizeof pp, "%s/preconcat_wv_ws.bin", d);
                fp2 = fopen(pp, "wb"); m = wv.ws.size();
                fwrite(&m, sizeof m, 1, fp2); fwrite(wv.ws.data(), 4, m, fp2); fclose(fp2);
            }
        }
        if (il == 0 && getenv("FERRO_DBGW")) {
            const char* d = getenv("FERRO_DBGWDIR");
            if (d) {
                char path[512]; snprintf(path, sizeof path, "%s/concat_ws.bin", d);
                FILE* fp = fopen(path, "wb"); size_t n = L.wqkv.ws.size();
                fwrite(&n, sizeof n, 1, fp); fwrite(L.wqkv.ws.data(), 4, n, fp); fclose(fp);
            }
        }
        wq = WMat(); wk = WMat(); wv = WMat();
        L.wo = load_q8(gguf, b + "attn_output.weight");
        WMat up = load_q8(gguf, b + "ffn_up.weight");
        WMat gate = load_q8(gguf, b + "ffn_gate.weight");
        std::vector<const WMat*> gu{&gate, &up};   // gate FIRST: gelu applied to first half
        L.ffn_gateup = pack_concat(gu, "gateup");
        up = WMat(); gate = WMat();
        L.ffn_down = load_q8(gguf, b + "ffn_down.weight");
        L.ple_gate = load_q8(gguf, b + "inp_gate.weight");
L.ple_proj = load_q8(gguf, b + "proj.weight");

        L.attn_norm       = load_f32(gguf, b + "attn_norm.weight");
        L.q_norm          = load_f32(gguf, b + "attn_q_norm.weight");
        L.k_norm          = load_f32(gguf, b + "attn_k_norm.weight");
        L.attn_post_norm  = load_f32(gguf, b + "post_attention_norm.weight");
        L.ffn_norm        = load_f32(gguf, b + "ffn_norm.weight");
        L.ffn_post_norm   = load_f32(gguf, b + "post_ffw_norm.weight");
        L.ple_post_norm   = load_f32(gguf, b + "post_norm.weight");
        L.out_scale       = load_f32(gguf, b + "layer_output_scale.weight")[0];
        is_swa.push_back(L.is_swa);
    }
    if (g_gpu_weights) {
        // make the big per-layer GEMMs GPU-resident; anything not uploadable gets CPU f32 planes
        size_t ok = 0, tot = 0;
        for (auto& L : layers)
            for (WMat* W : {&L.wqkv, &L.wo, &L.ffn_gateup, &L.ffn_down, &L.ple_gate, &L.ple_proj}) {
                ++tot;
                W->gpu_slot = gpu::upload_q8(*W);
                if (W->gpu_slot >= 0) ++ok;
                else if (!getenv("FERRO_LOW_MEM")) pack_q8_to_f32(*W);
            }
        if (!getenv("FERRO_LOW_MEM")) pack_q8_to_f32(output);   // pooled output GEMM stays on CPU
        fprintf(stderr, "[gpu] %zu/%zu matrices resident on %s (%.1f MB)\n", ok, tot,
                gpu::device_name(), gpu::resident_bytes() / 1e6);
    }
    rope_full.build(rope_base,     rot_full, 8192);
    rope_swa.build(rope_base_swa ? rope_base_swa : rope_base, rot_swa, 8192);
    if (g_gpu_weights) gpu_prepare_layers(*this);
    return true;
}

// ---------------- engine ----------------
static void dbg(const char* tag, const float* p, size_t n) {
    static int on = getenv("FERRO_DBG") ? 1 : 0;
    static const char* dir = getenv("FERRO_DBGDIR");
    if (!on && !dir) return;
    double mx = 0; size_t nan = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = p[i];
        if (std::isnan(v) || std::isinf(v)) { ++nan; continue; }
        mx = std::max(mx, (double)std::fabs(v));
    }
    if (on) fprintf(stderr, "[dbg] %-22s maxabs=%.4g nan=%zu\n", tag, mx, nan);
    if (dir) {
        char path[512]; snprintf(path, sizeof path, "%s/%s.bin", dir, tag);
        FILE* fp = fopen(path, "wb");
        if (fp) { fwrite(&n, sizeof n, 1, fp); fwrite(p, 4, n, fp); fclose(fp); }
    }
}
struct Engine::Impl {
    int T_max = 0;
    Arena A, B, C, OB, D, E, G, H, I, J, F;
    struct {
        Arena PLMB;
    } scratch_;
    // A: inpL [T][512]; B: norm scratch; C: qkv [T][3072]; OB: attn out [T][2048];
    // D: gemm out [T][2048? down=512 but reuse]; E: attn_res/pe_in; G: gateup [T][4096];
    // H: ple_base [24][T][512]; I: ple scratch [T][512]; J: out [T][768]; F: ffn hidden [T][2048]
    void ensure(int T, const EG2Model& m) {
        if (T <= T_max) return;
        T_max = std::max(T, 64);
        size_t t = T_max;
        int max_rowc = 0;
        for (auto& L : m.layers)
            max_rowc = std::max(max_rowc, (L.n_head + 2 * L.n_head_kv) * L.head_dim);
        A.ensure(t * m.n_embd);
        B.ensure(t * m.n_embd);
        C.ensure((size_t)t * max_rowc);
        OB.ensure(t * 2048);
        D.ensure(t * std::max(m.n_embd, m.n_ff));
        E.ensure(t * m.n_embd);
        G.ensure(t * 2 * m.n_ff);
        H.ensure((size_t)t * m.n_layer * m.n_pl);
        scratch_.PLMB.ensure((size_t)t * m.n_layer * m.n_pl);
        I.ensure(t * m.n_pl);
        J.ensure(t * m.n_embd_out);
        F.ensure(t * m.n_ff);
    }
};

bool Engine::init(const std::string& model_path, int n_threads, std::string* err) {
    if (!m_.load(model_path, err)) return false;
    Pool::get().init(n_threads);
    impl_ = std::make_shared<Impl>();
    return true;
}

// Heterogeneous split: sequences [0,k) run on the iGPU (driver thread) while [k,n) run on the CPU pool.
// Split point adapts to the measured rates so both halves finish together.
void Engine::encode(const int* ids, const int* seg, const int* pos, int T, int n_seq, float* out) {
    static const bool hetero = []{ const char* e = getenv("FERRO_HETERO"); return e && atoi(e) != 0; }();
    if (!(hetero && g_gpu_layers && n_seq >= 2)) { encode_impl(*impl_, ids, seg, pos, T, n_seq, out, 0); return; }
    // pick the sequence boundary closest to gpu_share_ * T (at least one sequence on each side)
    const double target = gpu_share_ * T;
    int k = 0, tk = 0, best_k = 1, best_t = 0; double best_d = 1e30;
    for (int t = 0; t < T; ) {
        int u = t; while (u < T && seg[u] == seg[t]) ++u;
        ++k; tk = u;
        if (k < n_seq) { double d = std::fabs(tk - target); if (d < best_d) { best_d = d; best_k = k; best_t = tk; } }
        t = u;
    }
    const int kg = best_k, Tg = best_t;
    if (!impl_gpu_) impl_gpu_ = std::make_shared<Impl>();
    using clk = std::chrono::steady_clock;
    double ms_gpu = 0, ms_cpu = 0;
    std::thread gt([&] {
        auto a = clk::now();
        encode_impl(*impl_gpu_, ids, seg, pos, Tg, kg, out, 2);
        ms_gpu = std::chrono::duration<double, std::milli>(clk::now() - a).count();
    });
    auto a = clk::now();
    set_thread_cpu_only(true);
    encode_impl(*impl_, ids + Tg, seg + Tg, pos + Tg, T - Tg, n_seq - kg, out + (size_t)kg * m_.n_embd_out, 1);
    set_thread_cpu_only(false);
    ms_cpu = std::chrono::duration<double, std::milli>(clk::now() - a).count();
    gt.join();
    // rate-balance: share = r_gpu / (r_gpu + r_cpu), smoothed
    const double rg = Tg / std::max(ms_gpu, 1e-3), rc = (T - Tg) / std::max(ms_cpu, 1e-3);
    gpu_share_ = std::min(0.97, std::max(0.5, 0.6 * gpu_share_ + 0.4 * (rg / (rg + rc))));
    if (getenv("FERRO_TIME"))
        fprintf(stderr, "[hetero] gpu %d tok %.1f ms | cpu %d tok %.1f ms | next share %.3f\n",
                Tg, ms_gpu, T - Tg, ms_cpu, gpu_share_);
}

void Engine::encode_impl(Impl& in_, const int* ids, const int* seg, const int* pos, int T, int n_seq, float* out, int mode) {
    static const bool timing = getenv("FERRO_TIME") != nullptr;
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now(), tprev = t0;
    double t_embd = 0, t_plm = 0, t_attn = 0, t_ffn = 0, t_ple = 0, t_rest = 0;
    static double g_qkv=0,g_wo=0,g_gu=0,g_dn=0,g_pg=0,g_pp=0,g_other=0; static int g_n=0;
    static double x_attn=0,x_norm=0,x_gelu=0,x_rope=0,x_rest2=0;
    auto GT=[&](double& v, auto fn){ auto a=clk::now(); fn(); auto b=clk::now(); v+=std::chrono::duration<double,std::milli>(b-a).count(); };
#define STAGE(v) { auto n_ = clk::now(); v += std::chrono::duration<double,std::milli>(n_ - tprev).count(); tprev = n_; }
    auto& m = m_;
    auto& in = in_;
    in.ensure(T, m);
    const int E0 = m.n_embd, FF = m.n_ff;

    // 1. token embeddings scaled by sqrt(n_embd)
    gather_embd(m.tok_embd, ids, T, std::sqrt((float)E0), in.A.get(0));
    STAGE(t_embd);
    dbg("embd", in.A.get(0), (size_t)T * E0);

    static const bool gpu_layers_off = getenv("FERRO_GPU_LAYERS") && atoi(getenv("FERRO_GPU_LAYERS")) == 0;
    static const bool gpu_unfused = getenv("FERRO_GPU_UNFUSED") != nullptr;
    const bool want_gpu = mode != 1 && g_gpu_layers && !gpu_layers_off && !getenv("FERRO_DBGDIR");
    auto tg0 = clk::now();
    // fused GPU path computes the PLM projection on-device (H = nullptr)
    bool on_gpu = want_gpu && !gpu_unfused && layers_gpu(m, seg, pos, T, in.A.get(0), nullptr);
    if (!on_gpu) {
        STAGE(t_plm);
        // 2. per-layer input projections: H[l][t][:] = rmsnorm(plm_proj @ inp)
        gemm_f32(m.plm_proj.data(), m.n_pl * m.n_layer, E0, in.A.get(0), T, in.scratch_.PLMB.get(0), m.n_pl * m.n_layer);
        {
            static thread_local std::vector<float> tmp;
            tmp.resize((size_t)T * m.n_layer * m.n_pl);
            const float* y = in.scratch_.PLMB.get(0);   // [T][n_layer*n_pl]
            // rearrange Y[t][l*n_pl + j] -> H[l][t][j]
            for (int t = 0; t < T; ++t)
                for (int l = 0; l < m.n_layer; ++l)
                    memcpy(in.H.get((size_t)l * T * m.n_pl + (size_t)t * m.n_pl),
                           y + (size_t)t * m.n_layer * m.n_pl + (size_t)l * m.n_pl,
                           m.n_pl * 4);
            if (getenv("FERRO_DBGDIR")) dbg("plm_pre", in.scratch_.PLMB.get(0), (size_t)T * m.n_layer * m.n_pl);
            if (getenv("FERRO_DBGDIR")) {
                dbg("l0_h0_pre", in.H.get(0), (size_t)T * m.n_pl);
                dbg("plm_norm", m.plm_norm.data(), m.plm_norm.size());
            }
            // rmsnorm each layer's [T][n_pl] with plm_norm (in place; two-pass rms is safe in-place)
            for (int l = 0; l < m.n_layer; ++l)
                rmsnorm_q(in.H.get((size_t)l * T * m.n_pl), T, m.n_pl, m.plm_norm.data(), m.rms_eps,
                          in.H.get((size_t)l * T * m.n_pl));
        }
        if (want_gpu) on_gpu = layers_gpu(m, seg, pos, T, in.A.get(0), in.H.get(0));
    }
    if (timing && on_gpu) {
        double pre = std::chrono::duration<double, std::milli>(tg0 - t0).count();
        double lay = std::chrono::duration<double, std::milli>(clk::now() - tg0).count();
        fprintf(stderr, "[gpu-time] cpu prologue (embd) %.1f ms | gpu plm+layers %.1f ms\n", pre, lay);
    }
    for (int il = 0; !on_gpu && il < m.n_layer; ++il) {
        auto& L = m.layers[il];
        const int nq = L.n_head, nkv = L.n_head_kv, hd = L.head_dim;
        const int qsz = nq * hd, ksz = nkv * hd;
        const int rowc = qsz + 2 * ksz;
        const bool d0 = (il == 0);
        const bool dbg_all = (getenv("FERRO_DBGDIR") != nullptr);

        // attention block
        STAGE(t_rest);
        rmsnorm_q(in.A.get(0), T, E0, L.attn_norm.data(), m.rms_eps, in.B.get(0));
        if (timing) GT(g_qkv, [&]{ gemm_q8_fma(L.wqkv, in.B.get(0), T, nullptr, nullptr, in.C.get(0), rowc); });
        else gemm_q8_fma(L.wqkv, in.B.get(0), T, nullptr, nullptr, in.C.get(0), rowc);
        if (d0) {
            dbg("l0_normed", in.B.get(0), (size_t)T * E0);
            dbg("l0_qkv_pre", in.C.get(0), (size_t)T * rowc);
            static const bool dw = getenv("FERRO_DBGW") != nullptr;
            if (dw) {
                const WMat& W = L.wqkv;
                std::vector<float> wqf((size_t)W.rows * W.cols);
                for (size_t i = 0; i < wqf.size(); ++i) wqf[i] = (float)W.wq[i];
                dbg("l0_wq_plane", wqf.data(), wqf.size());
                dbg("l0_ws_plane", W.ws.data(), W.ws.size());
            }
        }
        // per-head q/k norm, weightless v norm (strided within qkv rows)
        for (int h = 0; h < nq; ++h)
            rmsnorm_q(in.C.get(0) + h * hd, T, hd, L.q_norm.data(), m.rms_eps, in.C.get(0) + h * hd, rowc, rowc);
        for (int h = 0; h < nkv; ++h)
            rmsnorm_q(in.C.get(0) + qsz + h * hd, T, hd, L.k_norm.data(), m.rms_eps, in.C.get(0) + qsz + h * hd, rowc, rowc);
        for (int h = 0; h < nkv; ++h)
            rmsnorm_w1(in.C.get(0) + qsz + ksz + h * hd, T, hd, m.rms_eps, in.C.get(0) + qsz + ksz + h * hd, rowc, rowc);
        // rope (full head rotation)
        const RopeTable& rt = L.is_swa ? m.rope_swa : m.rope_full;
        if (timing) GT(x_rope, [&]{ rope_apply(in.C.get(0), T, nq, hd, rt, pos, rowc);
                                    rope_apply(in.C.get(0) + qsz, T, nkv, hd, rt, pos, rowc); });
        else { rope_apply(in.C.get(0), T, nq, hd, rt, pos, rowc);
               rope_apply(in.C.get(0) + qsz, T, nkv, hd, rt, pos, rowc); }
        if (d0) dbg("l0_qkv", in.C.get(0), (size_t)T * rowc);
        // attention
        if (timing) GT(x_attn, [&]{ attention(in.C.get(0), in.C.get(0) + qsz, in.C.get(0) + qsz + ksz, in.OB.get(0),
                  T, nq, nkv, hd, 1.0f, seg, L.is_swa ? m.swa_window : -1, rowc, rowc, qsz); });
        else attention(in.C.get(0), in.C.get(0) + qsz, in.C.get(0) + qsz + ksz, in.OB.get(0),
                  T, nq, nkv, hd, 1.0f, seg, L.is_swa ? m.swa_window : -1, rowc, rowc, qsz);
        if (d0) dbg("l0_attn", in.OB.get(0), (size_t)T * qsz);
        if (timing) GT(g_wo, [&]{ gemm_q8_fma(L.wo, in.OB.get(0), T, nullptr, nullptr, in.D.get(0), E0); });
        else gemm_q8_fma(L.wo, in.OB.get(0), T, nullptr, nullptr, in.D.get(0), E0);
        rmsnorm_q(in.D.get(0), T, E0, L.attn_post_norm.data(), m.rms_eps, in.D.get(0));
        if (dbg_all) { char tb[64]; snprintf(tb, sizeof tb, "L%d_attnpost", il); dbg(tb, in.D.get(0), (size_t)T * E0); }
        // residual
        add_rows(in.D.get(0), in.A.get(0), (size_t)T * E0, in.E.get(0));       // attn_res
        STAGE(t_attn);
        // FFN
        rmsnorm_q(in.E.get(0), T, E0, L.ffn_norm.data(), m.rms_eps, in.B.get(0));
        if (timing) GT(g_gu, [&]{ gemm_q8_fma(L.ffn_gateup, in.B.get(0), T, nullptr, nullptr, in.G.get(0), 2 * FF); });
        else gemm_q8_fma(L.ffn_gateup, in.B.get(0), T, nullptr, nullptr, in.G.get(0), 2 * FF);
        if (dbg_all) { char tb[64]; snprintf(tb, sizeof tb, "L%d_gateup", il); dbg(tb, in.G.get(0), (size_t)T * 2 * FF); }
        if (timing) GT(x_gelu, [&]{ gelu_mul_block(in.G.get(0), T, FF, in.F.get(0)); });
        else gelu_mul_block(in.G.get(0), T, FF, in.F.get(0));
        if (dbg_all) { char tb[64]; snprintf(tb, sizeof tb, "L%d_gelumul", il); dbg(tb, in.F.get(0), (size_t)T * FF); }
        if (timing) GT(g_dn, [&]{ gemm_q8_fma(L.ffn_down, in.F.get(0), T, nullptr, nullptr, in.D.get(0), E0); });
        else gemm_q8_fma(L.ffn_down, in.F.get(0), T, nullptr, nullptr, in.D.get(0), E0);
        rmsnorm_q(in.D.get(0), T, E0, L.ffn_post_norm.data(), m.rms_eps, in.D.get(0));
        if (dbg_all) { char tb[64]; snprintf(tb, sizeof tb, "L%d_ffnpost", il); dbg(tb, in.D.get(0), (size_t)T * E0); }
        add_rows(in.D.get(0), in.E.get(0), (size_t)T * E0, in.E.get(0));       // pe_in
        STAGE(t_ffn);
        if (dbg_all) { char tb[64]; snprintf(tb, sizeof tb, "L%d_pein", il); dbg(tb, in.E.get(0), (size_t)T * E0); }

        // per-layer embedding branch
        if (timing) GT(g_pg, [&]{ gemm_q8_fma(L.ple_gate, in.E.get(0), T, nullptr, nullptr, in.I.get(0), m.n_pl); });
        else gemm_q8_fma(L.ple_gate, in.E.get(0), T, nullptr, nullptr, in.I.get(0), m.n_pl);
        if (dbg_all) { char tb[64]; snprintf(tb, sizeof tb, "L%d_gate_pre", il); dbg(tb, in.I.get(0), (size_t)T * m.n_pl); }
        {
            const float* hl = in.H.get((size_t)il * T * m.n_pl);
            float* iv = in.I.get(0);
            gelu_mulvec(iv, hl, (int)((size_t)T * m.n_pl));
            if (dbg_all) { char tb[64]; snprintf(tb, sizeof tb, "L%d_h0", il); dbg(tb, hl, (size_t)T * m.n_pl); }
        }
        if (timing) GT(g_pp, [&]{ gemm_q8_fma(L.ple_proj, in.I.get(0), T, nullptr, nullptr, in.D.get(0), E0); });
        else gemm_q8_fma(L.ple_proj, in.I.get(0), T, nullptr, nullptr, in.D.get(0), E0);
        rmsnorm_q(in.D.get(0), T, E0, L.ple_post_norm.data(), m.rms_eps, in.D.get(0));
        if (dbg_all) { char tb[64]; snprintf(tb, sizeof tb, "L%d_pleout", il); dbg(tb, in.D.get(0), (size_t)T * E0); }
        // A(next) = pe_in + ple, scaled
        add_rows(in.E.get(0), in.D.get(0), (size_t)T * E0, in.A.get(0));
        if (L.out_scale != 1.0f) scale_rows(L.out_scale, (size_t)T * E0, in.A.get(0));
        STAGE(t_ple);
        char tag[64]; snprintf(tag, sizeof tag, "layer%d_out", il);
        dbg(tag, in.A.get(0), (size_t)T * E0);
    }

    STAGE(t_rest);
    // final norm + per-segment mean pool + output projection (linear: pool before
    // projecting saves (T-n_seq)/T of the work) + L2
    rmsnorm_q(in.A.get(0), T, E0, m.output_norm.data(), m.rms_eps, in.B.get(0));
    static thread_local std::vector<float> pooled;
    pooled.assign((size_t)n_seq * E0, 0.f);
    float* const Bp = pooled.data();
    {
        int t0 = 0;
        for (int s = 0; s < n_seq; ++s) {
            int t1 = t0;
            while (t1 < T && seg[t1] == seg[t0]) ++t1;
            mean_pool(in.B.get(0) + (size_t)t0 * E0, t1 - t0, E0, Bp + (size_t)s * E0, false);
            t0 = t1;
        }
    }
    gemm_q8_fma(m.output, Bp, n_seq, nullptr, nullptr, in.J.get(0), m.n_embd_out);
    {
        int t0 = 0;
        for (int s = 0; s < n_seq; ++s) {
            int t1 = t0;
            while (t1 < T && seg[t1] == seg[t0]) ++t1;
            mean_pool(in.J.get(0) + (size_t)s * m.n_embd_out, 1, m.n_embd_out,
                      out + (size_t)s * m.n_embd_out, true);
            t0 = t1;
        }
    }
    if (timing)
        fprintf(stderr, "[time] embd %.1f | plm %.1f | attn %.1f | ffn %.1f | ple %.1f | rest %.1f (ms)\n",
                t_embd, t_plm, t_attn, t_ffn, t_ple, t_rest);
    if (timing) { ++g_n;
        fprintf(stderr, "[gemm] qkv %.0f wo %.0f gateup %.0f down %.0f pleg %.0f plep %.0f | non-gemm %.0f (ms/fwd)\n",
                g_qkv/g_n, g_wo/g_n, g_gu/g_n, g_dn/g_n, g_pg/g_n, g_pp/g_n,
                (t_embd+t_plm+t_attn+t_ffn+t_ple+t_rest - g_qkv-g_wo-g_gu-g_dn-g_pg-g_pp)/g_n);
        fprintf(stderr, "[sub] attention %.0f | gelu_mul %.0f | rope %.0f (ms/fwd)\n",
                x_attn/g_n, x_gelu/g_n, x_rope/g_n);
    }
    if (timing) STAGE(t_rest);

}

} // namespace ferro
