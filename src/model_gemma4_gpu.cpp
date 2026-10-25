// model_gemma4_gpu.cpp - resident iGPU forward for Gemma 4 (text path).
//
// Everything runs on the device between two host syncs per forward: the token embeddings + PLE rows are
// computed on the host (Q4_0 mmap rows, cheap), uploaded once per chunk, then the whole 35-layer stack runs
// as a launch stream with the KV cache in device buffers. Hidden state never round-trips: producers write
// the int8 activation scratch that the next Q8 GEMM reads (op_row_fused / op_gelu_mul_q / op_quant_q).
//
// Layout notes
//   own-KV layer: one QKV buffer [tc][q | k | v] (fused GEMM output, qd + 2*kd per row); the k/v half is then
//     appended to that layer's KV cache [ctx][k | v] with op_lm_kv_write, and attention reads Q from the QKV
//     buffer and K/V from the cache (op_attn_split).
//   shared-KV layer (>= n_kv_layers): q only, compact [tc][qd]; K/V come from the source layer's cache.
#include "model_gemma4.h"
#include "gpu_cl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ferro {

static const int G4_CHUNK = 128;                 // rows per prefill chunk (bounds the scratch)

struct G4Gpu {
    int ctx = 0, cap = G4_CHUNK;
    int d = 0, nq = 0, nkv = 0, pl = 0, n_layer = 0, n_vocab = 0;
    int qd_max = 0, kv_max = 0, ff_max = 0, rowc_max = 0;
    std::vector<gpu::Buf> kv;                    // per own-kv layer: [ctx][2*kd]
    std::vector<gpu::Buf> bn_a, bn_pa, bn_f, bn_pf, bn_pp;   // per layer, d-sized
    std::vector<gpu::Buf> bn_q, bn_k;                        // per layer, hd-sized (k only when own_kv)
    gpu::Buf out_norm = nullptr, pl_norm = nullptr;
    gpu::Buf cos_t[2] = {nullptr, nullptr}, sin_t[2] = {nullptr, nullptr};   // [0]=swa [1]=global
    gpu::Buf X = nullptr, Z = nullptr, D = nullptr, QKV = nullptr, QS = nullptr, AO = nullptr, G = nullptr;
    gpu::Buf PG = nullptr, PP = nullptr, PL = nullptr, PLj = nullptr, PLz = nullptr;
    gpu::Buf POS = nullptr, S = nullptr, LOG = nullptr, LAST = nullptr;
};

static void gpu_dump(const char* name, gpu::Buf b, size_t nbytes) {   // debug: G4_DUMP_DIR
    static const char* dir = getenv("G4_DUMP_DIR");
    if (!dir || !b) return;
    std::vector<float> h(nbytes / 4);
    if (!gpu::read(b, h.data(), nbytes)) return;
    char fn[1024];
    snprintf(fn, sizeof fn, "%s/g_%s.bin", dir, name);
    FILE* f = fopen(fn, "wb");
    if (!f) return;
    fwrite(h.data(), 4, h.size(), f);
    fclose(f);
}

static void rel(gpu::Buf& b) { if (b) { gpu::release(b); b = nullptr; } }

void Gemma4Model::free_gpu() {
    if (!gpu) return;
    G4Gpu& g = *gpu;
    for (auto& b : g.kv) rel(b);
    for (auto* v : {&g.bn_a, &g.bn_pa, &g.bn_f, &g.bn_pf, &g.bn_pp, &g.bn_q, &g.bn_k})
        for (auto& b : *v) rel(b);
    for (auto* b : {&g.out_norm, &g.pl_norm, &g.cos_t[0], &g.cos_t[1], &g.sin_t[0], &g.sin_t[1],
                    &g.X, &g.Z, &g.D, &g.QKV, &g.QS, &g.AO, &g.G, &g.PG, &g.PP, &g.PL, &g.PLj, &g.PLz,
                    &g.POS, &g.S, &g.LOG, &g.LAST})
        rel(*b);
    delete gpu;
    gpu = nullptr;
}

static int up_vec(std::vector<float>& v) {        // norm weight -> device buffer (idempotent)
    if (v.empty()) return -1;
    gpu::Buf b = gpu::upload(v.data(), v.size() * 4);
    return b ? 0 : -1;
}

static gpu::Buf up_vec_buf(std::vector<float>& v) {
    if (v.empty()) return nullptr;
    return gpu::upload(v.data(), v.size() * 4);
}

bool Gemma4Model::upload_gpu(std::string* err) {
    if (!g4_gpu_ok()) return false;
    auto fail = [&](const char* what) { if (err) *err = std::string("gpu upload: ") + what; free_gpu(); return false; };
    gpu = new G4Gpu();
    G4Gpu& g = *gpu;
    g.ctx = ctx; g.cap = G4_CHUNK;
    g.d = n_embd; g.nq = n_head; g.nkv = n_kv; g.pl = n_embd_pl; g.n_layer = n_layer; g.n_vocab = n_vocab;
    g.qd_max = n_head * 512; g.kv_max = n_kv * 512; g.ff_max = 0;
    for (auto& l : L) g.ff_max = std::max(g.ff_max, l.ffn_gate.rows ? l.ffn_gate.rows : (int)l.ffn_gate.rows);
    if (!g.ff_max) g.ff_max = n_ff;
    for (auto& l : L) g.ff_max = std::max(g.ff_max, (int)L[0].ffn_gate.rows);
    g.rowc_max = g.qd_max + 2 * g.kv_max;

    // ---- KV caches (own-kv layers only)
    g.kv.assign(n_layer, nullptr);
    for (int il = 0; il < n_layer; ++il) {
        const G4Layer& l = L[il];
        if (!l.own_kv) continue;
        const int kd = n_kv * l.hd;
        g.kv[il] = gpu::alloc((size_t)g.ctx * 2 * kd * 4);
        if (!g.kv[il]) return fail("kv alloc");
    }
    // ---- per-layer norm weights
    for (int il = 0; il < n_layer; ++il) {
        G4Layer& l = L[il];
        g.bn_a.push_back(up_vec_buf(l.attn_norm));
        g.bn_pa.push_back(up_vec_buf(l.post_attn_norm));
        g.bn_f.push_back(up_vec_buf(l.ffn_norm));
        g.bn_pf.push_back(up_vec_buf(l.post_ffw_norm));
        g.bn_pp.push_back(up_vec_buf(l.post_norm));
        g.bn_q.push_back(up_vec_buf(l.q_norm));
        g.bn_k.push_back(l.own_kv ? up_vec_buf(l.k_norm) : nullptr);
        if (!g.bn_a[il] || !g.bn_pa[il] || !g.bn_f[il] || !g.bn_pf[il] || !g.bn_pp[il] || !g.bn_q[il]) return fail("norm upload");
    }
    g.out_norm = up_vec_buf(out_norm);
    g.pl_norm = up_vec_buf(pl_proj_norm);
    if (!g.out_norm || !g.pl_norm) return fail("out/ple norm upload");

    // ---- rope tables: [ctx][hlf] cos/sin, one set per layer type
    for (int kind = 0; kind < 2; ++kind) {
        const bool glb = kind == 1;
        const int hd = glb ? (hd_g ? hd_g : 512) : (hd_s ? hd_s : 256);
        const int hlf = hd / 2;
        const float base = glb ? rope_base : rope_base_swa;
        std::vector<float> cb((size_t)g.ctx * hlf), sb((size_t)g.ctx * hlf);
        for (int p = 0; p < g.ctx; ++p)
            for (int i = 0; i < hlf; ++i) {
                float th = powf(base, -2.f * i / hd);
                if (glb && (int)rope_freqs.size() > i) th /= rope_freqs[i];
                const float ang = (float)p * th;
                cb[(size_t)p * hlf + i] = cosf(ang);
                sb[(size_t)p * hlf + i] = sinf(ang);
            }
        g.cos_t[kind] = gpu::upload(cb.data(), cb.size() * 4);
        g.sin_t[kind] = gpu::upload(sb.data(), sb.size() * 4);
        if (!g.cos_t[kind] || !g.sin_t[kind]) return fail("rope table upload");
    }

    // ---- scratch
    const size_t R = (size_t)g.cap;
    g.X   = gpu::alloc(R * n_embd * 4);
    g.Z   = gpu::alloc(R * n_embd * 4);
    g.D   = gpu::alloc(R * n_embd * 4);
    g.QKV = gpu::alloc(R * g.rowc_max * 4);
    g.QS  = gpu::alloc(R * g.qd_max * 4);
    g.AO  = gpu::alloc(R * g.qd_max * 4);
    g.G   = gpu::alloc(R * 2 * g.ff_max * 4);
    g.PG  = gpu::alloc(R * std::max(1, n_embd_pl) * 4);
    g.PP  = gpu::alloc(R * n_embd * 4);
    g.LOG = gpu::alloc((size_t)n_vocab * 4);
    g.LAST = gpu::alloc((size_t)n_embd * 4);
    g.POS = gpu::alloc(R * 4);
    g.S   = gpu::alloc((size_t)n_head * g.cap * g.ctx * 4);
    if (n_embd_pl > 0) {
        const size_t plw = (size_t)n_embd_pl * n_layer;
        g.PL  = gpu::alloc(R * plw * 4);
        g.PLj = gpu::alloc(R * plw * 4);
        g.PLz = gpu::alloc(R * plw * 4);
    }
    std::vector<float> zeros(R * n_embd, 0.f);
    if (!g.X || !g.Z || !g.D || !g.QKV || !g.QS || !g.AO || !g.G || !g.PG || !g.PP || !g.LOG || !g.LAST ||
        !g.POS || !g.S || (n_embd_pl > 0 && (!g.PL || !g.PLj || !g.PLz)))
        return fail("scratch alloc");
    if (!gpu::write(g.Z, zeros.data(), zeros.size() * 4)) return fail("zero fill");
    if (n_embd_pl > 0) {
        std::vector<float> z((size_t)g.cap * n_embd_pl * n_layer, 0.f);
        if (!gpu::write(g.PLz, z.data(), z.size() * 4)) return fail("ple zero fill");
    }
    return true;
}

// one Q8 GEMM from the quantized activation scratch
static inline bool gemm_q(int slot, int T, gpu::Buf Y) { return slot >= 0 && gpu::op_gemm_q(slot, T, Y); }

bool Gemma4Model::forward_gpu(State& s, const int* ids, int T, float* logits, std::string* err) {
    if (!gpu) { if (err) *err = "gpu state missing"; return false; }
    G4Gpu& g = *gpu;
    const int d = n_embd, pl = n_embd_pl;
    if (s.n + T > g.ctx) { if (err) *err = "context overflow"; return false; }


    std::vector<float> xh((size_t)g.cap * d);
    std::vector<float> plh(pl > 0 ? (size_t)g.cap * pl * n_layer : 0);
    std::vector<int> posh(g.cap);
    const float emb_scale = sqrtf((float)d), pl_scale = sqrtf((float)pl), inv_sqrt2 = 1.0f / sqrtf(2.f);

    for (int done = 0; done < T; done += g.cap) {
        const int tc = std::min(T - done, g.cap);
        const int pos0 = s.n + done;
        // ---- host: token embeddings (x sqrt(d)) and this chunk's PLE rows (x sqrt(pl))
        for (int t = 0; t < tc; ++t) {
            const int id = ids[done + t];
            float* xr = xh.data() + (size_t)t * d;
            g4_q4_0_row(gguf, *tok_embd_raw, id, xr);
            for (int i = 0; i < d; ++i) xr[i] *= emb_scale;
            if (pl > 0) {
                float* pr = plh.data() + (size_t)t * pl * n_layer;
                g4_q4_0_row(gguf, *ple_tok_raw, id, pr);
                for (int i = 0; i < pl * n_layer; ++i) pr[i] *= pl_scale;
            }
            posh[t] = pos0 + t;
        }
        // the int8 activation scratch is shared by every producer: size it for the widest one (FFN gate|up)
        if (!gpu::ensure_qscratch(tc, std::max(d, std::max(g.ff_max, pl)))) { if (err) *err = "qscratch"; return false; }
        if (!gpu::write(g.X, xh.data(), (size_t)tc * d * 4)) { if (err) *err = "X upload"; return false; }
        if (!gpu::write(g.POS, posh.data(), (size_t)tc * 4)) { if (err) *err = "pos upload"; return false; }
        if (pl > 0 && !gpu::write(g.PL, plh.data(), (size_t)tc * pl * n_layer * 4)) { if (err) *err = "ple upload"; return false; }

        // ---- per-layer embeddings: ple = (proj@x * 1/sqrt(d) -> rmsnorm(pl) + tok_rows * sqrt(pl)) * 1/sqrt(2)
        if (pl > 0) {
            const size_t pn = (size_t)tc * pl * n_layer;
            if (!gpu::op_gemm(pl_model_proj.gpu_slot, g.X, tc, g.PLj)) { if (err) *err = "ple proj"; return false; }
            if (!gpu::op_add_scale(g.PLj, g.PLz, g.PLj, (int)pn, 1.0f / sqrtf((float)d))) { if (err) *err = "ple scale"; return false; }
            if (!gpu::op_rmsnorm(g.PLj, 0, n_layer * pl, g.PLj, 0, n_layer * pl, tc, n_layer, pl, g.pl_norm, rms_eps))
                { if (err) *err = "ple norm"; return false; }
            if (!gpu::op_add_scale(g.PLj, g.PL, g.PL, (int)pn, inv_sqrt2)) { if (err) *err = "ple combine"; return false; }
            gpu_dump("inp_per_layer", g.PL, pn * 4);
        }

        for (int il = 0; il < n_layer; ++il) {
            const G4Layer& l = L[il];
            const int kd = n_kv * l.hd, qd = n_head * l.hd, rowc = qd + 2 * kd;
            const int kind = l.swa ? 0 : 1;
            const int hlf = l.hd / 2;
            gpu::Buf cosb = g.cos_t[kind], sinb = g.sin_t[kind];
            const float osc = l.out_scale.empty() ? 1.f : l.out_scale[0];

            // ---- attention input norm (rms(X) * attn_norm), quantized for the QKV GEMM
            if (!gpu::op_row_fused(nullptr, g.X, g.bn_a[il], 1.0f, nullptr, nullptr, false, true, tc, d, rms_eps))
                { if (err) *err = "attn_norm"; return false; }
            gpu::Buf Qsrc; int qld;
            if (l.own_kv) {
                if (!gemm_q(l.qkv.gpu_slot, tc, g.QKV)) { if (err) *err = "qkv gemm"; return false; }
                if (!gpu::op_qkv_post(g.QKV, rowc, tc, n_head, n_kv, l.hd, g.bn_q[il], g.bn_k[il], cosb, sinb, g.POS, hlf, rms_eps, true))
                    { if (err) *err = "qkv post"; return false; }
                if (!gpu::op_lm_kv_write(g.QKV, rowc, qd, g.kv[il], g.POS, tc, 2 * kd)) { if (err) *err = "kv write"; return false; }
                Qsrc = g.QKV; qld = rowc;
            } else {
                if (!gemm_q(l.wq.gpu_slot, tc, g.QS)) { if (err) *err = "q gemm"; return false; }
                if (!gpu::op_rmsnorm(g.QS, 0, qd, g.QS, 0, qd, tc, n_head, l.hd, g.bn_q[il], rms_eps)) { if (err) *err = "q norm"; return false; }
                if (!gpu::op_rope(g.QS, 0, qd, tc, n_head, l.hd, hlf, cosb, sinb, g.POS)) { if (err) *err = "q rope"; return false; }
                Qsrc = g.QS; qld = qd;
            }
            const int kvil = l.own_kv ? il : l.kv_src;
            if (!gpu::op_attn_split(Qsrc, qld, g.kv[kvil], 2 * kd, 0, 0, kd, pos0 + tc, pos0, tc,
                                    n_head, n_kv, l.hd, 1.0f, g.S, g.AO, qd, l.swa ? swa_win : -1, 1))
                { if (err) *err = "attention"; return false; }
            { char nm[48]; snprintf(nm, sizeof nm, "attn_q-%d", il); gpu_dump(nm, g.AO, (size_t)tc * qd * 4); }
            // ---- out projection + post-attention norm + residual, then the FFN input norm (quantized)
            { char nm[48]; snprintf(nm, sizeof nm, "q_in-%d", il); gpu_dump(nm, Qsrc, (size_t)tc * qld * 4); }
            if (!gpu::op_gemm(l.wo.gpu_slot, g.AO, tc, g.D)) { if (err) *err = "wo gemm"; return false; }
            { char nm[48]; snprintf(nm, sizeof nm, "attn_wo-%d", il); gpu_dump(nm, g.D, (size_t)tc * d * 4); }
            if (!gpu::op_row_fused(g.X, g.D, g.bn_pa[il], 1.0f, g.X, g.bn_f[il], true, true, tc, d, rms_eps))
                { if (err) *err = "attn post"; return false; }
            { char nm[48]; snprintf(nm, sizeof nm, "attn_out-%d", il); gpu_dump(nm, g.X, (size_t)tc * d * 4); }
            // ---- FFN: gate|up -> gelu(gate)*up -> down -> post norm -> residual (X stays quantized for the PLE gate)
            if (!gemm_q(l.gateup.gpu_slot, tc, g.G)) { if (err) *err = "gateup gemm"; return false; }
            if (!gpu::op_gelu_mul_q(g.G, tc, l.ffn_gate.rows)) { if (err) *err = "gelu mul"; return false; }
            if (!gemm_q(l.ffn_down.gpu_slot, tc, g.D)) { if (err) *err = "down gemm"; return false; }
            { char nm[48]; snprintf(nm, sizeof nm, "ffn_out-%d", il); gpu_dump(nm, g.D, (size_t)tc * d * 4); }
            // with no PLE the layer output ends here, so the out_scale rides along in the same launch
            if (!gpu::op_row_fused(g.X, g.D, g.bn_pf[il], pl > 0 ? 1.0f : osc, g.X, nullptr, false, pl > 0, tc, d, rms_eps))
                { if (err) *err = "ffn post"; return false; }
            { char nm[48]; snprintf(nm, sizeof nm, "pre_ple-%d", il); gpu_dump(nm, g.X, (size_t)tc * d * 4); }
            // ---- per-layer embedding: X += rms(gelu(inp_gate@X) * ple[il]) @ proj * post_norm
            if (pl > 0) {
                if (!gemm_q(l.inp_gate.gpu_slot, tc, g.PG)) { if (err) *err = "ple gate"; return false; }
                if (!gpu::op_gelu_mulvec_q(g.PG, g.PL, il * pl, n_layer * pl, tc, pl)) { if (err) *err = "ple gate mul"; return false; }
                if (!gemm_q(l.proj.gpu_slot, tc, g.PP)) { if (err) *err = "ple proj down"; return false; }
                if (!gpu::op_row_fused(g.X, g.PP, g.bn_pp[il], osc, g.X, nullptr, false, false, tc, d, rms_eps))
                    { if (err) *err = "ple post"; return false; }
            }
            if (pl == 0 && osc != 1.0f && !gpu::op_add_scale(g.X, g.Z, g.X, tc * d, osc)) { if (err) *err = "out scale"; return false; }
            { char nm[48]; snprintf(nm, sizeof nm, "l_out-%d", il); gpu_dump(nm, g.X, (size_t)tc * d * 4); }
        }
        // ---- final norm + head, only for the last chunk
        if (done + tc == T && logits) {
            if (!gpu::copy(g.LAST, 0, g.X, (size_t)(tc - 1) * d * 4, (size_t)d * 4)) { if (err) *err = "copy last"; return false; }
            if (!gpu::op_row_fused(nullptr, g.LAST, g.out_norm, 1.0f, nullptr, nullptr, false, true, 1, d, rms_eps))
                { if (err) *err = "out norm"; return false; }
            if (!gemm_q(tok_embd.gpu_slot, 1, g.LOG)) { if (err) *err = "head gemm"; return false; }
            if (!gpu::read(g.LOG, logits, (size_t)n_vocab * 4)) { if (err) *err = "logits read"; return false; }
            if (softcap > 0.f)                                   // the only host-side math on the output
                for (int i = 0; i < n_vocab; ++i) logits[i] = softcap * tanhf(logits[i] / softcap);
        }
    }
    s.n += T;
    return true;
}

}  // namespace ferro
