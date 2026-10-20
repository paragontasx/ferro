// mm_gemma4.cpp — see mm_gemma4.h. CPU reference forward (GEMMs route to the iGPU when weights are resident).
#include "mm_gemma4.h"
#include "mm_util.h"
#include "mm_clin.h"
#include "mm_gemma4_impl.h"
#include "threadpool.h"
#include "gpu_cl.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ferro {

void load_audio_weights(const GgufFile& g, Gemma4Proj::Impl& I);   // mm_gemma4a.cpp

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
                G4VLayer& L = I.vl[i];
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
        if (g_.get_bool("clip.has_audio_encoder") && g_.get_str("clip.audio.projector_type") == "gemma4a") {
            load_audio_weights(g_, I);                          // mm_gemma4a.cpp
            a_ok_ = true;
            if (!n_out_) n_out_ = I.a_mm_proj.w.rows;
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


// FERRO_G4_DUMP=dir: write the residual stream X [T][d] after the patch embed and after every layer (CPU and iGPU paths
// use the same file names, prefixed cpu_/gpu_) so the two can be compared layer by layer.   G4_DUMP
static void g4_dump(const char* pfx, int layer, const float* x, size_t n) {
    const char* dir = getenv("FERRO_G4_DUMP");
    if (!dir) return;
    char path[512]; snprintf(path, sizeof path, "%s/%s_L%02d.bin", dir, pfx, layer);
    if (FILE* f = fopen(path, "wb")) { fwrite(x, 4, n, f); fclose(f); }
}

// ------------------------------------------------------------------ iGPU
// The whole 16-layer tower on the device (int8 DP4A GEMMs, fused [q|k|v] and [gate|up] weights). Layer:
//   X += post_attn_norm(o(attn(rope2d(qknorm(qkv(ln1(X))))))) ; X += post_ffn_norm(down(gelu(gate) * up) (ln2(X)))
// expressed with row_fused (residual + rms(branch)*w -> next rms-norm -> int8) so every GEMM reads int8 straight
// from the producer. Vision linears carry no clamps (all +-FLT_MAX), so nothing extra is needed on the device.
static gpu::Buf up_vec(const std::vector<float>& v) { return gpu::upload(v.data(), v.size() * 4); }

void Gemma4Proj::upload_gpu() {
    if (!impl_ || (!v_ok_ && !a_ok_)) return;
    Impl& I = *impl_;
    const char* env = getenv("FERRO_MM_GPU");
    if (env && atoi(env) == 0) return;
    if (!gpu::available()) return;
    if (a_ok_) {                                  // audio: every Q8 linear resident on the device; apply() routes the GEMMs there
        const char* ae = getenv("FERRO_MM_AUDIO_GPU");
        if (!ae || atoi(ae) != 0) {
            int up = 0, tot = 0;
            auto slot = [&](CLin& c) { ++tot; if (!c.w.q.empty() && c.w.q.gpu_slot < 0) { c.w.q.gpu_slot = gpu::upload_q8(c.w.q); } up += c.w.q.gpu_slot >= 0; };
            for (auto& L : I.al) for (CLin* c : {&L.ffn_up, &L.ffn_down, &L.ffn_up1, &L.ffn_down1, &L.q, &L.k, &L.v, &L.o, &L.k_rel, &L.pw1, &L.pw2}) slot(*c);
            slot(I.a_out_proj); slot(I.a_mm_proj);
            if (getenv("FERRO_VERBOSE")) fprintf(stderr, "[g4a] %d/%d audio linears resident on the iGPU\n", up, tot);
        }
    }
    if (!v_ok_) return;
    bool ok = true;
    auto slot = [&](WMat& W) { if (W.gpu_slot < 0) W.gpu_slot = gpu::upload_q8(W); ok = ok && W.gpu_slot >= 0; };
    for (auto& L : I.vl) if (L.q.w.q.empty() || L.gate.w.q.empty()) ok = false;       // f32 weights: not supported on-device
    if (ok && I.mm_proj.w.q.empty()) ok = false;
    if (!ok) { if (getenv("FERRO_VERBOSE")) fprintf(stderr, "[g4] vision weights are not Q8_0; staying on CPU\n"); return; }
    const int d = I.d, pd = 3 * I.ps * I.ps;
    I.patch_q = pack_from_f32(I.patch_w.data(), d, pd, pd, "g4.patch_embd");
    slot(I.patch_q);
    I.gl.assign(I.vl.size(), Impl::GL());
    for (size_t i = 0; ok && i < I.vl.size(); ++i) {
        G4VLayer& L = I.vl[i]; auto& g = I.gl[i];
        g.qkv = pack_concat({&L.q.w.q, &L.k.w.q, &L.v.w.q}, "g4.qkv");
        g.gu = pack_concat({&L.gate.w.q, &L.up.w.q}, "g4.gate_up");
        slot(g.qkv); slot(g.gu); slot(L.o.w.q); slot(L.down.w.q);
        g.ln1 = up_vec(L.ln1); g.ln2 = up_vec(L.ln2); g.qn = up_vec(L.qn); g.kn = up_vec(L.kn);
        g.pa = up_vec(L.post_attn); g.pf = up_vec(L.post_ffn);
        ok = ok && g.ln1 && g.ln2 && g.qn && g.kn && g.pa && g.pf;
    }
    slot(I.mm_proj.w.q);
    const int hn = I.hd / 4;
    std::vector<float> invf(hn);
    for (int i = 0; i < hn; ++i) invf[i] = std::pow(I.theta, -(float)i / (float)hn);
    I.invf = up_vec(invf);
    std::vector<float> ones(I.d, 1.0f);
    I.ones = up_vec(ones);                                  // weight of the weightless RMSNorm (pooler)
    ok = ok && I.invf && I.ones;
    I.gpu_ok = ok;
    if (getenv("FERRO_VERBOSE")) fprintf(stderr, "[g4] vision tower on %s\n", ok ? "iGPU" : "CPU");
}

bool Gemma4Proj::vision_on_gpu() const { return impl_ && impl_->gpu_ok; }

// returns false -> caller runs the CPU path
static bool encode_image_gpu(Gemma4Proj::Impl& I, const VisionInput& in, std::vector<float>& out) {
    const int wp = in.nx / I.ps, hp = in.ny / I.ps, T = wp * hp, d = I.d, H = I.heads, hd = I.hd, ffn = I.ffn;
    const int pd = 3 * I.ps * I.ps;
    static const bool vb = getenv("FERRO_VERBOSE") != nullptr;
    auto bail = [&](const char* why) { if (vb) fprintf(stderr, "[g4] iGPU vision bail: %s (T=%d)\n", why, T); return false; };
    if (hd != 64 || d % 32 || ffn % 32) return bail("shape");
    const bool dense = !(getenv("FERRO_MM_DENSE") && atoi(getenv("FERRO_MM_DENSE")) == 0);   // dense kernels handle ragged T
    const int ow = wp / I.merge, oh = hp / I.merge, Tm = ow * oh, k2 = I.merge * I.merge;
    const size_t t = (size_t)T;
    if (t > I.cap) {
        for (gpu::Buf* b : {&I.bX, &I.bC, &I.bD, &I.bG, &I.bT, &I.pos_x, &I.pos_y, &I.blo, &I.bhi}) { gpu::release(*b); *b = nullptr; }
        I.bX = gpu::alloc(t * d * 4);
        I.bC = gpu::alloc(t * std::max(3 * d, pd) * 4);
        I.bD = gpu::alloc(t * d * 4);
        I.bG = gpu::alloc(t * 2 * ffn * 4);
        I.bT = gpu::alloc(t * std::max(I.proj, d) * 4);
        I.pos_x = gpu::alloc(t * 4); I.pos_y = gpu::alloc(t * 4); I.blo = gpu::alloc(t * 4); I.bhi = gpu::alloc(t * 4);
        for (gpu::Buf b : {I.bX, I.bC, I.bD, I.bG, I.bT, I.pos_x, I.pos_y, I.blo, I.bhi}) if (!b) { I.cap = 0; return bail("alloc"); }
        I.cap = t;
    }
    // host-side inputs: patches [T][pd] (planar c,y,x, pixels 2x-1), per-token positions, learned x/y position rows
    std::vector<float> P(t * pd), posrow(t * d);
    std::vector<int> px(T), py(T);
    Pool::get().parallel_for(T, [&](int lo, int hi) {
        for (int tt = lo; tt < hi; ++tt) {
            const int x = tt % wp, y = tt / wp;
            px[tt] = x; py[tt] = y;
            float* o = P.data() + (size_t)tt * pd;
            for (int c = 0; c < 3; ++c)
                for (int yy = 0; yy < I.ps; ++yy) {
                    const float* src = in.chw.data() + ((size_t)c * in.ny + (size_t)y * I.ps + yy) * in.nx + (size_t)x * I.ps;
                    for (int xx = 0; xx < I.ps; ++xx) *o++ = src[xx] * 2.0f - 1.0f;
                }
            const float* ex = I.pos_tbl.data() + (size_t)x * d;
            const float* ey = I.pos_tbl.data() + ((size_t)I.pos_size + y) * d;
            float* pr = posrow.data() + (size_t)tt * d;
            for (int i = 0; i < d; ++i) pr[i] = ex[i] + ey[i];
        }
    });
    // pooler gather map: output token o <- the merge x merge patch block (host builds it once per size)
    if ((size_t)Tm * k2 > I.capm) { gpu::release(I.bsrc); I.capm = (size_t)Tm * k2; I.bsrc = gpu::alloc(I.capm * 4); if (!I.bsrc) { I.capm = 0; return false; } }
    if (!gpu::ensure_qscratch(T, std::max({pd, 3 * d, 2 * ffn, ffn, d}))) return bail("qscratch");
    const float eps = I.eps;
    // one fully visible segment: lo = 0, hi = T-1 for every token. gpu::write() is a non-blocking enqueue whose source
    // must stay valid until the next read/finish, so these live for the whole call (not in an inner block).
    std::vector<int> seg_lo(dense ? 0 : T, 0), seg_hi(dense ? 0 : T, T - 1);
    if (!dense && (!gpu::write(I.blo, seg_lo.data(), (size_t)T * 4) || !gpu::write(I.bhi, seg_hi.data(), (size_t)T * 4)))
        return bail("seg bounds");
    if (dense && (size_t)H * t * t > I.capS) {                        // dense attention scores S[h][i][j]
        gpu::release(I.bS); I.capS = (size_t)H * t * t; I.bS = gpu::alloc(I.capS * 4);
        if (!I.bS) { I.capS = 0; return bail("score scratch alloc"); }
    }
    gpu::Buf X = I.bX, C = I.bC, D = I.bD, G = I.bG;
    bool ok = true; int pre = 0;
    auto pstep = [&](int id, bool r) { if (ok && !r) { ok = false; pre = id; } return r; };
    pstep(1, gpu::write(C, P.data(), P.size() * 4));
    pstep(2, gpu::write(X, posrow.data(), posrow.size() * 4));
    pstep(3, gpu::write(I.pos_x, px.data(), (size_t)T * 4) && gpu::write(I.pos_y, py.data(), (size_t)T * 4));
    // patch embedding GEMM (int8) -> D, then X = D + learned positions
    pstep(4, gpu::op_quant_q(C, T, pd));
    pstep(5, gpu::op_gemm_q(I.patch_q.gpu_slot, T, D));
    pstep(6, gpu::op_add_scale(X, D, X, T * d, 1.0f));
    // pre-attn rms-norm of layer 0 -> int8
    pstep(7, gpu::op_row_fused(X, nullptr, nullptr, 1.0f, nullptr, I.gl[0].ln1, true, true, T, d, eps));
    if (!ok && vb) fprintf(stderr, "[g4] pre-loop failed at step %d (1 write patches,2 write posrow,3 write pos,4 quant,5 patch gemm,6 add pos,7 first norm) T=%d pd=%d\n", pre, T, pd);
    const int rowc = 3 * d;
    int stage = -1;
    // FERRO_G4_SYNC=1: finish() after every step so a device-side fault is pinned to the exact op (slow; debugging)
    static const bool sync_dbg = getenv("FERRO_G4_SYNC") != nullptr;
    double tstep[10] = {0}; auto tlast = std::chrono::steady_clock::now();
    auto step = [&](int id, bool r) {
        if (ok && sync_dbg) {
            r = r && gpu::finish();
            const auto tn = std::chrono::steady_clock::now();
            tstep[id] += std::chrono::duration<double, std::milli>(tn - tlast).count(); tlast = tn;
        }
        if (ok && !r) { ok = false; stage = id; }
        return r; };
    for (int l = 0; ok && l < I.layers; ++l) {
        const G4VLayer& L = I.vl[l]; const auto& g = I.gl[l];
        step(1, gpu::op_gemm_q(g.qkv.gpu_slot, T, C));
        step(2, gpu::op_qkv_post_g4(C, rowc, T, H, H, hd, g.qn, g.kn, I.invf, I.pos_x, I.pos_y, eps));
        step(3, dense ? (gpu::op_dense_attention(C, rowc, 0, d, 2 * d, 0, T, H, 1.0f, I.bS, D, d) && gpu::op_quant_q(D, T, d))
                      : gpu::op_attention_q(C, rowc, 0, d, 2 * d, T, H, H, hd, I.blo, I.bhi, 0, 1.0f));
        step(4, gpu::op_gemm_q(L.o.w.q.gpu_slot, T, G));
        step(5, gpu::op_row_fused(X, G, g.pa, 1.0f, X, g.ln2, true, true, T, d, eps));
        step(6, gpu::op_gemm_q(g.gu.gpu_slot, T, G));
        step(7, gpu::op_gelu_mul_q(G, T, ffn));
        step(8, gpu::op_gemm_q(L.down.w.q.gpu_slot, T, D));
        step(9, l + 1 < I.layers ? gpu::op_row_fused(X, D, g.pf, 1.0f, X, I.gl[l + 1].ln1, true, true, T, d, eps)
                                 : gpu::op_row_fused(X, D, g.pf, 1.0f, X, nullptr, false, false, T, d, eps));
        if (ok && !gpu::finish()) { ok = false; stage = 0; }
        if (ok && getenv("FERRO_G4_DUMP")) { std::vector<float> xd(t * d); if (gpu::read(X, xd.data(), xd.size() * 4)) g4_dump("gpu", l, xd.data(), xd.size()); }
        if (!ok && vb) fprintf(stderr, "[g4] layer %d failed at step %d (1 qkv gemm,2 qkv_post,3 attention,4 o gemm,5 resid+ln2,6 gate|up gemm,7 gelu_mul,8 down gemm,9 resid)\n", l, stage);
    }
    if (ok && sync_dbg)
        fprintf(stderr, "[g4-time] T=%d  qkv gemm %.0f  qkv_post %.0f  attention %.0f  o gemm %.0f  resid+ln2 %.0f  gate|up gemm %.0f  gelu_mul %.0f  down gemm %.0f  resid %.0f ms (sum over 16 layers)\n",
                T, tstep[1], tstep[2], tstep[3], tstep[4], tstep[5], tstep[6], tstep[7], tstep[8], tstep[9]);
    // tail: X back to the host, 3x3 average pool x sqrt(d), weightless RMSNorm (CPU: small, memory-bound), then the
    // 768 -> proj GEMM on the device.   GPU_TAIL_DONE
    std::vector<float> Xh(t * d);
    ok = ok && gpu::read(X, Xh.data(), Xh.size() * 4);
    if (!ok) return bail("layer loop / readback");
    std::vector<float> pooled((size_t)Tm * d);
    const float sc = std::sqrt((float)d);
    Pool::get().parallel_for(Tm, [&](int lo, int hi) {
        for (int o = lo; o < hi; ++o) {
            const int ox = o % ow, oy = o / ow;
            float* p = pooled.data() + (size_t)o * d;
            for (int i = 0; i < d; ++i) p[i] = 0;
            for (int dy = 0; dy < I.merge; ++dy)
                for (int dx = 0; dx < I.merge; ++dx) {
                    const float* x = Xh.data() + ((size_t)(oy * I.merge + dy) * wp + ox * I.merge + dx) * d;
                    for (int i = 0; i < d; ++i) p[i] += x[i];
                }
            for (int i = 0; i < d; ++i) p[i] = p[i] / (float)k2 * sc;
        }
    });
    rmsnorm_w1(pooled.data(), Tm, d, eps, pooled.data());
    if (!gpu::ensure_qscratch(Tm, d)) return false;
    ok = gpu::write(I.bT, pooled.data(), pooled.size() * 4) && gpu::op_quant_q(I.bT, Tm, d) &&
         gpu::op_gemm_q(I.mm_proj.w.q.gpu_slot, Tm, D);
    out.resize((size_t)Tm * I.proj);
    ok = ok && gpu::read(D, out.data(), out.size() * 4);
    return ok;
}

std::vector<float> Gemma4Proj::encode_image(const VisionInput& in) {
    if (!v_ok_) throw std::runtime_error("mmproj has no gemma4v vision encoder");
    Impl& I = *impl_;
    if (I.gpu_ok) {
        std::vector<float> o;
        if (encode_image_gpu(I, in, o)) return o;
        static bool warned = false;
        if (!warned && getenv("FERRO_VERBOSE")) { warned = true; fprintf(stderr, "[g4] iGPU vision encode failed; using the CPU path\n"); }
    }
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
        const G4VLayer& L = I.vl[l];
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
        g4_dump("cpu", l, X.data(), (size_t)T * d);
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
