// ggml_ops_ref.cpp - per-op oracle for the Gemma 4 audio encoder (clip_graph_gemma4a): runs the layout-sensitive ggml
// ops on random data and dumps inputs + outputs, so the CPU reference can be checked op by op without model taps.
// Usage: ggml_ops_ref OUT_DIR       writes <name>.in*.bin / <name>.out.bin (raw f32, ggml memory order)
// Ops (each matches the sequence in tools/mtmd/models/gemma4a.cpp):
//   sscp   : ggml_conv_2d(w, x, 2,2,1,1,1,1) + LayerNorm over channels + ReLU        (x: [freq, time, ch])
//   shift  : blocked Transformer-XL relative shift (pad / reshape / view)
//   blocks : K/V block-context extraction by an overlapping view (left pad P, chunk C, context S)
//   ssm    : causal depthwise conv via ggml_ssm_conv with left padding 4
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static void dump(const std::string& dir, const std::string& name, const ggml_tensor* t) {
    std::string p = dir + "/" + name + ".bin";
    FILE* f = fopen(p.c_str(), "wb");
    // CPU-only context (no_alloc = false): tensor data lives in the context, there is no backend buffer to read through
    std::vector<float> buf(ggml_nelements(t));
    memcpy(buf.data(), t->data, buf.size() * 4);
    // write ne[0..3] first so the python side knows the shape in ggml order
    int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    fwrite(ne, 8, 4, f); fwrite(buf.data(), 4, buf.size(), f); fclose(f);
}
static std::vector<float> rnd(size_t n, unsigned seed, float scale = 1.0f) {
    std::mt19937 g(seed); std::normal_distribution<float> d(0.f, scale);
    std::vector<float> v(n); for (auto& x : v) x = d(g); return v;
}
static ggml_tensor* in(ggml_context* c, std::vector<float>& host, int64_t a, int64_t b = 1, int64_t cc = 1, int64_t dd = 1) {
    ggml_tensor* t = ggml_new_tensor_4d(c, GGML_TYPE_F32, a, b, cc, dd);
    memcpy(t->data, host.data(), host.size() * 4); return t;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ggml_ops_ref OUT_DIR\n"); return 1; }
    const std::string dir = argv[1];
    ggml_init_params ip = {256u << 20, nullptr, false};
    ggml_context* c = ggml_init(ip);
    // ---- sscp: x [freq=128, time=40], two convs (1->128, 128->32) like the mmproj [3 3 1 128] and [3 3 128 32]
    {
        const int F = 128, Tm = 40;
        auto hx = rnd((size_t)F * Tm, 1), hw0 = rnd((size_t)3 * 3 * 1 * 128, 2, 0.3f), hw1 = rnd((size_t)3 * 3 * 128 * 32, 3, 0.05f);
        auto hn0 = rnd(128, 4, 0.2f), hn1 = rnd(32, 5, 0.2f);
        for (auto& v : hn0) v += 1.f; for (auto& v : hn1) v += 1.f;
        // build(): inp = build_inp_raw(1) is [n_frames, n_mel]; cur = cont(transpose(inp)) -> ne0 = mel (fast), ne1 = time.
        // Host layout of the mel is data[mel * n_len + frame] = [mel][frame] row-major = ggml ne0 = frame, ne1 = mel, so the
        // transposed tensor has ne0 = mel, ne1 = frame: x[frame * F + mel].
        ggml_tensor* x = in(c, hx, F, Tm, 1);
        ggml_tensor* w0 = in(c, hw0, 3, 3, 1, 128), *w1 = in(c, hw1, 3, 3, 128, 32);
        ggml_tensor* n0 = in(c, hn0, 128), *n1 = in(c, hn1, 32);
        ggml_tensor* cur = x;
        ggml_tensor* ws[2] = {w0, w1}; ggml_tensor* ns[2] = {n0, n1};
        for (int i = 0; i < 2; ++i) {
            cur = ggml_conv_2d(c, ws[i], cur, 2, 2, 1, 1, 1, 1);
            cur = ggml_cont(c, ggml_permute(c, cur, 1, 2, 0, 3));
            cur = ggml_norm(c, cur, 1e-6f);
            cur = ggml_mul(c, cur, ns[i]);
            cur = ggml_cont(c, ggml_permute(c, cur, 2, 0, 1, 3));
            cur = ggml_relu(c, cur);
        }
        cur = ggml_cont(c, ggml_permute(c, cur, 1, 2, 0, 3));
        cur = ggml_reshape_2d(c, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
        {   // stage taps for layer 0: conv | conv + norm*w (permuted back) | + relu
            ggml_tensor* t0 = ggml_conv_2d(c, w0, x, 2, 2, 1, 1, 1, 1);                       // [F/2, T/2, 128]
            ggml_tensor* t1 = ggml_cont(c, ggml_permute(c, t0, 1, 2, 0, 3));                  // [128, F/2, T/2]
            ggml_tensor* t2 = ggml_norm(c, t1, 1e-6f);
            ggml_tensor* t3 = ggml_mul(c, t2, n0);
            ggml_tensor* t4 = ggml_cont(c, ggml_permute(c, t3, 2, 0, 1, 3));                  // back to [F/2, T/2, 128]
            ggml_tensor* t5 = ggml_relu(c, t4);
            ggml_cgraph* g2 = ggml_new_graph(c);
            for (ggml_tensor* t : {t0, t2, t3, t5}) ggml_build_forward_expand(g2, t);
            ggml_graph_compute_with_ctx(c, g2, 2);
            dump(dir, "sscp.s0_conv", t0); dump(dir, "sscp.s0_norm", t2); dump(dir, "sscp.s0_normw", t3); dump(dir, "sscp.s0_relu", t5);
        }
        ggml_cgraph* gf = ggml_new_graph(c); ggml_build_forward_expand(gf, cur);
        ggml_graph_compute_with_ctx(c, gf, 2);
        dump(dir, "sscp.in", x); dump(dir, "sscp.w0", w0); dump(dir, "sscp.w1", w1); dump(dir, "sscp.n0", n0); dump(dir, "sscp.n1", n1);
        dump(dir, "sscp.out", cur);
        const long long exp0 = 32LL * (((F - 1) / 2 + 1 - 1) / 2 + 1), exp1 = ((Tm - 1) / 2 + 1 - 1) / 2 + 1;
        fprintf(stderr, "sscp: out ne = [%lld %lld]  expected [%lld %lld]  %s\n", (long long)cur->ne[0], (long long)cur->ne[1], exp0, exp1, (cur->ne[0] == exp0 && cur->ne[1] == exp1) ? "OK" : "SHAPE MISMATCH");
    }

    // ---- chunked local attention geometry (gemma4a.cpp): C = chunk 12, P = max past 12, S = C + P context, R = P + 1 rel positions
    {
        const int D = 16, H = 2, NPOS = 30;                    // small d_head / heads keep the dump tiny; the geometry is what matters
        const int64_t Cc = 12, Pp = 12, Ss = Cc + Pp, Rr = Pp + 1, B = (NPOS + Cc - 1) / Cc, Np = B * Cc, pad_seq = Np - NPOS;
        auto hk = rnd((size_t)D * H * NPOS, 11);
        ggml_tensor* K = in(c, hk, D, H, NPOS);                // [D, H, N]
        // extract_blocks: left pad by P, right pad to S*B, overlapping view with stride C positions, cont
        const int64_t pad_kv = Ss * B - NPOS;
        ggml_tensor* t = ggml_pad_ext(c, K, 0, 0, 0, 0, Pp, pad_kv - Pp, 0, 0);                 // [D, H, S*B]
        t = ggml_view_4d(c, t, D, H, Ss, B, t->nb[1], t->nb[2], Cc * t->nb[2], 0);              // [D, H, S, B]
        t = ggml_cont(c, t);
        ggml_tensor* Kb = ggml_cont(c, ggml_permute(c, t, 0, 3, 1, 2));                        // [D, S, B, H]
        ggml_cgraph* g = ggml_new_graph(c); ggml_build_forward_expand(g, Kb);
        ggml_graph_compute_with_ctx(c, g, 2);
        dump(dir, "blk.k", K); dump(dir, "blk.out", Kb);
        fprintf(stderr, "blocks: out ne = [%lld %lld %lld %lld]  expected [%d %lld %lld %d]  %s\n", (long long)Kb->ne[0], (long long)Kb->ne[1],
                (long long)Kb->ne[2], (long long)Kb->ne[3], D, (long long)Ss, (long long)B, H,
                (Kb->ne[0] == D && Kb->ne[1] == Ss && Kb->ne[2] == B && Kb->ne[3] == H) ? "OK" : "SHAPE MISMATCH");
    }
    // ---- blocked Transformer-XL relative shift (matrix_bd [R, C, B, H] -> [S, C, B, H])
    {
        const int H = 2; const int64_t Cc = 12, Pp = 12, Ss = Cc + Pp, Rr = Pp + 1, B = 3;
        auto hb = rnd((size_t)Rr * Cc * B * H, 12);
        ggml_tensor* bd = in(c, hb, Rr, Cc, B, H);
        ggml_tensor* m = ggml_pad(c, bd, Ss + 1 - Rr, 0, 0, 0);                                 // [S+1, C, B, H]
        m = ggml_reshape_3d(c, m, (Ss + 1) * Cc, B, H);
        m = ggml_view_3d(c, m, Cc * Ss, B, H, m->nb[1], m->nb[2], 0);
        m = ggml_cont(c, m);                                                                    // [C*S, B, H]
        m = ggml_reshape_4d(c, m, Ss, Cc, B, H);                                                // [S, C, B, H]
        ggml_cgraph* g = ggml_new_graph(c); ggml_build_forward_expand(g, m);
        ggml_graph_compute_with_ctx(c, g, 2);
        dump(dir, "shift.in", bd); dump(dir, "shift.out", m);
        fprintf(stderr, "shift: out ne = [%lld %lld %lld %lld]  expected [%lld %lld %lld %d]  %s\n", (long long)m->ne[0], (long long)m->ne[1],
                (long long)m->ne[2], (long long)m->ne[3], (long long)Ss, (long long)Cc, (long long)B, H,
                (m->ne[0] == Ss && m->ne[1] == Cc && m->ne[2] == B && m->ne[3] == H) ? "OK" : "SHAPE MISMATCH");
    }
    // ---- causal depthwise conv: GLU output transposed to [n_pos(ne0), ch(ne1)], left pad 4, ggml_ssm_conv with a [5, ch] kernel
    {
        const int CH = 24, NPOS = 20, KS = 5;
        auto hx = rnd((size_t)NPOS * CH, 13), hw = rnd((size_t)KS * CH, 14, 0.4f);
        ggml_tensor* x = in(c, hx, NPOS, CH);                                                   // after the GLU transpose: ne0 = time
        ggml_tensor* w = in(c, hw, KS, CH);
        ggml_tensor* p = ggml_pad_ext(c, x, KS - 1, 0, 0, 0, 0, 0, 0, 0);                       // left pad only
        ggml_tensor* o = ggml_ssm_conv(c, p, w);                                                // -> [CH, NPOS]
        ggml_cgraph* g = ggml_new_graph(c); ggml_build_forward_expand(g, o);
        ggml_graph_compute_with_ctx(c, g, 2);
        dump(dir, "ssm.x", x); dump(dir, "ssm.w", w); dump(dir, "ssm.out", o);
        fprintf(stderr, "ssm: out ne = [%lld %lld]  expected [%d %d]  %s\n", (long long)o->ne[0], (long long)o->ne[1], CH, NPOS,
                (o->ne[0] == CH && o->ne[1] == NPOS) ? "OK" : "SHAPE MISMATCH");
    }
    ggml_free(c);
    return 0;
}
