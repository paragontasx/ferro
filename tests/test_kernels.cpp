// test_kernels.cpp — kernel correctness: fast paths vs scalar reference
#include "kernels.h"
#include "threadpool.h"
#include "gpu_cl.h"
#include "sampler.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <random>
#include <unistd.h>
#include <vector>
using namespace ferro;

static std::mt19937 rng(42);
static float frand() { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); }

static double maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}
static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += std::fabs(a[i]-b[i]); den += std::fabs(b[i]) + 1e-9; }
    return num / den;
}

// LFM2 decode attention (iGPU): split-K kernels vs the per-head kernels vs a double reference on the same K/V, exact invariance, device timing
static void quant_rows_q8(const std::vector<float>& x, std::vector<int8_t>& q, std::vector<float>& s) {   // int8 + one scale per 32 (the KV cache format)
    q.resize(x.size()); s.resize(x.size() / 32);
    for (size_t k = 0; k < s.size(); ++k) {
        float am = 0.f; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[k * 32 + i]));
        const float d = am / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        s[k] = d;
        for (int i = 0; i < 32; ++i) q[k * 32 + i] = (int8_t)std::lrint(std::max(-127.f, std::min(127.f, x[k * 32 + i] * id)));
    }
}
static int test_dec_attn() {
    if (!gpu::dec_split_available()) { printf("gpu decode attention split-K: FAIL (kernels did not build)\n"); return 1; }
    int fails = 0;
    std::mt19937 rg(7); std::normal_distribution<float> nd(0.f, 1.f);
    const float scale = 0.125f;
    const int nkv = 8, kvw = 2 * nkv * 64;
    // key counts around the 128-key slice boundaries; the last three rows are one sequence verified as a speculative window
    const std::vector<int> lens = {1, 127, 128, 129, 300, 1921, 129, 130, 131};
    const std::vector<int> base = {0, 1, 128, 256, 385, 685, 2606, 2606, 2606};
    const int B = (int)lens.size(), rows = 2606 + 131, maxlen = 1921;
    std::vector<float> kv((size_t)rows * kvw);
    for (auto& v : kv) v = nd(rg);
    std::vector<int8_t> kq; std::vector<float> ks, kvd(kv.size());
    quant_rows_q8(kv, kq, ks);
    for (size_t i = 0; i < kv.size(); ++i) kvd[i] = kq[i] * ks[i / 32];                    // what the int8 kernels see
    gpu::Buf dKV = gpu::upload(kv.data(), kv.size() * 4), dKQ = gpu::upload(kq.data(), kq.size()), dKS = gpu::upload(ks.data(), ks.size() * 4);
    gpu::Buf dB = gpu::upload(base.data(), B * 4), dL = gpu::upload(lens.data(), B * 4);
    for (int G : {2, 4}) {
        const int nq = nkv * G, cols = nq * 64, ldc = cols + kvw;
        std::vector<float> Cq((size_t)B * ldc);
        for (auto& v : Cq) v = nd(rg);
        gpu::Buf dC = gpu::upload(Cq.data(), Cq.size() * 4);
        gpu::ensure_qscratch(B, cols);
        for (int q8 = 0; q8 < 2; ++q8) {
            const std::vector<float>& K = q8 ? kvd : kv;
            std::vector<int8_t> xo, xn, xt; std::vector<float> so, sn, st;
            auto run = [&](gpu::Buf c, gpu::Buf bb, gpu::Buf ll, int nb, int ml, std::vector<int8_t>& xp, std::vector<float>& xs) {
                const bool ok = q8 ? gpu::op_lm_attn_dec_q(c, ldc, dKQ, dKS, kvw, nkv * 64, bb, ll, nq, nkv, scale, nb, ml)
                                   : gpu::op_lm_attn_dec(c, ldc, dKV, kvw, nkv * 64, bb, ll, nq, nkv, scale, nb, ml);
                xp.assign((size_t)nb * cols, 0); xs.assign((size_t)nb * cols / 32, 0.f);
                return ok && gpu::read_qscratch(xp.data(), xp.size(), xs.data(), xs.size() * 4);
            };
            bool ok = run(dC, dB, dL, B, 0, xo, so) && run(dC, dB, dL, B, maxlen, xn, sn);
            // exact invariance of the split path: launch stride (slice count), each row alone, reversed batch order
            bool inv = ok && run(dC, dB, dL, B, 4096, xt, st) && xt == xn && st == sn;
            for (int b = 0; inv && b < B; ++b) {
                gpu::Buf c1 = gpu::upload(Cq.data() + (size_t)b * ldc, (size_t)ldc * 4), b1 = gpu::upload(&base[b], 4), l1 = gpu::upload(&lens[b], 4);
                inv = run(c1, b1, l1, 1, lens[b], xt, st) && std::equal(xt.begin(), xt.end(), xn.begin() + (size_t)b * cols)
                      && std::equal(st.begin(), st.end(), sn.begin() + (size_t)b * cols / 32);
                gpu::release(c1); gpu::release(b1); gpu::release(l1);
            }
            if (inv) {
                std::vector<float> Cr((size_t)B * ldc); std::vector<int> br(B), lr(B);
                for (int b = 0; b < B; ++b) {
                    std::copy(Cq.begin() + (size_t)(B - 1 - b) * ldc, Cq.begin() + (size_t)(B - b) * ldc, Cr.begin() + (size_t)b * ldc);
                    br[b] = base[B - 1 - b]; lr[b] = lens[B - 1 - b];
                }
                gpu::Buf cr = gpu::upload(Cr.data(), Cr.size() * 4), bR = gpu::upload(br.data(), B * 4), lR = gpu::upload(lr.data(), B * 4);
                inv = run(cr, bR, lR, B, maxlen, xt, st);
                for (int b = 0; inv && b < B; ++b)
                    inv = std::equal(xt.begin() + (size_t)b * cols, xt.begin() + (size_t)(b + 1) * cols, xn.begin() + (size_t)(B - 1 - b) * cols)
                          && std::equal(st.begin() + (size_t)b * cols / 32, st.begin() + (size_t)(b + 1) * cols / 32, sn.begin() + (size_t)(B - 1 - b) * cols / 32);
                gpu::release(cr); gpu::release(bR); gpu::release(lR);
            }
            // accuracy: worst error in units of the reference block's int8 step (rounding alone gives 0.5)
            double wn = 0, wo = 0;
            for (int b = 0; ok && b < B; ++b) for (int h = 0; h < nq; ++h) {
                const int kvh = h / G; const float* q = &Cq[(size_t)b * ldc + h * 64];
                std::vector<double> p(lens[b]); double mx = -1e300, l = 0, ov[64];
                for (int j = 0; j < lens[b]; ++j) {
                    const float* k = &K[(size_t)(base[b] + j) * kvw + kvh * 64];
                    double d = 0; for (int e = 0; e < 64; ++e) d += (double)q[e] * k[e];
                    p[j] = d * scale; mx = std::max(mx, p[j]);
                }
                for (auto& v : p) { v = std::exp(v - mx); l += v; }
                for (int e = 0; e < 64; ++e) {
                    double o = 0; for (int j = 0; j < lens[b]; ++j) o += p[j] * K[(size_t)(base[b] + j) * kvw + nkv * 64 + kvh * 64 + e];
                    ov[e] = o / l;
                }
                for (int e = 0; e < 64; ++e) {
                    double am = 0; for (int i = e & ~31; i < (e & ~31) + 32; ++i) am = std::max(am, std::fabs(ov[i]));
                    const double step = am / 127.0;
                    const size_t xi = (size_t)b * cols + h * 64 + e, si = (size_t)b * cols / 32 + h * 2 + e / 32;
                    wn = std::max(wn, std::fabs(xn[xi] * (double)sn[si] - ov[e]) / step);
                    wo = std::max(wo, std::fabs(xo[xi] * (double)so[si] - ov[e]) / step);
                }
            }
            const bool pass = ok && inv && wn < 0.75 && wo < 0.75;
            printf("gpu decode attn G=%d %s KV, %d rows (1..1921 keys, a 3-row spec window): split-K %.3f / per-head %.3f int8 steps from the double reference; split-K invariant to stride, batch, order: %s %s\n",
                   G, q8 ? "int8" : "f32", B, wn, wo, inv ? "yes" : "NO", pass ? "PASS" : "FAIL");
            if (!pass) ++fails;
        }
        gpu::release(dC);
    }
    {   // device time per attention layer at the LFM2.5-350M decode shape (16 q / 8 kv heads): best of 3 interleaved rounds of 30 launches
        const int G = 2, nq = nkv * G, cols = nq * 64, ldc = cols + kvw, R = 4, LM = 1921;
        std::vector<float> Cq((size_t)R * ldc), kt((size_t)R * LM * kvw);
        for (auto& v : Cq) v = nd(rg);
        for (auto& v : kt) v = nd(rg);
        std::vector<int8_t> tq; std::vector<float> ts; quant_rows_q8(kt, tq, ts);
        std::vector<int> bs(R), la(R, LM), lb(R, 200);
        for (int r = 0; r < R; ++r) bs[r] = r * LM;
        gpu::Buf dC = gpu::upload(Cq.data(), Cq.size() * 4), dK = gpu::upload(kt.data(), kt.size() * 4), dQ = gpu::upload(tq.data(), tq.size()),
                 dS = gpu::upload(ts.data(), ts.size() * 4), dBs = gpu::upload(bs.data(), R * 4), dLa = gpu::upload(la.data(), R * 4), dLb = gpu::upload(lb.data(), R * 4);
        gpu::ensure_qscratch(R, cols);
        struct Cfg { int q8, nb, L; const char* name; };
        const Cfg cfg[] = {{0, 1, LM, "f32 1x1921"}, {1, 1, LM, "int8 1x1921"}, {0, 4, LM, "f32 4x1921"}, {1, 4, LM, "int8 4x1921"}, {0, 1, 200, "f32 1x200"}, {1, 1, 200, "int8 1x200"}};
        double best[6][2];
        for (auto& v : best) v[0] = v[1] = 1e9;
        for (int round = 0; round < 3; ++round) for (int c = 0; c < 6; ++c) for (int sp = 0; sp < 2; ++sp) {
            const Cfg& f = cfg[c]; const gpu::Buf dl = f.L == LM ? dLa : dLb; const int ml = sp ? f.L : 0;
            auto op = [&] { return f.q8 ? gpu::op_lm_attn_dec_q(dC, ldc, dQ, dS, kvw, nkv * 64, dBs, dl, nq, nkv, scale, f.nb, ml)
                                        : gpu::op_lm_attn_dec(dC, ldc, dK, kvw, nkv * 64, dBs, dl, nq, nkv, scale, f.nb, ml); };
            for (int i = 0; i < 3; ++i) op();
            gpu::finish();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 30; ++i) op();
            gpu::finish();
            best[c][sp] = std::min(best[c][sp], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 30);
        }
        printf("gpu decode attn, one layer at the 350M shape, ms (per-head -> split-K):");
        for (int c = 0; c < 6; ++c) printf(" %s %.3f -> %.3f%s", cfg[c].name, best[c][0], best[c][1], c < 5 ? "," : "\n");
        for (gpu::Buf x : {dC, dK, dQ, dS, dBs, dLa, dLb}) gpu::release(x);
    }
    gpu::release(dKV); gpu::release(dKQ); gpu::release(dKS); gpu::release(dB); gpu::release(dL);
    return fails;
}

// ---- fp16 KV cache (--kv-dtype f16): the write kernel stores rounded halves of norm+rope K and raw V; split-K decode reads them back
static uint16_t f2h_bits(float x) {                    // round to nearest even, clamp to the fp16 range (what vstore_half4_rte does)
    uint32_t u; memcpy(&u, &x, 4);
    const uint16_t s = (uint16_t)((u >> 16) & 0x8000u);
    const uint32_t eb = (u >> 23) & 0xffu;
    uint32_t m = u & 0x7fffffu;
    if (eb == 0xff) return (uint16_t)(s | 0x7c00u);
    int e = (int)eb - 127 + 15;
    if (e >= 31) return (uint16_t)(s | 0x7bffu);
    if (e <= 0) {
        if (e < -10) return s;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t)(s | r);
    }
    uint32_t r = m >> 13, rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) { ++r; if (r == 0x400u) { r = 0; if (++e >= 31) return (uint16_t)(s | 0x7bffu); } }
    return (uint16_t)(s | ((uint32_t)e << 10) | r);
}
static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) {
        if (!m) o = s;
        else { int sh = 0; while (!(m & 0x400u)) { m <<= 1; ++sh; } m &= 0x3ffu; o = s | ((uint32_t)(127 - 14 - sh) << 23) | (m << 13); }
    } else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &o, 4); return f;
}
static int test_kv_f16() {
    if (!gpu::kv_f16_available()) { printf("gpu f16 KV: FAIL (kernels did not build)\n"); return 1; }
    int fails = 0;
    const int nkv = 8, nq = 16, hd = 64, hlf = 32, kvw = 2 * nkv * hd, rowc = nq * hd + kvw;
    const int rows = 1921, pmax = 2048;
    std::mt19937 rg(21); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> C((size_t)rows * rowc), qw(hd), kw(hd), cosf((size_t)pmax * hlf), sinf((size_t)pmax * hlf);
    for (auto& v : C) v = nd(rg);
    for (int i = 0; i < hd; ++i) { qw[i] = 1.f + 0.1f * (i % 3); kw[i] = 1.f + 0.1f * (i % 5); }
    for (int p = 0; p < pmax; ++p) for (int i = 0; i < hlf; ++i) {
        const double th = p / std::pow(10000.0, 2.0 * i / hd);
        cosf[(size_t)p * hlf + i] = (float)std::cos(th); sinf[(size_t)p * hlf + i] = (float)std::sin(th);
    }
    const std::vector<float> Craw = C;                                  // the kernel rewrites q/k in place
    std::vector<int> pos(rows), dst(rows);
    for (int t = 0; t < rows; ++t) { pos[t] = t; dst[t] = t; }
    gpu::Buf dC = gpu::upload(C.data(), C.size() * 4), dqw = gpu::upload(qw.data(), hd * 4), dkw = gpu::upload(kw.data(), hd * 4);
    gpu::Buf dcos = gpu::upload(cosf.data(), cosf.size() * 4), dsin = gpu::upload(sinf.data(), sinf.size() * 4);
    gpu::Buf dpos = gpu::upload(pos.data(), rows * 4), ddst = gpu::upload(dst.data(), rows * 4);
    gpu::Buf dKV = gpu::alloc((size_t)rows * kvw * 2);
    bool wok = gpu::op_lm_qkv_post_kv_f(dC, rowc, rows, nq, nkv, hd, dqw, dkw, dcos, dsin, dpos, hlf, 1e-6f, dKV, ddst, kvw);
    std::vector<uint16_t> kh((size_t)rows * kvw);
    wok = wok && gpu::read(dKV, kh.data(), kh.size() * 2);
    std::vector<float> deq(kh.size());
    for (size_t i = 0; i < kh.size(); ++i) deq[i] = h2f(kh[i]);
    double worst = 0, vmax = 0; int clampn = 0;
    for (int t = 0; wok && t < rows; ++t) {
        const float* r = &Craw[(size_t)t * rowc];
        for (int s = nq; s < nq + 2 * nkv; ++s) {                        // cached slots: K heads, then V heads (q is not cached)
            double ref[64];
            if (s >= nq + nkv) { for (int i = 0; i < hd; ++i) ref[i] = r[(size_t)s * hd + i]; }
            else {
                double ss = 0; for (int i = 0; i < hd; ++i) ss += (double)r[(size_t)s * hd + i] * r[(size_t)s * hd + i];
                const double inv = 1.0 / std::sqrt(ss / hd + 1e-6);
                for (int i = 0; i < hlf; ++i) {
                    const double a = r[(size_t)s * hd + i] * inv * kw[i], b = r[(size_t)s * hd + i + hlf] * inv * kw[i + hlf];
                    const double c = cosf[(size_t)pos[t] * hlf + i], sn = sinf[(size_t)pos[t] * hlf + i];
                    ref[i] = a * c - b * sn; ref[i + hlf] = b * c + a * sn;
                }
            }
            const int off = s < nq + nkv ? (s - nq) * hd : nkv * hd + (s - nq - nkv) * hd;
            for (int i = 0; i < hd; ++i) {
                const double got = deq[(size_t)t * kvw + off + i];
                vmax = std::max(vmax, std::fabs(ref[i]));
                if (std::fabs(ref[i]) > 65504.0) ++clampn;
                worst = std::max(worst, std::fabs(got - ref[i]) / (std::fabs(ref[i]) * (1.0 / 1024.0) + 1e-7));
            }
        }
    }
    // split-K decode on the stored rows, vs a double reference on the same (dequantized) values; q of row 0 as the kernel left it
    std::vector<float> C0(C.begin(), C.begin() + rowc);
    std::vector<int> base1{0}, len1{rows};
    gpu::Buf dB = gpu::upload(base1.data(), 4), dL = gpu::upload(len1.data(), 4), dC0 = gpu::upload(C0.data(), rowc * 4);
    gpu::ensure_qscratch(1, nq * hd);
    std::vector<int8_t> xp((size_t)nq * hd); std::vector<float> xs((size_t)nq * hd / 32);
    bool aok = wok && gpu::op_lm_attn_dec_f(dC0, rowc, dKV, kvw, nkv * hd, dB, dL, nq, nkv, 0.125f, 1, rows)
               && gpu::read_qscratch(xp.data(), xp.size(), xs.data(), xs.size() * 4);
    double worst_a = 0;
    for (int h = 0; aok && h < nq; ++h) {
        const int kvh = h / (nq / nkv);
        std::vector<double> p(rows); double mx = -1e300, l = 0, ov[64];
        for (int j = 0; j < rows; ++j) {
            double d = 0; for (int e = 0; e < hd; ++e) d += (double)C0[(size_t)h * hd + e] * deq[(size_t)j * kvw + kvh * hd + e];
            p[j] = d * 0.125; mx = std::max(mx, p[j]);
        }
        for (auto& v : p) { v = std::exp(v - mx); l += v; }
        for (int e = 0; e < hd; ++e) { double o = 0; for (int j = 0; j < rows; ++j) o += p[j] * deq[(size_t)j * kvw + nkv * hd + kvh * hd + e]; ov[e] = o / l; }
        for (int e = 0; e < hd; ++e) {
            double am = 0; for (int i = e & ~31; i < (e & ~31) + 32; ++i) am = std::max(am, std::fabs(ov[i]));
            const double got = xp[(size_t)h * hd + e] * (double)xs[h * 2 + (e >> 5)];
            worst_a = std::max(worst_a, std::fabs(got - ov[e]) / (am / 127.0));
        }
    }
    // speed: f32 vs f16 storage of the same values, split-K, 1921 keys, 1 stream (best of 2 x 20 launches)
    std::vector<float> deqv(deq.begin(), deq.end());
    gpu::Buf dKV32 = gpu::upload(deqv.data(), deqv.size() * 4);
    double ms[2] = {1e9, 1e9};
    for (int round = 0; aok && round < 2; ++round) for (int m = 0; m < 2; ++m) {
        auto op = [&] { return m ? gpu::op_lm_attn_dec_f(dC0, rowc, dKV, kvw, nkv * hd, dB, dL, nq, nkv, 0.125f, 1, rows)
                                : gpu::op_lm_attn_dec(dC0, rowc, dKV32, kvw, nkv * hd, dB, dL, nq, nkv, 0.125f, 1, rows); };
        for (int i = 0; i < 3; ++i) op();
        gpu::finish();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) op();
        gpu::finish();
        ms[m] = std::min(ms[m], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20);
    }
    // the same host reference against the f32 write kernel: isolates a wrong reference from a wrong f16 store
    double worst32 = 0;
    std::vector<float> Kv32v((size_t)rows * kvw);
    {
        std::vector<float> C2 = Craw, &Kv32 = Kv32v;
        gpu::Buf dC2 = gpu::upload(C2.data(), C2.size() * 4), dKV32w = gpu::alloc((size_t)rows * kvw * 4);
        if (gpu::op_lm_qkv_post_kv(dC2, rowc, rows, nq, nkv, hd, dqw, dkw, dcos, dsin, dpos, hlf, 1e-6f, dKV32w, ddst, kvw)
            && gpu::read(dKV32w, Kv32.data(), Kv32.size() * 4)) {
            for (int t = 0; t < rows; ++t) {
                const float* r = &Craw[(size_t)t * rowc];
                for (int s = nq; s < nq + 2 * nkv; ++s) {
                    double ref[64];
                    if (s >= nq + nkv) { for (int i = 0; i < hd; ++i) ref[i] = r[(size_t)s * hd + i]; }
                    else {
                        double ss = 0; for (int i = 0; i < hd; ++i) ss += (double)r[(size_t)s * hd + i] * r[(size_t)s * hd + i];
                        const double inv = 1.0 / std::sqrt(ss / hd + 1e-6);
                        for (int i = 0; i < hlf; ++i) {
                            const double a = r[(size_t)s * hd + i] * inv * kw[i], b = r[(size_t)s * hd + i + hlf] * inv * kw[i + hlf];
                            const double c = cosf[(size_t)pos[t] * hlf + i], sn = sinf[(size_t)pos[t] * hlf + i];
                            ref[i] = a * c - b * sn; ref[i + hlf] = b * c + a * sn;
                        }
                    }
                    const int off = s < nq + nkv ? (s - nq) * hd : nkv * hd + (s - nq - nkv) * hd;
                    for (int i = 0; i < hd; ++i) worst32 = std::max(worst32, std::fabs(Kv32[(size_t)t * kvw + off + i] - ref[i]));
                }
            }
        }
        gpu::release(dC2); gpu::release(dKV32w);
    }
    // direct comparison of the two storages (same inputs, same math): f16 must equal round(f32) wherever it differs, the kernel is wrong
    double d1632 = 0, wr16 = 0, wr32 = 0; int wt = -1, ws = -1, wi = -1;
    for (int t = 0; t < rows; ++t) for (int s = nq; s < nq + 2 * nkv; ++s) {
        const int off = s < nq + nkv ? (s - nq) * hd : nkv * hd + (s - nq - nkv) * hd;
        for (int i = 0; i < hd; ++i) {
            const double a = h2f(kh[(size_t)t * kvw + off + i]), b = Kv32v[(size_t)t * kvw + off + i];
            const double u = std::fabs(a - b) / (std::fabs(b) * (1.0 / 1024.0) + 1e-7);
            if (u > d1632) { d1632 = u; wt = t; ws = s; wi = i; wr16 = a; wr32 = b; }
        }
    }
    printf("gpu f16 KV: f16 vs f32 storage worst %.2f ulp at t=%d s=%d i=%d (f16 %.9g, f32 %.9g)\n", d1632, wt, ws, wi, wr16, wr32);
    const bool pass = wok && aok && worst < 1.5 && worst_a < 0.75;   // 1 ulp of the fp16 subnormal grid is coarser than |x|/1024
    printf("gpu f16 KV: f32 write kernel vs the same reference: maxabs %.3e; f16 store worst %.1f ulp; decode %.2f int8 steps; 1 layer 1921 keys f32 %.3f vs f16 %.3f ms; max|V| %.1f, %d clamped %s\n",
           worst32, worst, worst_a, ms[0], ms[1], vmax, clampn, pass ? "PASS" : "FAIL");
    printf("gpu f16 KV: K/V stored within %.2f fp16 ulp (max|V| %.1f, %d clamped); split-K decode %.2f int8 steps from the double reference; "
           "1 layer 1921 keys f32 %.3f vs f16 %.3f ms %s\n", worst, vmax, clampn, worst_a, ms[0], ms[1], pass ? "PASS" : "FAIL");
    if (!pass) ++fails;
    for (gpu::Buf x : {dC, dqw, dkw, dcos, dsin, dpos, ddst, dKV, dKV32, dB, dL, dC0}) gpu::release(x);
    return fails;
}

int main() {
    Pool::get().init(4);
    int fails = 0;
    {   // pool: exact results, no CPU burn while idle (workers park after a short spin), no lost wake-ups across park/wake cycles
        std::vector<int> v(10007);
        auto run = [&](int n) {
            std::fill(v.begin(), v.end(), -1);
            Pool::get().parallel_for(n, [&](int lo, int hi) { for (int i = lo; i < hi; ++i) v[i] = i; });
            for (int i = 0; i < n; ++i) if (v[i] != i) return false;
            return true;
        };
        bool ok = run((int)v.size());
        timespec c0, c1;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0);
        usleep(300000);                                                    // 3 workers, 300 ms with nothing to do
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c1);
        const double idle_ms = (c1.tv_sec - c0.tv_sec) * 1e3 + (c1.tv_nsec - c0.tv_nsec) / 1e6;
        for (int r = 0; ok && r < 300; ++r) { ok = run(997 + r); if (r % 3 == 0) usleep(1000); }   // a lost wake-up hangs here
        const bool pass = ok && idle_ms < 30.0;
        printf("pool: results %s, CPU used by 3 idle workers over 300 ms = %.1f ms (spinning pool: ~900) %s\n", ok ? "exact" : "WRONG", idle_ms, pass ? "PASS" : "FAIL");
        if (!pass) ++fails;
    }
    {   // sampler: repetition-penalty window (-1 whole context, N last tokens, 0 off) and scope (penalty_from = prompt length)
        auto pick = [](int last_n, int from) {
            SamplerParams p; p.temperature = 0.f; p.repetition_penalty = 2.f; p.repeat_last_n = last_n; p.penalty_from = from;
            Sampler s(p);
            std::vector<float> lg = {0.f, 9.f, 10.f, 9.5f, 6.f, 0.f};
            return s.sample(lg.data(), (int)lg.size(), std::vector<int>{1, 2, 3});   // prompt {1, 2}, generated {3}
        };
        const int a = pick(-1, 0), g = pick(-1, 2), w = pick(2, 0), o = pick(0, 0), gw = pick(2, 2);
        const bool pass = a == 4 && g == 2 && w == 1 && o == 2 && gw == 2;
        printf("sampler penalty window/scope: whole=%d generated=%d last2=%d off=%d generated+last2=%d (want 4 2 1 2 2) %s\n", a, g, w, o, gw, pass ? "PASS" : "FAIL");
        if (!pass) ++fails;
    }
    // --- Q8 GEMM: fma path vs scalar reference (same WMat) ---
    {
        int rows = 128, cols = 512, T = 20;
        std::vector<float> src((size_t)rows * cols);
        for (auto& v : src) v = frand();
        WMat W = pack_from_f32(src.data(), rows, cols, cols, "test");
        std::vector<float> X((size_t)T * cols);
        for (auto& v : X) v = frand() * 2;
        // aligned copy of X for fma path
        float* Xa; posix_memalign((void**)&Xa, 64, ((size_t)T * cols * 4 + 63) / 64 * 64);
        memcpy(Xa, X.data(), (size_t)T * cols * 4);
        // fma path: dequantize W to f32 and use gemm_f32 ref as oracle
        std::vector<float> Wf((size_t)rows * cols);
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < W.nb; ++b)
                for (int i = 0; i < QK8_0; ++i)
                    Wf[(size_t)r * cols + b * QK8_0 + i] = (float)W.wq[(size_t)r * cols + b * QK8_0 + i] * W.ws[(size_t)r * W.nb + b];
        std::vector<float> Yref((size_t)T * rows), Ygot((size_t)T * rows);
        gemm_f32(Wf.data(), rows, cols, Xa, T, Yref.data(), rows);
        gemm_q8_fma(W, Xa, T, nullptr, nullptr, Ygot.data(), rows);
        double d = rel(Ygot, Yref);
        printf("gemm_q8_fma vs gemm_f32(Wf): rel=%.3e %s\n", d, d < 1e-4 ? "PASS" : "FAIL");
        if (d > 1e-4) ++fails;
        free(Xa);
    }
    // --- int8 GEMM path vs its own scalar ref (self-consistent) ---
    {
        int rows = 64, cols = 256, T = 8;
        std::vector<float> src((size_t)rows * cols);
        for (auto& v : src) v = frand();
        WMat W = pack_from_f32(src.data(), rows, cols, cols, "test8");
        std::vector<float> X((size_t)T * cols);
        for (auto& v : X) v = frand() * 2;
        ActQ aq; quant_act(X.data(), T, cols, aq);
        std::vector<float> Y((size_t)T * rows);
        gemm_q8(W, aq, T, Y.data(), rows);
        // oracle: dequant both and dot
        double worst = 0;
        for (int t = 0; t < T; ++t) for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int c = 0; c < cols; ++c) {
                float xv = ((float)aq.qx[(size_t)t * cols + c] - 128.f) * aq.sx[(size_t)t * aq.nb + c / QK8_0];
                float wv = (float)W.wq[(size_t)r * cols + c] * W.ws[(size_t)r * W.nb + c / QK8_0];
                acc += xv * wv;
            }
            worst = std::max(worst, std::fabs(acc - Y[(size_t)t * rows + r]));
        }
        printf("gemm_q8 int8 vs dequant oracle: maxabs=%.3e %s\n", worst, worst < 1e-2 ? "PASS" : "FAIL");
        if (worst > 1e-2) ++fails;
    }
    // --- rmsnorm ---
    {
        int T = 16, cols = 512;
        std::vector<float> x((size_t)T * cols), w(cols), y1((size_t)T * cols), y2((size_t)T * cols);
        for (auto& v : x) v = frand();
        for (auto& v : w) v = 1 + frand() * 0.2f;
        // manual oracle
        for (int t = 0; t < T; ++t) {
            double ss = 0; for (int c = 0; c < cols; ++c) ss += (double)x[(size_t)t*cols+c]*x[(size_t)t*cols+c];
            float inv = (float)(1.0/std::sqrt(ss/cols + 1e-6));
            for (int c = 0; c < cols; ++c) y1[(size_t)t*cols+c] = x[(size_t)t*cols+c]*inv*w[c];
        }
        rmsnorm_q(x.data(), T, cols, w.data(), 1e-6f, y2.data());
        double d = rel(y2, y1);
        printf("rmsnorm: rel=%.3e %s\n", d, d < 1e-5 ? "PASS" : "FAIL");
        if (d > 1e-5) ++fails;
    }
    // --- rope: scalar vs fma ---
    {
        RopeTable rt; rt.build(1000000.0, 256, 128);
        int T = 6, heads = 4, hd = 256;
        std::vector<float> a((size_t)T*heads*hd), b = a;
        for (auto& v : a) v = frand();
        b = a;
        std::vector<int> pos(T); for (int i = 0; i < T; ++i) pos[i] = i * 7;
        // scalar path via isa override
        set_isa_level(0);
        rope_apply(a.data(), T, heads, hd, rt, pos.data());
        set_isa_level(1);
        rope_apply(b.data(), T, heads, hd, rt, pos.data());
        double d = maxdiff(a, b);
        printf("rope fma vs scalar: max=%.3e %s\n", d, d < 1e-5 ? "PASS" : "FAIL");
        if (d > 1e-5) ++fails;
    }
    // --- attention: scalar vs fma ---
    {
        int T = 12, nq = 4, nkv = 2, hd = 64;
        std::vector<float> Q((size_t)T*nq*hd), K((size_t)T*nkv*hd), V((size_t)T*nkv*hd);
        for (auto& v : Q) v = frand(); for (auto& v : K) v = frand(); for (auto& v : V) v = frand();
        std::vector<int> seg(T); for (int i = 0; i < T; ++i) seg[i] = i / 5;
        std::vector<float> O1((size_t)T*nq*hd, 0), O2((size_t)T*nq*hd, 0);
        set_isa_level(0);
        attention(Q.data(), K.data(), V.data(), O1.data(), T, nq, nkv, hd, 0.5f, seg.data(), 6, nq*hd, nkv*hd, nq*hd);
        set_isa_level(1);
        attention(Q.data(), K.data(), V.data(), O2.data(), T, nq, nkv, hd, 0.5f, seg.data(), 6, nq*hd, nkv*hd, nq*hd);
        double d = maxdiff(O1, O2);
        printf("attention fma vs scalar: max=%.3e %s\n", d, d < 1e-5 ? "PASS" : "FAIL");
        if (d > 1e-5) ++fails;
    }
    { // gelu_mul vs exact erff reference
        int T = 5, cols = 64; std::vector<float> gu((size_t)T * 2 * cols), y((size_t)T * cols);
        for (size_t i = 0; i < gu.size(); ++i) gu[i] = -30.f + 60.f * (float)i / (float)gu.size();
        gelu_mul_block(gu.data(), T, cols, y.data());
        double worst = 0;
        for (int t = 0; t < T; ++t) for (int i = 0; i < cols; ++i) {
            float g = gu[(size_t)t * 2 * cols + i], u = gu[(size_t)t * 2 * cols + cols + i];
            float ref = 0.5f * g * (1.f + erff(g * 0.70710678118654752440f)) * u;
            worst = std::max(worst, (double)std::fabs(y[(size_t)t * cols + i] - ref));
        }
        printf("gelu_mul vs erff oracle: maxabs=%.3g %s\n", worst, worst < 2e-4 ? "PASS" : "FAIL");
        if (worst >= 2e-4) fails++;
    }
    {   // iGPU int8 DP4A GEMM vs CPU GEMM (skipped when no OpenCL GPU with integer dot product)
        if (!gpu::init(true)) printf("gpu dp4a gemm: SKIP (no capable GPU)\n");
        else {
            const int rows = 512, cols = 1024, T = 37;
            std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<float> wsrc((size_t)rows * cols), X((size_t)T * cols), Yc((size_t)T * rows), Yg((size_t)T * rows);
            for (auto& v : wsrc) v = nd(rng) * 0.05f;
            for (auto& v : X) v = nd(rng);
            WMat W = pack_from_f32(wsrc.data(), rows, cols, cols, "t");
            pack_q8_to_f32(W);
            gemm_q8_fma(W, X.data(), T, nullptr, nullptr, Yc.data(), rows);
            int slot = gpu::upload_q8(W);
            bool ok = slot >= 0 && gpu::gemm(slot, X.data(), T, Yg.data());
            double num = 0, den = 0;
            for (size_t i = 0; i < Yc.size(); ++i) { num += (Yg[i] - Yc[i]) * (double)(Yg[i] - Yc[i]); den += (double)Yc[i] * Yc[i]; }
            double rel = std::sqrt(num / den);
            printf("gpu dp4a gemm (%s) vs cpu: rel_l2=%.2e %s\n", gpu::device_name(), rel, ok && rel < 1e-2 ? "PASS" : "FAIL");
            if (!ok || rel >= 1e-2) ++fails;
            fails += test_dec_attn();
            fails += test_kv_f16();
        }
    }
    Pool::get().shutdown();
    printf(fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
    return fails;
}
