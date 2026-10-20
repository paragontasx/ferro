// test_kernels.cpp — kernel correctness: fast paths vs scalar reference
#include "kernels.h"
#include "threadpool.h"
#include "gpu_cl.h"
#include <cstdio>
#include <cmath>
#include <random>
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

int main() {
    Pool::get().init(4);
    int fails = 0;
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
        if (!gpu::init(false)) printf("gpu dp4a gemm: SKIP (no capable GPU)\n");
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
        }
    }
    Pool::get().shutdown();
    printf(fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
    return fails;
}
