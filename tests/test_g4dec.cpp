// tests/test_g4dec.cpp -- E1 unit test: decode-shape attention (nr==1, causal) vs CPU reference.
// Usage: test_g4dec [hd] [n] ; env FERRO_G4_DEC=0 tests the tiled path instead.
#include "gpu_cl.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace ferro;
int main(int argc, char** argv) {
    if (!gpu::init(false)) { fprintf(stderr, "gpu init failed\n"); return 2; }
    const int nq = 8, nkv = 1;
    const int hd = argc > 1 ? atoi(argv[1]) : 512;
    const int n = argc > 2 ? atoi(argv[2]) : 355;
    const int kd = nkv * hd, ldkv = 2 * kd, qld = nq * hd;
    const int q0 = n - 1;
    std::vector<float> Q(qld), KV((size_t)n * ldkv), O(qld, -777.f), ref(qld);
    srand(1234);
    auto rnd = [] { return (rand() / (float)RAND_MAX - 0.5f) * 2.f; };
    for (auto& x : Q) x = rnd();
    for (auto& x : KV) x = rnd();
    gpu::Buf bQ = gpu::upload(Q.data(), Q.size() * 4);
    gpu::Buf bKV = gpu::upload(KV.data(), KV.size() * 4);
    gpu::Buf bS = gpu::alloc((size_t)nq * n * 4);
    gpu::Buf bO = gpu::alloc(O.size() * 4);
    if (!bQ || !bKV || !bS || !bO) { fprintf(stderr, "alloc failed\n"); return 2; }
    std::vector<double> sraw0;
    for (int h = 0; h < nq; ++h) {
        const float* q = Q.data() + (size_t)h * hd;
        std::vector<double> s(n);
        double m = -1e300;
        for (int j = 0; j <= q0; ++j) {
            const float* k = KV.data() + (size_t)j * ldkv;
            double d = 0; for (int e = 0; e < hd; ++e) d += (double)q[e] * k[e];
            s[j] = d; if (d > m) m = d;
        }
        if (h == 0) sraw0 = s;   // keep head-0 raw dot products for the debug dump
        double l = 0; for (int j = 0; j <= q0; ++j) { s[j] = exp(s[j] - m); l += s[j]; }
        for (int c = 0; c < hd; ++c) {
            double a = 0;
            for (int j = 0; j <= q0; ++j) a += s[j] * (double)KV[(size_t)j * ldkv + kd + c];
            ref[h * hd + c] = (float)(a / l);
        }
    }
    if (!gpu::op_attn_split(bQ, qld, bKV, ldkv, 0, 0, kd, n, q0, 1, nq, nkv, hd, 1.0f, bS, bO, qld, -1, 1)) {
        fprintf(stderr, "op_attn_split failed\n"); return 3;
    }
    if (!gpu::read(bO, O.data(), O.size() * 4)) { fprintf(stderr, "read failed\n"); return 4; }
    double worst = 0; int wi = -1; double refmax = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = fabs((double)O[i] - ref[i]);
        if (fabs(ref[i]) > refmax) refmax = fabs(ref[i]);
        if (d > worst) { worst = d; wi = (int)i; }
    }
    printf("hd=%d n=%d maxdiff=%.5g @ %d (ref %.4g got %.4g, refmax %.4g)\n",
           hd, n, worst, wi, ref[wi], O[wi], refmax);
    return worst > 0.02 ? 1 : 0;
}
