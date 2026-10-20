// mm_clin.h - Gemma4ClippableLinear helpers shared by the gemma4 vision (mm_gemma4.cpp) and audio (mm_gemma4a.cpp) encoders:
//   y = clamp(W clamp(x, in_min, in_max), out_min, out_max); the four limits are 1-element f32 tensors next to the weight
//   (absent -> +-FLT_MAX = no clamp; the vision tower of the shipped mmproj has none, every audio linear has them).
#pragma once
#include "mm_util.h"
#include "threadpool.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <string>
#include <vector>

namespace ferro {

struct CLin {
    Lin w;
    float imin = -FLT_MAX, imax = FLT_MAX, omin = -FLT_MAX, omax = FLT_MAX;
};
inline float scalar_or(const GgufFile& g, const std::string& name, float dflt) {
    if (!g.tensor(name)) return dflt;
    return tf32(g, name).at(0);
}
inline CLin clin(const GgufFile& g, const std::string& wname) {
    CLin c;
    c.w = lin(g, wname);
    const std::string base = wname.substr(0, wname.size() - strlen(".weight"));
    c.imin = scalar_or(g, base + ".input_min", -FLT_MAX);
    c.imax = scalar_or(g, base + ".input_max", FLT_MAX);
    c.omin = scalar_or(g, base + ".output_min", -FLT_MAX);
    c.omax = scalar_or(g, base + ".output_max", FLT_MAX);
    return c;
}
inline void clamp_buf(float* x, size_t n, float lo, float hi) {
    if (lo <= -FLT_MAX && hi >= FLT_MAX) return;
    Pool::get().parallel_for((int)((n + 8191) / 8192), [&](int a, int b) {
        const size_t i1 = std::min(n, (size_t)b * 8192);
        for (size_t i = (size_t)a * 8192; i < i1; ++i) x[i] = std::min(hi, std::max(lo, x[i]));
    });
}
// Y[T][rows]; X is copied into `scratch` only when an input clamp exists (callers keep X intact)
inline void capply(const CLin& c, const float* X, int T, float* Y, std::vector<float>& scratch) {
    const float* xin = X;
    if (c.imin > -FLT_MAX || c.imax < FLT_MAX) {
        scratch.assign(X, X + (size_t)T * c.w.cols);
        clamp_buf(scratch.data(), scratch.size(), c.imin, c.imax);
        xin = scratch.data();
    }
    apply(c.w, xin, T, Y);
    clamp_buf(Y, (size_t)T * c.w.rows, c.omin, c.omax);
}

}  // namespace ferro
