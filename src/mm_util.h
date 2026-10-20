// mm_util.h — weight-loading / small-op helpers shared by the multimodal encoders (mm_encoder.cpp, mm_gemma4.cpp)
#pragma once
#include "gguf.h"
#include "kernels.h"
#include <string>
#include <vector>

namespace ferro {

// linear layer weight: Q8_0 kept bit-exact (int8 path), F32/F16 kept as f32 (exact)
struct Lin {
    WMat q;                    // Q8 weights
    std::vector<float> f;      // f32 weights [rows][cols] (when the GGUF has float weights)
    std::vector<float> b;      // bias (optional)
    int rows = 0, cols = 0;
};
enum Act { GELU_TANH, GELU_ERF, SILU, RELU };

std::vector<float> tf32(const GgufFile& g, const std::string& name);
Lin lin(const GgufFile& g, const std::string& w, const std::string& b = "");
void apply(const Lin& L, const float* X, int T, float* Y);      // Y[T][rows] = X[T][cols] W^T + b
void act(float* x, size_t n, Act a);
void add_inplace(float* y, const float* x, size_t n, float s = 1.0f);

}  // namespace ferro
