// ferro — lightweight Intel CPU/iGPU inference engine for Gemma embedding models
// types.h — quantized block types aligned with GGUF Q8_0 / BF16 / F32
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

namespace ferro {

constexpr int QK8_0 = 32;
struct block_q8_0 {
    int16_t d;              // stored as fp16 bits — ggml layout: scale FIRST
    int8_t  qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 must be 34 bytes");

struct block_q4_0 {
    int16_t d;
    uint8_t qs[QK8_0 / 2];
};
static_assert(sizeof(block_q4_0) == 18, "q4_0 must be 18 bytes");

constexpr int QK_K = 256;
struct block_q6_K {
    uint8_t ql[QK_K / 2];   // low 4 bits of 128 + 128 (pairs of 32-element groups)
    uint8_t qh[QK_K / 4];   // high 2 bits per 32-element group
    int8_t  scales[16];     // per 16-element sub-block
    int16_t d;              // fp16 bits
};
static_assert(sizeof(block_q6_K) == 210, "q6_K must be 210 bytes");

// ggml type ids we support
enum GgType : uint32_t {
    T_F32 = 0, T_F16 = 1, T_BF16 = 30, T_Q4_0 = 2, T_Q8_0 = 8, T_Q6_K = 14,
};

inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (expo == 0) {
        if (mant == 0) f = sign;
        else { // subnormal — normalize: value = (1+m'/1024) * 2^(-14-s), biased exp = 113-s
            f = sign;
            uint32_t e = 0x71;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            f |= (e << 23) | (mant << 13);
        }
    } else if (expo == 0x1F) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((expo + 112) << 23) | (mant << 13);
    }
    float out; std::memcpy(&out, &f, 4); return out;
}

inline float bf16_to_f32(uint16_t h) {
    uint32_t f = (uint32_t)h << 16; float out; std::memcpy(&out, &f, 4); return out;
}

// Per-row-block quantized weight tensor stored in our native streaming layout.
// A "GEMM operand" = rows x cols matrix stored row-major, rows padded to
// row_stride elements, each row = a contiguous run of quant blocks.
struct QuantMat {
    int32_t  rows = 0, cols = 0;
    int32_t  row_stride = 0;         // bytes per row
    GgType   type = T_Q8_0;
    const uint8_t* data = nullptr;   // rows * row_stride bytes, 64B aligned
};

} // namespace ferro
