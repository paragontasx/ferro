// mm_gemma4_impl.h - private state of Gemma4Proj, shared by the vision (mm_gemma4.cpp) and audio (mm_gemma4a.cpp) encoders.
#pragma once
#include "gpu_cl.h"
#include "mm_clin.h"
#include "mm_gemma4.h"

namespace ferro {

struct G4VLayer {
    std::vector<float> ln1, ln2, qn, kn, post_attn, post_ffn;
    CLin q, k, v, o, gate, up, down;
};

struct Gemma4Proj::Impl {
    int d = 768, layers = 16, heads = 12, hd = 64, ps = 16, merge = 3, ffn = 3072, proj = 512;
    float eps = 1e-6f, theta = 100.0f;
    std::vector<float> patch_w, pos_tbl;   // patch_w [768][3*16*16]; pos_tbl [2][10240][768]
    int pos_size = 10240;
    std::vector<G4VLayer> vl;
    CLin mm_proj;                          // 768 -> 512
    // ---- audio (gemma4a conformer; compute in mm_gemma4a.cpp)
    struct ALayer {
        std::vector<float> ffn_norm, ffn_post_norm, ffn_norm1, ffn_post_norm1, attn_pre_norm, attn_post_norm,
                           norm_conv, conv_norm, ln2, per_dim_scale, conv_dw;
        CLin ffn_up, ffn_down, ffn_up1, ffn_down1, q, k, v, o, k_rel, pw1, pw2;
    };
    std::vector<ALayer> al;
    std::vector<float> a_conv_w[2], a_conv_norm[2];     // sscp convs [Cout][Cin][3][3] (f32 in the file) + channel LayerNorm weights
    std::vector<float> a_inp_proj;                      // [1024][1024] f32
    CLin a_out_proj, a_mm_proj;                         // 1024 -> 1536 (+bias), 1536 -> 512
    std::vector<float> a_out_bias;
    int a_d = 1024, a_layers = 12, a_heads = 8, a_hd = 128, a_ffn = 4096, a_mel = 128;
    // ---- iGPU state (upload_gpu)
    struct GL { WMat qkv, gu; gpu::Buf ln1 = nullptr, ln2 = nullptr, qn = nullptr, kn = nullptr, pa = nullptr, pf = nullptr; };
    std::vector<GL> gl;
    WMat patch_q;
    int patch_slot = -1, qkv_slot_ = -1;
    gpu::Buf invf = nullptr, pos_x = nullptr, pos_y = nullptr, bX = nullptr, bC = nullptr, bD = nullptr, bG = nullptr, bT = nullptr;
    gpu::Buf bS = nullptr, bsrc = nullptr, ones = nullptr, bxs = nullptr, bys = nullptr, blo = nullptr, bhi = nullptr;
    size_t cap = 0, capS = 0, capm = 0;
    bool gpu_ok = false;
    // ---- audio conformer resident on the device (upload_audio_dev, mm_gemma4a.cpp)
    struct AGL { WMat qkv; gpu::Buf n_ffn = nullptr, p_ffn = nullptr, n_ffn1 = nullptr, p_ffn1 = nullptr, n_att = nullptr, p_att = nullptr,
                                    n_conv = nullptr, w_conv = nullptr, ln2 = nullptr, pds = nullptr, dw = nullptr, pk = nullptr; };
    std::vector<AGL> agl;
    gpu::Buf aX = nullptr, aY = nullptr, aO = nullptr;
    size_t a_cap = 0;                                   // tokens the three activation buffers hold
    bool a_dev_ok = false;
};
bool upload_audio_dev(Gemma4Proj::Impl& I);             // mm_gemma4a.cpp

}  // namespace ferro
