// mm_gemma4.h — EmbeddingGemma 2 multimodal projector ("mmproj" GGUF, arch clip, projectors gemma4v + gemma4a):
//  * vision: Gemma 4 ViT (16 x 768, patch 16, 12 heads x 64, learned x/y position tables, 2-D neox RoPE theta 100,
//    q/k/v RMS norms, clamped linears, GEGLU FFN) -> 3x3 average pool -> x sqrt(768) -> RMSNorm -> 768->512
//  * audio : see mm_gemma4_audio (conformer)
// Reference: llama.cpp tools/mtmd/models/gemma4v.cpp + clip.cpp build_vit.
#pragma once
#include "gguf.h"
#include "mm_preproc.h"
#include <memory>
#include <string>
#include <vector>

namespace ferro {

class Gemma4Proj {
public:
    bool load(const std::string& path, std::string* err);
    bool has_vision() const { return v_ok_; }
    bool has_audio() const { return a_ok_; }
    int n_embd_out() const { return n_out_; }
    const Gemma4VisionPre& vision_params() const { return vp_; }
    // embeddings [n_tok][n_embd_out] for one preprocessed image (n_tok = (nx/48)*(ny/48))
    std::vector<float> encode_image(const VisionInput& in);
    // move the vision weights to the iGPU (fused qkv / gate|up GEMMs, int8 DP4A); call after gpu::init().
    // encode_image() then runs the whole tower on the device (FERRO_MM_GPU=0 or any failure -> CPU path).
    void upload_gpu();
    bool vision_on_gpu() const;
    struct Impl;
private:
    GgufFile g_;
    bool v_ok_ = false, a_ok_ = false;
    int n_out_ = 0;
    Gemma4VisionPre vp_;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ferro
