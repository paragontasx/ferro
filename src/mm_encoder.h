// mm_encoder.h — d1-omni multimodal projector ("mmproj" GGUF, arch clip):
//  * vision: SigLIP ViT (12 x 768, patch 16, learned 16x16 pos-embd resized per input) + 2x2 pixel
//    unshuffle + MLP (GELU-erf) -> LM embeddings (projector d1omni_v)
//  * audio : FastConformer (conv subsampling x8, 17 blocks, rel-pos attention, conv module) +
//    adapter MLP + residual block (projector d1omni_a)
// Reference: llama.cpp tools/mtmd/models/siglip.cpp, conformer.cpp, clip.cpp.
#pragma once
#include "gguf.h"
#include "kernels.h"
#include "mm_preproc.h"
#include <memory>
#include <string>
#include <vector>

namespace ferro {

class MmProj {
public:
    bool load(const std::string& path, std::string* err);
    bool has_vision() const { return v_ok_; }
    bool has_audio() const { return a_ok_; }
    int n_embd_out() const { return n_out_; }
    int n_mel() const { return a_n_mel_; }
    const VisionPreParams& vision_params() const { return vp_; }
    // move Q8 GEMM weights to the iGPU (GEMMs then run there); call after gpu::init()
    void upload_gpu();
    // embeddings [n_tok][n_embd_out] for a batch of vision inputs (tiles of one or more images), in order
    std::vector<float> encode_vision(const std::vector<VisionInput>& in);
    std::vector<float> encode_audio(const Mel& mel);
    struct Impl;
private:
    GgufFile g_;
    bool v_ok_ = false, a_ok_ = false;
    int n_out_ = 0, a_n_mel_ = 128;
    VisionPreParams vp_;
    std::shared_ptr<Impl> impl_;
};

// path of an mmproj GGUF next to the model (mmproj*.gguf in the same directory), or "" if none
std::string find_mmproj_near(const std::string& model_path);

}  // namespace ferro
