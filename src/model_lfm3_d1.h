// model_lfm3_d1.h — LiquidAI d1-2B (arch "lfm3" + "lfm3.decision.type" == "lfm3-d1"): the LFM3 backbone as a
// STATELESS decision model. One forward per (state, question) prompt, no KV/conv caches:
//   - the 7-tap short convolutions run from a zero state (causal within the prompt)
//   - the 5 SWA attention layers stay causal with the sliding window; the 6 global layers are fully BIDIRECTIONAL
//   - the answer is read at the LAST prompt token: one score per option = max over its label tokens of the
//     tied token_embd logits (there is no decision head in the GGUF)
// Reuses LmModel (model_lfm2_lm.h) for weights and rope tables; the causal LM engine is not involved.
// The iGPU path reuses the LFM3 prefill kernels (tile GEMMs, 7-tap conv from zero state) with dense GEMM-tiled
// attention per prompt (bidirectional on the global layers, causal + window on SWA). score_batch() packs many
// prompts into one forward of up to Tmax rows (segments stay independent: per-segment positions, zero conv state
// at each segment start, attention per segment). Prompts longer than the device Tmax fall back to the CPU forward.
#pragma once
#include "model_lfm2_lm.h"
#include <memory>
#include <string>
#include <vector>

namespace ferro {

// true if the GGUF is an LFM3 decision model (peeks metadata only)
bool is_lfm3_d1(const GgufFile& g);

// one decision prompt for score_batch (pointers must outlive the call)
struct D1Item {
    const std::vector<int>* toks = nullptr;
    const std::vector<std::vector<int>>* labels = nullptr;
    const float* media_rows = nullptr;                                // image rows [d] replacing the embeddings at media_pos
    std::vector<int> media_pos;
};

class Lfm3D1Engine {
public:
    bool init(const std::string& path, int n_threads, std::string* err);
    // one prompt: run it as a whole (no cache) and return scores[j] = max over labels[j] of the last-token logits
    // media_rows/media_pos: image rows [d] each replacing the embedding rows at those token positions (vision)
    bool score(const std::vector<int>& toks, const std::vector<std::vector<int>>& labels,
               std::vector<float>& scores, std::string* err,
               const float* media_rows = nullptr, const std::vector<int>& media_pos = {});
    // many prompts: packed into as few iGPU forwards as fit (Tmax rows each); scores[i] as score() would return
    bool score_batch(const std::vector<D1Item>& items, std::vector<std::vector<float>>& scores, std::string* err);
    int max_prompt() const { return max_ctx_; }
    const LmModel& model() const { return m_; }
    const GgufFile& gguf() const { return m_.gguf; }
    bool on_gpu() const;                                              // impl_ && impl_->ok (Impl defined in the .cpp)
    int gpu_tmax() const;                                             // the device Tmax (0 = CPU only)

    struct Impl;                                                      // opaque iGPU state (public so the .cpp free functions can name it)

private:
    LmModel m_;
    int n_threads_ = 0, max_ctx_ = 0;
    std::shared_ptr<Impl> impl_;
    void forward(const std::vector<int>& ids, float* last_logits,
                 const float* media_rows = nullptr, const std::vector<int>& media_pos = {});   // [n_vocab] logits at the last row; media rows overwrite the embedding rows at media_pos
    // one packed iGPU forward over items[i0, i1): logits [i1 - i0][n_vocab] at each segment's last row
    bool forward_gpu(const std::vector<D1Item>& items, size_t i0, size_t i1, float* logits);
};

}  // namespace ferro
