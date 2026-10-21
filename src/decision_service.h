// decision_service.h - the d1-omni decision pipeline as a reusable object (CLI `decide` and `serve --d1` share it):
//   prepare(request JSON): parse, decode + crop + preprocess media, build one token sequence per question   (CPU)
//   infer(prepared...):    media encoders + ONE packed forward over the questions of all given requests    (iGPU)
//   format(prepared, scores): the llama.cpp /v1/systemone response JSON
#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "decision.h"
#include "media.h"
#include "mm_encoder.h"
#include "model_lfm2.h"
#include "tokenizer_bpe.h"

namespace ferro {

struct DecisionPrepared {
    struct ReqMedia { bool is_audio = false; int n_files = 0; std::vector<VisionInput> vin; Mel mel; int n_tok = 0; };
    std::vector<DecisionRequest> reqs;
    bool batch = false;                       // the body was a JSON array -> the response is an array
    std::vector<DecisionSeq> seqs;            // all questions, request order
    std::vector<ReqMedia> rmedia;
    std::vector<int> req_media;               // request -> index into rmedia (-1 = text only)
    size_t n_tok = 0, n_media_tok = 0;        // text + media positions (for batching budgets / logs)
    int n_images = 0;                         // images actually encoded (after crops)
    double prep_ms = 0;
};

class DecisionService {
public:
    // mmproj_path: "" = look next to the model; the projector is loaded on first use (or now with eager_mm) - media requests are
    // answered 400 when none exists.
    bool init(const std::string& model, const std::string& mmproj_path, int threads, std::string* err, bool eager_mm = false);
    DecisionPrepared prepare(const Json& body);                 // throws std::invalid_argument (bad request)
    // media encoders + one scoring pass over all given requests; scores[i] = per question of job i. false + err on an engine error.
    bool infer(const std::vector<DecisionPrepared*>& jobs, std::vector<std::vector<std::vector<float>>>& scores, std::string* err);
    Json format(const DecisionPrepared& p, const std::vector<std::vector<float>>& scores) const;   // array if p.batch
    std::string name() const { return eng_.gguf().get_str("general.name"); }
    bool on_gpu() const { return eng_.on_gpu(); }
    bool has_mm() const { return mm_loaded_; }
    const std::string& mmproj_path() const { return mmproj_; }
    bool has_vision() const { return mm_loaded_ && mm_.has_vision(); }
    bool has_audio() const { return mm_loaded_ && mm_.has_audio(); }
    DecisionEngine& engine() { return eng_; }
    const GgufFile& gguf() const { return eng_.gguf(); }
    const DecisionPrompter& prompter() const { return dp_; }

private:
    bool ensure_mm(std::string* err);
    DecisionEngine eng_;
    BpeTokenizer tok_;
    DecisionPrompter dp_;
    MmProj mm_;
    std::string model_, mmproj_;
    bool mm_loaded_ = false, mm_tried_ = false;
    std::mutex mm_mu_;
};

}  // namespace ferro
