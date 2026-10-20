// decision.h — TypeSafe "systemone" decision protocol for LFM2 d1-omni (text, image and audio states).
// Native port of llama.cpp tools/server/server-decision.cpp (LFM2_D1_OMNI path) and the
// GGUF "systemone" jinja template: request parsing, prompt rendering + token budgets,
// calibrated answers. One sequence per question; the model scores one <|mask|> per option.
// Media (images, or one audio clip) are a prefix of every question sequence of the request.
#pragma once
#include "gguf.h"
#include "json.h"
#include "tokenizer_bpe.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ferro {

enum DecisionType { DQ_CHOICE = 0, DQ_SCORE = 1, DQ_NOUL = 2 };   // = model output column

struct DecisionOption { std::string key; Json description; };
struct DecisionQuestion {
    std::string id;
    DecisionType type;
    Json instructions;
    std::vector<DecisionOption> options;   // in model output order
};
struct DecisionRequest {
    Json state;                                  // media parts removed
    std::vector<DecisionQuestion> questions;
    std::vector<std::vector<uint8_t>> files;     // encoded media (images or one audio), in request order
};
// what the media of a request turned into (filled by the caller after decoding/encoding)
struct DecisionMediaInfo {
    int n_tokens = 0;        // media positions in each question sequence (prefix)
    bool is_audio = false;
    int n_files = 0;
};
// one model sequence (= one question)
struct DecisionSeq {
    std::vector<int> tokens;    // text tokens (after the media prefix)
    std::vector<int> markers;   // positions of the option <|mask|> tokens within tokens, in option order
    DecisionType type;
    int media = -1;             // index of the media prefix (DecisionEngine::score media list), -1 = none
    int n_media = 0;            // media prefix length
};

class DecisionPrompter {
public:
    static const size_t MAX_MEDIA_FILES = 8;   // DECISION_MAX_IMAGES
    bool init(const GgufFile& g, const BpeTokenizer* tok, std::string* err);
    DecisionRequest parse(const Json& body) const;                       // throws std::invalid_argument
    DecisionSeq build(const DecisionRequest& req, const DecisionQuestion& q, const DecisionMediaInfo& media = {}) const;
    // has_media: d1-omni image/audio answers are not calibrated (temperature 1), as in server-decision format_answer
    Json answer(const DecisionQuestion& q, const std::vector<float>& scores, bool has_media = false) const;
    std::string render_debug(const DecisionRequest& req, const DecisionQuestion& q, const DecisionMediaInfo& media = {}) const;

private:
    std::vector<std::string> render_pieces(const DecisionRequest& req, const DecisionQuestion& q, const DecisionMediaInfo& media) const;
    float temperature(const DecisionQuestion& q) const;
    const BpeTokenizer* tok_ = nullptr;
    std::map<std::string, float> temps_;
    std::map<size_t, std::vector<float>> score_bias_;
};

}  // namespace ferro
