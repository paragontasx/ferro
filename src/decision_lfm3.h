// decision_lfm3.h — the d1-2B (LFM3-d1) half of the decision pipeline: request parsing, systemone prompt
// rendering + label groups and answer decoding. A faithful port of the LFM3_D1 path of llama.cpp
// tools/server/server-decision.cpp (parse_state / parse_questions / d1_labels / render / format_answer).
// The GGUF's "systemone" template is rendered directly (it has no control flow beyond the three question types).
#pragma once
#include "decision.h"
#include "tokenizer_bpe.h"
#include <map>
#include <string>
#include <vector>

namespace ferro {

struct Lfm3D1Ctx {
    const BpeTokenizer* tok = nullptr;
    std::map<std::string, float> temps;                 // "noul" | "choice.3_5" | ... -> temperature
    std::map<int, std::vector<double>> score_bias;      // n levels -> per-level bias
    std::vector<std::string> quoted;                    // CONTROL token spellings (>= 2 bytes) broken by d1_break
    std::string quoted_first;                           // their first bytes (prefilter, as in llama.cpp)
};

bool lfm3d1_load(Lfm3D1Ctx& c, const GgufFile& g, const BpeTokenizer& tok, std::string* err);

// one question of the request, with its rendered prompt and per-option label groups
struct Lfm3D1Built {
    std::string id;
    DecisionType type = DQ_CHOICE;
    std::vector<std::string> keys;                      // option keys in model order
    std::vector<Json> descriptions;                     // aligned with keys (may be Null)
    std::vector<std::string> codes;                     // the rendered option labels (A, B, .. / key / digit)
    std::vector<std::vector<int>> groups;               // label token groups, option order
    std::vector<int> tokens;                            // the whole prompt, tokenized
    std::vector<std::pair<int, int>> media_runs;        // (token position, row count) of each image run inside tokens
};

// parse a /v1/systemone body (one request: {state, questions}); throws std::invalid_argument with llama.cpp's messages.
// per_entry_tokens: one LM-token count per image entry (files/images of the request, in file order); the prompt gets
// one <|vision_start|>/<|vision_end|> bracketed placeholder run per entry (llama.cpp mtmd piece tokenization:
// the prefix/suffix are tokenized as separate pieces around each image run, no BOS/EOS added).
std::vector<Lfm3D1Built> lfm3d1_build(const Lfm3D1Ctx& c, const Json& body, const std::vector<int>& per_entry_tokens = {});

// the answer JSON: softmax(scores/temperature) -> noul P(true) | choice + probabilities + confidence | score (median level) + legend
Json lfm3d1_answer(const Lfm3D1Ctx& c, const Lfm3D1Built& q, const std::vector<float>& scores);

}  // namespace ferro
