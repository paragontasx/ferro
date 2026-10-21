// sampler.h - token sampling for generation: repetition penalty -> temperature -> top-k -> top-p / min-p -> multinomial.
//   temperature <= 0 (or top_k == 1) = greedy (argmax after the repetition penalty).
// Defaults are LiquidAI's recommended generation parameters for LFM2.5 (model card): temperature 0.05, top_k 50, repetition_penalty 1.05.
// Repetition penalty follows the Hugging Face / llama.cpp convention: for a token already in the history, logit > 0 ? logit / p : logit * p.
#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace ferro {

struct SamplerParams {
    float temperature = 0.05f;
    int top_k = 50;                  // 0 = off
    float top_p = 1.0f;              // 1 = off
    float min_p = 0.0f;              // 0 = off
    float repetition_penalty = 1.05f;
    int repeat_last_n = -1;          // history window for the penalty; -1 = the whole context (prompt + generated), as Hugging Face
    uint64_t seed = 0;               // 0 = random
};

class Sampler {
public:
    explicit Sampler(const SamplerParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}
    // logits: [n_vocab], modified in place. history: all tokens of the context so far.
    int sample(float* logits, int n_vocab, const std::vector<int>& history);
    const SamplerParams& params() const { return p_; }

private:
    SamplerParams p_;
    std::mt19937_64 rng_;
    std::vector<std::pair<float, int>> cand_;
};

}  // namespace ferro
