// lm_cli.h - `ferro-embed generate`: one-shot text generation with an LFM2 / LFM2.5 language model (reference + benchmarking tool).
#pragma once
#include "sampler.h"
#include <string>

namespace ferro {

struct GenArgs {
    std::string lm, prompt, messages_file, dump_logits, system, logits_batch;   // logits_batch: JSON array of raw prompts -> <dump_logits>.<i>.bin (next-token logits of each)
    int n_predict = 256, ctx = 8192, threads = 2;
    SamplerParams sp;
    bool raw = false;          // --prompt is tokenized as given (BOS prepended) instead of being wrapped in the chat template
    bool print_ids = false;    // list prompt + generated token ids on stderr
    bool quiet = false;        // no streamed text on stdout
    bool gpu = true;
};

int run_generate(const GenArgs& a);

}  // namespace ferro
