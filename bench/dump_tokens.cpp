// dump_tokens.cpp — reference tokenizer via llama.cpp API: one sentence (line) per output line of ids
#include "llama.h"
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: dump_tokens model.gguf sentences.txt\n"); return 2; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab* v = llama_model_get_vocab(m);
    std::ifstream f(argv[2]);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<llama_token> ids(4096);
        int n = llama_tokenize(v, line.c_str(), (int)line.size(), ids.data(), (int)ids.size(),
                               /*add_special*/ true, /*parse_special*/ true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        for (int i = 0; i < n; ++i) printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
    }
    llama_model_free(m);
    return 0;
}
