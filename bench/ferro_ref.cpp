// ferro_ref.cpp — reference: tokenize+embed via llama.cpp (correctness oracle + baseline timing)
//   ferro_ref model.gguf sentences.txt out.json [--time reps]
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: ferro_ref model.gguf sentences.txt out.json [--time reps]\n"); return 2; }
    int reps = 1;
    if (argc > 4 && !strcmp(argv[4], "--time")) reps = atoi(argv[5]);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab* v = llama_model_get_vocab(m);

    std::ifstream f(argv[2]);
    std::string line;
    std::vector<std::vector<llama_token>> seqs;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<llama_token> ids(8192);
        int n = llama_tokenize(v, line.c_str(), (int)line.size(), ids.data(), (int)ids.size(), true, true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        ids.resize(n);
        seqs.push_back(ids);
    }
    int n_seq = (int)seqs.size();
    int total = 0; for (auto& s : seqs) total += (int)s.size();
    fprintf(stderr, "[ref] %d seqs, %d tokens\n", n_seq, total);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = std::max(total, 1024);
    cp.n_threads = 4;
    cp.n_seq_max = 16; cp.n_outputs_max_per_seq = 1024;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    llama_context* lctx = llama_init_from_model(m, cp);
    if (!lctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_set_embeddings(lctx, true);

    llama_batch batch = llama_batch_init(std::max(total, 512), 0, 1);
    std::vector<float> out;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        // build one batch across all sequences
        batch.n_tokens = 0;
        for (int s = 0; s < n_seq; ++s) {
            for (int i = 0; i < (int)seqs[s].size(); ++i) {
                int p = batch.n_tokens++;
                batch.token[p] = seqs[s][i];
                batch.pos[p] = i;
                batch.n_seq_id[p] = 1;
                batch.seq_id[p][0] = s;
                batch.logits[p] = false;
            }
        }
        int rc = llama_encode(lctx, batch);
        if (rc) { fprintf(stderr, "encode failed %d\n", rc); return 1; }
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (r) fprintf(stderr, "[ref] rep %d: %.1f ms (%.0f tok/s)\n", r, dt*1e3, total/dt);
        out.clear();
        int dim = llama_model_n_embd_out(m);
        for (int s = 0; s < n_seq; ++s) {
            const float* e = llama_get_embeddings_seq(lctx, s);
            if (!e) { fprintf(stderr, "no embd seq %d\n", s); return 1; }
            double ss = 0; for (int i = 0; i < dim; ++i) ss += (double)e[i]*e[i];
            double inv = 1.0/(std::sqrt(ss) + 1e-12);
            for (int i = 0; i < dim; ++i) out.push_back((float)(e[i]*inv));
        }
    }
    int dim = llama_model_n_embd_out(m);
    FILE* fo = fopen(argv[3], "w");
    fprintf(fo, "{\"dim\":%d,\"embd\":[\n", dim);
    for (int s = 0; s < n_seq; ++s) {
        fprintf(fo, "[");
        for (int i = 0; i < dim; ++i) fprintf(fo, "%.8g%s", out[s*dim+i], i+1==dim?"":",");
        fprintf(fo, "]%s\n", s+1==n_seq?"":",");
    }
    fprintf(fo, "]}\n"); fclose(fo);
    fprintf(stderr, "[ref] wrote %s\n", argv[3]);
    llama_batch_free(batch);
    llama_free(lctx);
    llama_model_free(m);
    return 0;
}
