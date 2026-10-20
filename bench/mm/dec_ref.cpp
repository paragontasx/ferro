// llama.cpp oracle for the d1-omni decision trunk with a media prefix: one sequence = media embeddings + text tokens,
// in one mixed ubatch (as llama-server does). Dumps every trunk layer output (l_out-N, all rows) and prints the
// decision scores of the given marker rows.
// usage: dec_ref MODEL MMPROJ MEDIA "tok tok ..." "marker marker ..." layers.bin [threads]
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static FILE* g_dump = nullptr;
static bool cb(struct ggml_tensor* t, bool ask, void*) {
    const bool want = strncmp(t->name, "l_out-", 6) == 0 || strcmp(t->name, "result_norm") == 0;
    if (ask) return want;
    if (want && g_dump) {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
        fwrite(v.data(), 4, v.size(), g_dump);
        fprintf(stderr, "dump %s [%lld x %lld]\n", t->name, (long long)t->ne[0], (long long)t->ne[1]);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s MODEL MMPROJ MEDIA TOKENS MARKERS layers.bin [threads]\n", argv[0]); return 1; }
    const int nth = argc > 7 ? atoi(argv[7]) : 2;
    std::vector<int> toks, marks;
    { std::istringstream ss(argv[4]); int x; while (ss >> x) toks.push_back(x); }
    { std::istringstream ss(argv[5]); int x; while (ss >> x) marks.push_back(x); }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 4096; cp.n_ubatch = 4096; cp.n_threads = nth; cp.n_threads_batch = nth;
    cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.n_outputs_max = cp.n_batch; cp.n_outputs_max_per_seq = 1;
    if (getenv("REF_NO_FA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    llama_context* ctx = llama_init_from_model(model, cp);
    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = false; mcp.n_threads = nth; mcp.warmup = false;
    mtmd_context* mctx = mtmd_init_from_file(argv[2], model, mcp);
    auto bw = mtmd_helper_bitmap_init_from_file(mctx, argv[3], false, mtmd_helper_init_opt_default());
    const mtmd_bitmap* bm = bw.bitmap;
    std::string mk = mtmd_default_marker();
    mtmd_input_text txt{mk.c_str(), mk.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bms[1] = {bm};
    if (mtmd_tokenize(mctx, chunks, &txt, bms, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 2; }
    const int n_embd = llama_model_n_embd_inp(model);
    std::vector<float> me;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
        if (mtmd_encode_chunk(mctx, ch) != 0) return 3;
        const size_t n = mtmd_input_chunk_get_n_tokens(ch);
        const float* e = mtmd_get_output_embd(mctx);
        me.insert(me.end(), e, e + n * n_embd);
    }
    const int nm = (int)(me.size() / n_embd);
    if (getenv("REF_NO_MEDIA")) me.clear();
    const int nmu = (int)(me.size() / n_embd);
    llama_batch_ext* b = llama_batch_ext_init(ctx);
    for (int i = 0; i < nmu; ++i) {
        llama_embd e{me.data() + (size_t)i * n_embd, 1, (size_t)n_embd};
        int idx = llama_batch_ext_add_embd(b, 0, e);
        llama_pos p = i;
        llama_batch_ext_set_pos(b, idx, &p);
    }
    std::vector<int> out_idx;
    for (size_t i = 0; i < toks.size(); ++i) {
        int idx = llama_batch_ext_add_token(b, 0, toks[i]);
        llama_pos p = nmu + (int)i;
        llama_batch_ext_set_pos(b, idx, &p);
        llama_batch_ext_set_output_embd(b, idx, true);
        out_idx.push_back(idx);
    }
    g_dump = fopen(argv[6], "wb");
    int r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
    fclose(g_dump); g_dump = nullptr;
    if (r != 0) { fprintf(stderr, "decode failed %d\n", r); return 4; }
    const int n_out = llama_model_n_embd_out(model);
    printf("media %d (used %d) text %zu n_embd_out %d\n", nm, nmu, toks.size(), n_out);
    for (int mk : marks) {
        const float* e = llama_get_embeddings_ith(ctx, out_idx[mk]);
        printf("marker %d:", mk);
        for (int j = 0; j < n_out; ++j) printf(" %.5f", e[j]);
        printf("\n");
    }
    return 0;
}
