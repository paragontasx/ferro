// mm_ref.cpp — llama.cpp oracle for media embeddings: mmproj encode of one media file -> f32 dump.
// Output: int32 n_chunks, then per media chunk: int32 n_tok, int32 n_embd, f32 data[n_tok*n_embd].
// Usage: mm_ref TEXT_MODEL.gguf MMPROJ.gguf media.file out.bin [n_threads]
#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: mm_ref model mmproj media out.bin [threads]\n"); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.n_threads = argc > 5 ? atoi(argv[5]) : 2;
    cp.warmup = false;
    cp.print_timings = false;
    if (getenv("REF_NO_FA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mtmd_context* ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx) return 3;
    auto bw = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { fprintf(stderr, "bitmap failed\n"); return 4; }
    std::string text = mtmd_default_marker();
    mtmd_input_text it{text.c_str(), text.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bms[1] = {bw.bitmap};
    if (mtmd_tokenize(ctx, chunks, &it, bms, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 5; }
    FILE* f = fopen(argv[4], "wb");
    int32_t nmedia = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i)
        if (mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, i)) != MTMD_INPUT_CHUNK_TYPE_TEXT) ++nmedia;
    fwrite(&nmedia, 4, 1, f);
    const int32_t n_embd = llama_model_n_embd_inp(model);
    if (getenv("REF_BATCH")) {
        // the llama-server path: mtmd_batch_* (encode_mtmd_chunk in server-context.cpp)
        mtmd_batch* mb = mtmd_batch_init(ctx);
        std::vector<const mtmd_input_chunk*> media;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
            int r = mtmd_batch_add_chunk(mb, ch);
            fprintf(stderr, "batch add chunk %zu -> %d\n", i, r);
            media.push_back(ch);
        }
        if (mtmd_batch_encode(mb) != 0) { fprintf(stderr, "batch encode failed\n"); return 7; }
        for (auto* ch : media) {
            int32_t n = (int32_t)mtmd_input_chunk_get_n_tokens(ch);
            const float* e = mtmd_batch_get_output_embd(mb, ch);
            if (!e) { fprintf(stderr, "no batch output\n"); return 8; }
            fwrite(&n, 4, 1, f); fwrite(&n_embd, 4, 1, f); fwrite(e, 4, (size_t)n * n_embd, f);
        }
        fclose(f);
        return 0;
    }
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) { fprintf(stderr, "chunk %zu: text\n", i); continue; }
        if (mtmd_encode_chunk(ctx, ch) != 0) { fprintf(stderr, "encode failed\n"); return 6; }
        int32_t n = (int32_t)mtmd_input_chunk_get_n_tokens(ch);
        fprintf(stderr, "chunk %zu: media, %d tokens\n", i, n);
        const float* e = mtmd_get_output_embd(ctx);
        fwrite(&n, 4, 1, f); fwrite(&n_embd, 4, 1, f);
        fwrite(e, 4, (size_t)n * n_embd, f);
    }
    fclose(f);
    return 0;
}
