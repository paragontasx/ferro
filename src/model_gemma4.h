// model_gemma4.h - Google DeepMind Gemma 4 (arch "gemma4"), text path only.
//
// Ported from llama.cpp src/models/gemma4.cpp (MIT) — the oracle this engine validates against.
// Architecture (E2B as shipped by unsloth QAT): 35 decoder layers, n_embd 1536, 8 query heads with
// n_head_kv 1 (MQA), head_dim 512 on global layers / 256 on sliding-window layers, SwiGLU->GELU FFN
// (ffn 6144, 12288 on the last layer), RMSNorm everywhere (q/k per-head norms, V norm WITHOUT weights),
// attention scores q.k with scale 1.0 (no 1/sqrt(d)), hybrid SWA(512)/global pattern with the last
// layer global, proportional RoPE on global layers (rope_freqs [hd/2]), and Per-Layer Embeddings:
//   ple[t,il] = rmsnorm(per_layer_model_proj @ emb[t] * 1/sqrt(d), 256) + ple_tok_embd[t] * sqrt(256), * 1/sqrt(2)
//   layer: h' = h + rmsnorm(gelu(inp_gate @ h) * ple[il], post_norm) @ proj ; h'' = h' * layer_output_scale
// 20 of the 35 layers share KV caches: a shared layer reuses the cache of layer 13 (SWA) or 14 (global),
// per llama.cpp's kv mapping (n_layer_kv_from_start - (swa ? 2 : 1)).
// Final: rmsnorm -> tied lm_head -> logits softcapped at 30.
#pragma once
#include "gguf.h"
#include "kernels.h"
#include <string>
#include <vector>

namespace ferro {

struct G4Layer {
    bool swa = true;          // sliding-window layer (head_dim_swa) vs global (head_dim)
    bool own_kv = true;       // owns k/v projections + cache
    int kv_src = -1;          // shared layers: index of the layer whose cache is reused
    int hd = 0;               // head dim for this layer (512 global, 256 swa)
    std::vector<float> attn_norm, q_norm, k_norm, post_attn_norm, ffn_norm, post_ffw_norm, post_norm, out_scale;
    WMat wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down, inp_gate, proj;
};

struct Gemma4Model {
    int n_layer = 0, n_embd = 0, n_head = 0, n_kv = 1, n_vocab = 0, n_ff = 0, n_embd_pl = 0;
    int swa_win = 512, n_kv_layers = 0, n_ctx_train = 0;
    float rms_eps = 1e-6f, softcap = 0.f, rope_base = 1e6f, rope_base_swa = 1e4f;
    int bos_id = 2, eos_id = 1;
    int stop_ids[4] = {1, -1, -1, -1};   // eos + end-of-turn
    int n_stop = 1;
    GgufFile gguf;
    WMat tok_embd;                 // tied LM head, int8 planes
    WMat pl_model_proj;            // [n_embd_pl * n_layer][n_embd] f32-origin (small)
    std::vector<float> pl_proj_norm, out_norm, rope_freqs;
    std::vector<G4Layer> L;
    const GGUFTensor* tok_embd_raw = nullptr;   // Q4_0 rows, dequantized on demand (embeddings)
    const GGUFTensor* ple_tok_raw = nullptr;    // Q4_0 rows [n_embd_pl*n_layer] per token
    bool load(const std::string& path, int max_ctx, std::string* err);

    struct State {
        int n = 0;                                  // tokens in the caches
        std::vector<std::vector<float>> k, v;       // per own-kv layer, n * n_kv*hd floats
    };
    void reset(State& s) const;
    // Run T tokens starting at position s.n, extend the caches, write [T][n_vocab] logits (may be null).
    bool forward(State& s, const int* ids, int T, float* logits, std::string* err);
};

bool is_gemma4(const GgufFile& g);   // arch gemma4

}  // namespace ferro
