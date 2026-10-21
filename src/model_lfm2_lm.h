// model_lfm2_lm.h - LiquidAI LFM2 / LFM2.5 *generative* models (arch "lfm2", causal): LFM2.5-1.2B-Thinking, -Instruct, ...
//
// Hybrid trunk of 10 short-conv blocks and 6 GQA attention blocks (head_count_kv[i] == 0 marks a conv block), SwiGLU FFN, RMSNorm,
// tied LM head (token_embd). Unlike the d1 decision trunk in model_lfm2.h this one is CAUSAL:
//   conv block:  y = out_proj( c * causal_depthwise_conv3( b * x ) ),  (b | c | x) = in_proj(norm(h)),  taps on tokens t-2, t-1, t
//   attn block:  q/k RMSNorm per head -> RoPE (neox) -> causal GQA attention over the KV cache
// Per-sequence state: the last two `b*x` rows of every conv block + K/V of every attention block.
#pragma once
#include "gguf.h"
#include "kernels.h"
#include <memory>
#include <string>
#include <vector>

namespace ferro {

struct LmLayer {
    bool conv = false;
    std::vector<float> attn_norm, ffn_norm;
    WMat in_proj, out_proj;            // conv block: in_proj rows = 3*d (b | c | x)
    std::vector<float> taps;           // [3][d]: tap k multiplies token t-2+k
    WMat wqkv, wo;                     // attention block: wqkv rows = nq*hd + 2*nkv*hd
    std::vector<float> q_norm, k_norm; // [hd]
    WMat gateup, down;                 // gateup rows = 2*ff (gate | up)
};

struct LmModel {
    int n_layer = 0, n_embd = 0, n_head = 0, n_kv = 0, head_dim = 0, n_ff = 0, n_vocab = 0, n_ctx_train = 0;
    float rms_eps = 1e-5f;
    double rope_base = 1e6;
    int bos_id = 1, eos_id = 7;
    GgufFile gguf;
    WMat tok_embd;                     // [n_vocab][d]: embedding gather and (tied) LM head
    std::vector<float> out_norm;       // token_embd_norm.weight
    std::vector<LmLayer> L;
    RopeTable rope;
    int n_conv_layers = 0, n_attn_layers = 0;
    bool load(const std::string& path, int max_ctx, std::string* err);
};
bool is_lfm2_lm(const GgufFile& g);    // arch lfm2, causal, no decision head

// state of one generation stream
struct LmSeq {
    int n_past = 0;                                  // tokens whose state is in the caches
    std::vector<std::vector<float>> k, v;            // [attention layer] host caches, n_past * n_kv * hd floats (CPU path)
    std::vector<std::vector<float>> conv;            // [conv layer] 2 * d floats: bx of tokens t-2, t-1 (zeros at the start)
    int gpu_slot = -1;                               // device-resident state slot (iGPU path), -1 = none
};

class LmEngine {
public:
    struct Impl;
    // n_slots: number of concurrent sequences with device-resident state (KV cache for max_ctx tokens each + conv state)
    bool init(const std::string& path, int n_threads, int max_ctx, std::string* err, int n_slots = 1);
    int n_slots() const { return n_slots_; }
    void release(LmSeq& s);                          // give the sequence's device slot back (no-op on CPU)
    const LmModel& model() const { return m_; }
    int n_vocab() const { return m_.n_vocab; }
    int max_ctx() const { return max_ctx_; }
    void reset(LmSeq& s) const;
    // Advance every sequence by its tokens (toks[i][0..n[i]) are appended after s[i]->n_past) and return the logits of each sequence's
    // LAST token: logits[i * n_vocab ...]. Prefill (n large) and decode (n == 1) steps can be mixed in one call.
    bool step(const std::vector<LmSeq*>& seqs, const std::vector<const int*>& toks, const std::vector<int>& n, float* logits, std::string* err);
    bool on_gpu() const;

private:
    LmModel m_;
    int max_ctx_ = 0, n_slots_ = 1;
    std::shared_ptr<Impl> impl_;
    bool forward_gpu(const std::vector<LmSeq*>& seqs, const std::vector<const int*>& toks, const std::vector<int>& n, float* logits, std::string* err);
    void forward_cpu(const std::vector<LmSeq*>& seqs, const std::vector<const int*>& toks, const std::vector<int>& n, float* logits);
};

}  // namespace ferro
