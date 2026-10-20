// model_lfm2.h — LiquidAI LFM2 decision models (arch "lfm2" + "lfm2.decision.*", e.g. d1-omni-600M)
// Trunk: LFM2 hybrid (centred 3-tap short-conv blocks + GQA attention blocks, bidirectional),
// SwiGLU FFN, RMSNorm. Decision head: LayerNorm transformer blocks over (trunk + type embedding),
// scorer MLP read at the option <|mask|> positions -> one logit per option.
#pragma once
#include "decision.h"
#include "gguf.h"
#include "kernels.h"
#include <memory>
#include <string>
#include <vector>

namespace ferro {

struct LfmTrunkLayer {
    bool conv = false;
    std::vector<float> attn_norm, ffn_norm;
    // short-conv block
    WMat in_proj, out_proj;            // in_proj rows = 3*d (b | c | x)
    std::vector<float> taps;           // [3][d]: tap k applied to token t-1+k
    // attention block
    WMat wqkv, wo;                     // wqkv rows = nq*hd + 2*nkv*hd
    std::vector<float> q_norm, k_norm;
    int n_head_kv = 0;
    // FFN (SwiGLU)
    WMat gateup, down;                 // gateup rows = 2*ff (gate | up)
};

struct LfmHeadLayer {
    std::vector<float> ln1_w, ln1_b, qkv_b, wo_b, ln2_w, ln2_b, up_b, down_b;
    WMat wqkv, wo, up, down;
    int n_head = 0, n_ff = 0;
};

struct LfmModel {
    int n_embd = 0, n_head = 0, head_dim = 0, n_ff = 0, n_vocab = 0;
    int n_trunk = 0, n_head_layers = 0;
    float rms_eps = 1e-5f, ln_eps = 1e-5f;
    double rope_base = 1e6;
    GgufFile gguf;
    WMat tok_embd;
    std::vector<float> out_norm, type_embd;               // type_embd [3][d]
    std::vector<float> cls_norm_w, cls_norm_b, cls_b, cls_out_w;
    float cls_out_b = 0;
    WMat cls;
    std::vector<LfmTrunkLayer> L;
    std::vector<LfmHeadLayer> H;
    RopeTable rope;
    bool load(const std::string& path, std::string* err);
};

// media prefix of a request: LM-space embeddings from the mmproj encoder
struct DecisionMediaEmbd {
    const float* embd = nullptr;   // [n_tok][n_embd]
    int n_tok = 0;
};

// one packed forward: media prefix rows first (each block once, shared by the question sequences of its
// request: media never read text, so their states do not depend on the question), then the text rows
struct DecisionBatch {
    int T = 0, Tm = 0;                       // rows, media rows (rows [0, Tm))
    std::vector<float> membd;                // [Tm][d]
    std::vector<int> ids;                    // text token ids, rows [Tm, T)
    std::vector<int> pos, lo, hi, plo, phi;  // per row: position, own key range, prefix key range (plo > phi: none)
    std::vector<int> prev, next;             // per row: conv neighbour rows (-1 = none)
    std::vector<int> type;                   // per text row: question type (head input)
    std::vector<int> hlo, hhi;               // per text row: head key range, text-row indices
    std::vector<int> marks;                  // option markers, text-row indices
    bool has_prefix() const { return Tm > 0; }
};

class DecisionEngine {
public:
    bool init(const std::string& model_path, int n_threads, std::string* err);
    // one score per marker of each sequence (raw logits, before temperature); seq.media indexes media
    void score(const std::vector<DecisionSeq>& seqs, std::vector<std::vector<float>>& scores,
               const std::vector<DecisionMediaEmbd>& media = {});
    const LfmModel& model() const { return m_; }
    const GgufFile& gguf() const { return m_.gguf; }
    bool on_gpu() const;

private:
    LfmModel m_;
    struct Impl;
    std::shared_ptr<Impl> impl_;
    void forward_cpu(const DecisionBatch& b, float* out);
    bool forward_gpu(const DecisionBatch& b, float* out);
};

// true if the GGUF at path is an LFM2 decision model (peeks metadata only)
bool is_lfm2_decision(const GgufFile& g);

}  // namespace ferro
