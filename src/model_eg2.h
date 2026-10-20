// model_eg2.h — EmbeddingGemma-2 (arch "gemma-embedding2") engine model
#pragma once
#include "gguf.h"
#include "kernels.h"
#include <memory>
#include <string>
#include <vector>

namespace ferro {

struct EG2Layer {
    WMat wqkv;          // fused rows = nq*hd + 2*nkv*hd, cols = n_embd
    WMat wo;            // rows = n_embd, cols = nq*hd
    WMat ffn_gateup;    // fused rows = 2*n_ff (up first, gate second)
    WMat ffn_down;
    WMat ple_gate;      // per_layer_inp_gate: rows = n_pl, cols = n_embd
    WMat ple_proj;      // rows = n_embd, cols = n_pl
    std::vector<float> attn_norm, q_norm, k_norm, attn_post_norm;
    std::vector<float> ffn_norm, ffn_post_norm, ple_post_norm;
    float out_scale;
    // per-layer geometry (SWA vs full differ)
    int n_head, n_head_kv, head_dim, n_rot;
    bool is_swa;
};

struct EG2Model {
    // geometry
    int n_layer = 0, n_embd = 0, n_ff = 0, n_pl = 0, n_embd_out = 0, n_vocab = 0;
    float rms_eps = 1e-6f;
    int swa_window = 0;
    double rope_base = 1e6, rope_base_swa = 1e4;
    std::vector<bool> is_swa;

    GgufFile gguf;
    WMat tok_embd;
    std::vector<float> plm_proj;                 // [n_pl*n_layer][n_embd] f32 (scaled)
    std::vector<float> plm_norm;                 // [n_pl]
    std::vector<float> output_norm;              // [n_embd]
    WMat output;                                 // rows = n_embd_out, cols = n_embd
    std::vector<EG2Layer> layers;
    RopeTable rope_full, rope_swa;  // one each (per head_dim)

    bool load(const std::string& path, std::string* err);
    int max_pos_table() const { return 4096; }
};

// ---- forward ----
class Engine {
public:
    bool init(const std::string& model_path, int n_threads, std::string* err);

    // encode packed token stream: T tokens, seg[t] = sequence id (contiguous from 0),
    // pos[t] = position within segment. out: [n_seq][n_embd_out] (L2-normalized).
    // buffer out must hold n_seq * n_embd_out floats.
    void encode(const int* ids, const int* seg, const int* pos, int T, int n_seq, float* out);

    int dim_out() const { return m_.n_embd_out; }
    int dim_in() const { return m_.n_embd; }
    const EG2Model& model() const { return m_; }

    // int8 activation path toggle (for kernel experiments)
    void set_use_int8(bool b) { use_int8_ = b; }

private:
    EG2Model m_;
    bool use_int8_ = false;
    struct Impl;
    std::shared_ptr<Impl> impl_;
    std::shared_ptr<Impl> impl_gpu_;      // buffers for the concurrent iGPU half of a split batch
    double gpu_share_ = 0.78;             // fraction of tokens sent to the iGPU (adapted per call)
    // mode: 0 = auto (GPU layer loop if available), 1 = CPU only, 2 = GPU (falls back to CPU on failure)
    void encode_impl(Impl& in, const int* ids, const int* seg, const int* pos, int T, int n_seq, float* out, int mode);
};

} // namespace ferro
