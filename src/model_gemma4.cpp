// model_gemma4.cpp — see model_gemma4.h. Ported from llama.cpp src/models/gemma4.cpp (MIT).
#include "model_gemma4.h"
#include "gpu_cl.h"
#include "types.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ferro {

// ------------------------------------------------------------------ helpers
static inline float gelu_tanh(float x) {          // ggml_gelu: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
    if (x <= -10.f) return 0.f;
    if (x >= 10.f) return x;
    const float x3 = x * x * x;
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x3)));
}

static void rmsnorm(const float* x, const float* w, int n, float eps, float* y) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float s = 1.f / sqrtf((float)(ss / n) + eps);
    if (w) for (int i = 0; i < n; ++i) y[i] = x[i] * s * w[i];
    else   for (int i = 0; i < n; ++i) y[i] = x[i] * s;
}

// Q4_0 blocks (18 B: fp16 d + 16 nibble bytes) -> WMat int8 planes, streamed (no f32 temp)
// Y[T x rows] = X[T x cols] @ W^T: iGPU when the weights are resident, else per-32-block int8 on the CPU
static inline void mm(const WMat& W, const float* X, int T, float* Y, int y_ld) {
    if (W.gpu_slot >= 0 && y_ld == W.rows && gpu::gemm(W.gpu_slot, X, T, Y)) return;
    if (!W.wq_row.empty()) { gemm_q8_int8(W, X, T, Y, y_ld); return; }
    ActQ aq;
    quant_act(X, T, W.cols, aq);
    gemm_q8(W, aq, T, Y, y_ld);
}

// debug: dump a tensor as raw f32 (G4_DUMP_DIR), sequence-major like the llama.cpp graph dumps
static void g4_dump(const char* name, const float* p, size_t n) {
    static const char* dir = getenv("G4_DUMP_DIR");
    if (!dir || !p) return;
    char fn[1024];
    snprintf(fn, sizeof fn, "%s/mine_%s.bin", dir, name);
    FILE* f = fopen(fn, "wb");
    if (!f) return;
    fwrite(p, 4, n, f);
    fclose(f);
}

static WMat pack_q4_0(const uint8_t* raw, int rows, int cols, const char* name) {
    const int nb = cols / 32;
    WMat W;
    W.rows = rows; W.cols = cols; W.nb = nb; W.name = name;
    W.wq.resize((size_t)rows * cols);
    W.ws.resize((size_t)rows * nb);
    W.rs.resize((size_t)rows * nb);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* rowp = raw + (size_t)r * nb * 18;
        int8_t* qrow = W.wq.data() + (size_t)r * cols;
        float* srow = W.ws.data() + (size_t)r * nb;
        float* rrow = W.rs.data() + (size_t)r * nb;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = rowp + (size_t)b * 18;
            uint16_t dh; memcpy(&dh, blk, 2);
            const float d = fp16_to_f32(dh);
            float v[32];
            for (int j = 0; j < 16; ++j) {
                v[j]      = ((float)((blk[2 + j] & 0x0F) - 8)) * d;
                v[j + 16] = ((float)((blk[2 + j] >> 4) - 8)) * d;
            }
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(v[j]));
            const float sc = amax > 0.f ? amax / 127.f : 0.f;
            const float inv = sc > 0.f ? 1.f / sc : 0.f;
            int isum = 0;
            for (int j = 0; j < 32; ++j) {
                int q = (int)lrintf(v[j] * inv);
                if (q > 127) q = 127; if (q < -127) q = -127;
                qrow[(size_t)b * 32 + j] = (int8_t)q;
                isum += q;
            }
            srow[b] = sc;
            rrow[b] = -128.f * (float)isum * sc;
        }
    }
    return W;
}

// dequantize one row of a Q4_0 tensor (embeddings / PLE lookups)
void g4_q4_0_row(const GgufFile& g, const GGUFTensor& t, int row, float* out);   // see below
static void q4_0_row(const GgufFile& g, const GGUFTensor& t, int row, float* out) {
    const int cols = (int)t.dims[0], nb = cols / 32;
    const uint8_t* rowp = g.tensor_data(t) + (size_t)row * nb * 18;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = rowp + (size_t)b * 18;
        uint16_t dh; memcpy(&dh, blk, 2);
        const float d = fp16_to_f32(dh);
        for (int j = 0; j < 16; ++j) {
            out[b * 32 + j]        = ((float)((blk[2 + j] & 0x0F) - 8)) * d;
            out[b * 32 + j + 16]   = ((float)((blk[2 + j] >> 4) - 8)) * d;
        }
    }
}

void g4_q4_0_row(const GgufFile& g, const GGUFTensor& t, int row, float* out) { q4_0_row(g, t, row, out); }

// any GGUF tensor -> f32 (F32/F16/BF16/Q8_0/Q4_0)
static std::vector<float> to_f32(const GgufFile& g, const GGUFTensor& t) {
    size_t n = 1;
    for (auto d : t.dims) n *= d;
    std::vector<float> v(n);
    const uint8_t* p = g.tensor_data(t);
    if (t.type == T_F32) memcpy(v.data(), p, n * 4);
    else if (t.type == T_F16) for (size_t i = 0; i < n; ++i) v[i] = fp16_to_f32(((const uint16_t*)p)[i]);
    else if (t.type == T_BF16) for (size_t i = 0; i < n; ++i) v[i] = bf16_to_f32(((const uint16_t*)p)[i]);
    else if (t.type == T_Q8_0) {
        const size_t bs = 34, nb = n / 32;
        for (size_t b = 0; b < nb; ++b) {
            uint16_t dh; memcpy(&dh, p + b * bs, 2);
            const float d = fp16_to_f32(dh);
            for (int i = 0; i < 32; ++i) v[b * 32 + i] = ((const int8_t*)(p + b * bs + 2))[i] * d;
        }
    } else if (t.type == T_Q4_0) {
        const size_t nb = n / 32;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t* blk = p + b * 18;
            uint16_t dh; memcpy(&dh, blk, 2);
            const float d = fp16_to_f32(dh);
            for (int j = 0; j < 16; ++j) {
                v[b * 32 + j]      = ((float)((blk[2 + j] & 0x0F) - 8)) * d;
                v[b * 32 + j + 16] = ((float)((blk[2 + j] >> 4) - 8)) * d;
            }
        }
    } else throw std::runtime_error(t.name + ": unsupported tensor type " + std::to_string(t.type));
    return v;
}

static const GGUFTensor& need(const GgufFile& g, const std::string& name) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return *t;
}

static std::vector<float> vec_of(const GgufFile& g, const std::string& name) {
    return to_f32(g, need(g, name));
}

// GGUF [ne0=cols, ne1=rows] -> WMat (rows, cols)
static bool g4_gpu = false;            // weights resident on the iGPU (set by Gemma4Model::load)
static bool g4_defer = false;          // load() builds fused qkv/gateup matrices before uploading

static void g4_free_planes(WMat& W) {
    W.wq.clear(); W.wq.shrink_to_fit();
    W.ws.clear(); W.ws.shrink_to_fit();
    W.rs.clear(); W.rs.shrink_to_fit();
    W.wf.clear(); W.wf.shrink_to_fit();
    W.wq_row.clear(); W.wq_row.shrink_to_fit();
}

static void g4_up_free(WMat& W) {              // upload to the iGPU, then drop the host planes (4 GiB budget)
    if (W.rows <= 0 || W.wq.empty()) return;
    W.gpu_slot = gpu::upload_q8(W);
    g4_free_planes(W);
}

static WMat mat_of(const GgufFile& g, const std::string& name) {
    const GGUFTensor& t = need(g, name);
    const int cols = (int)t.dims[0], rows = t.dims.size() > 1 ? (int)t.dims[1] : 1;
    WMat W = (t.type == T_Q4_0) ? pack_q4_0(g.tensor_data(t), rows, cols, name.c_str())
                                : pack_from_f32(to_f32(g, t).data(), rows, cols, cols, name.c_str());
    if (g4_gpu && g4_defer) return W;             // caller fuses + uploads (gemma4 GPU path)
    if (g4_gpu) {
        W.gpu_slot = gpu::upload_q8(W);           // int8 DP4A resident copy
        if (W.gpu_slot >= 0) {                     // iGPU owns the weights; drop the host planes (4 GiB budget)
            W.wq.clear(); W.wq.shrink_to_fit();
            W.ws.clear(); W.ws.shrink_to_fit();
            W.rs.clear(); W.rs.shrink_to_fit();
        }
    } else {
        // CPU: keep only the row-domain int8 planes (one scale per row, VNNI madd) - 4 GiB budget, 4x
        // faster than the per-32-block path (gemm_q8) on this N100
        pack_q8_rowdomain(W);
        W.wq.clear(); W.wq.shrink_to_fit();
        W.ws.clear(); W.ws.shrink_to_fit();
        W.rs.clear(); W.rs.shrink_to_fit();
    }
    return W;                      // CPU fallback keeps block planes: gemm_q8 + ActQ (per-32-block int8)
}

bool is_gemma4(const GgufFile& g) { return g.arch == "gemma4"; }

// ------------------------------------------------------------------ load
bool Gemma4Model::load(const std::string& path, int max_ctx, std::string* err) {
    ctx = max_ctx > 0 ? max_ctx : 2048;
    try {
        if (!gguf.open_file(path.c_str(), err)) return false;
        if (!is_gemma4(gguf)) { *err = "not a gemma4 model"; return false; }
        {   // iGPU backend: the resident forward (one host sync per forward). Opt out with FERRO_LM_GPU=0.
            const char* ge = getenv("FERRO_LM_GPU");
            gpu_enabled = (!ge || atoi(ge) != 0) && gpu::init(getenv("FERRO_VERBOSE") != nullptr);
            g4_gpu = gpu_enabled;
            g4_defer = gpu_enabled;
            if (g4_gpu) fprintf(stderr, "[gemma4] iGPU: %s (resident forward)\n", gpu::device_name());
        }
        const std::string P = "gemma4.";
        n_layer  = (int)gguf.get_int(P + "block_count", 0);
        n_embd   = (int)gguf.get_int(P + "embedding_length", 0);
        n_head   = (int)gguf.get_int(P + "attention.head_count", 0);
        n_kv     = (int)gguf.get_int(P + "attention.head_count_kv", 1);
        n_vocab  = (int)gguf.get_iarr("tokenizer.ggml.token_type").size();
        n_embd_pl = (int)gguf.get_int(P + "embedding_length_per_layer_input", 0);
        swa_win  = (int)gguf.get_int(P + "attention.sliding_window", 512);
        n_ctx_train = (int)gguf.get_int(P + "context_length", 8192);
        rms_eps  = (float)gguf.get_f64(P + "attention.layer_norm_rms_epsilon", 1e-6);
        softcap  = (float)gguf.get_f64(P + "final_logit_softcapping", 0.0);
        rope_base = (float)gguf.get_f64(P + "rope.freq_base", 1e6);
        rope_base_swa = (float)gguf.get_f64(P + "rope.freq_base_swa", 1e4);
        hd_g = (int)gguf.get_int(P + "attention.key_length", 512);
        hd_s = (int)gguf.get_int(P + "attention.key_length_swa", hd_g);
        const int n_kv_shared = (int)gguf.get_int(P + "attention.shared_kv_layers", 0);
        n_kv_layers = n_layer - n_kv_shared;
        bos_id = (int)gguf.get_int("tokenizer.ggml.bos_token_id", 2);
        eos_id = (int)gguf.get_int("tokenizer.ggml.eos_token_id", 1);
        // gemma chat turns end on <turn|> (eot); the CLI stops on either
        n_stop = 1;
        stop_ids[0] = eos_id;
        const int eot = (int)gguf.get_int("tokenizer.ggml.eot_token_id", -1);
        if (eot >= 0 && eot != eos_id && n_stop < 4) stop_ids[n_stop++] = eot;

        auto is_swa = gguf.get_iarr(P + "attention.sliding_window_pattern");
        auto ffn_arr = gguf.get_iarr(P + "feed_forward_length");

        out_norm = vec_of(gguf, "output_norm.weight");
        if (gguf.tensor("rope_freqs.weight")) rope_freqs = vec_of(gguf, "rope_freqs.weight");
        tok_embd_raw = gguf.tensor("token_embd.weight");
        if (!tok_embd_raw) throw std::runtime_error("missing token_embd.weight");
        tok_embd = mat_of(gguf, "token_embd.weight");           // tied lm_head
        if (g4_gpu) g4_up_free(tok_embd);


        if (n_embd_pl > 0) {
            pl_proj_norm = vec_of(gguf, "per_layer_proj_norm.weight");
            pl_model_proj = mat_of(gguf, "per_layer_model_proj.weight");
            if (g4_gpu) g4_up_free(pl_model_proj);
            ple_tok_raw = gguf.tensor("per_layer_token_embd.weight");
            if (!ple_tok_raw) throw std::runtime_error("missing per_layer_token_embd.weight");
        }

        L.resize(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            G4Layer& l = L[il];
            const std::string B = "blk." + std::to_string(il) + ".";
            l.swa = is_swa.empty() ? true : (is_swa[il] != 0);
            l.hd  = l.swa ? hd_s : hd_g;
            l.own_kv = il < n_kv_layers;
            l.kv_src = l.own_kv ? il : (n_kv_layers - (l.swa ? 2 : 1));
            l.attn_norm       = vec_of(gguf, B + "attn_norm.weight");
            l.q_norm          = vec_of(gguf, B + "attn_q_norm.weight");
            if (l.own_kv) l.k_norm = vec_of(gguf, B + "attn_k_norm.weight");   // shared-KV layers have no k/v projections or k_norm
            l.post_attn_norm  = vec_of(gguf, B + "post_attention_norm.weight");
            l.ffn_norm        = vec_of(gguf, B + "ffn_norm.weight");
            l.post_ffw_norm   = vec_of(gguf, B + "post_ffw_norm.weight");
            l.post_norm       = vec_of(gguf, B + "post_norm.weight");
            if (gguf.tensor(B + "layer_output_scale.weight")) l.out_scale = vec_of(gguf, B + "layer_output_scale.weight");
            l.wq = mat_of(gguf, B + "attn_q.weight");
            l.wo = mat_of(gguf, B + "attn_output.weight");
            if (l.own_kv) {
                l.wk = mat_of(gguf, B + "attn_k.weight");
                if (gguf.tensor(B + "attn_v.weight")) l.wv = mat_of(gguf, B + "attn_v.weight");
            }
            l.ffn_gate = mat_of(gguf, B + "ffn_gate.weight");
            l.ffn_up   = mat_of(gguf, B + "ffn_up.weight");
            l.ffn_down = mat_of(gguf, B + "ffn_down.weight");
            if (n_embd_pl > 0) {
                l.inp_gate = mat_of(gguf, B + "inp_gate.weight");
                l.proj     = mat_of(gguf, B + "proj.weight");
            }
            if (ffn_arr.size() > (size_t)il) n_ff = (int)ffn_arr[il];
            if (g4_gpu) {   // fuse q|k|v and gate|up -> ONE GEMM per block; upload, drop the host planes
                if (l.own_kv) {
                    l.qkv = pack_concat({&l.wq, &l.wk, &l.wv}, "qkv");
                    g4_up_free(l.qkv);
                } else {
                    g4_up_free(l.wq);      // shared-KV layer: q projection only
                }
                g4_free_planes(l.wq); g4_free_planes(l.wk); g4_free_planes(l.wv);
                l.gateup = pack_concat({&l.ffn_gate, &l.ffn_up}, "gateup");
                g4_up_free(l.gateup);
                g4_free_planes(l.ffn_gate); g4_free_planes(l.ffn_up);
                g4_up_free(l.wo); g4_up_free(l.ffn_down); g4_up_free(l.inp_gate); g4_up_free(l.proj);
            }
        }

        if (g4_gpu) {                       // buffers for the resident forward (weights are already device-resident)
            g4_defer = false;
            if (!upload_gpu(err)) return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

void Gemma4Model::reset(State& s) const {
    s.n = 0;
    s.k.assign(L.size(), {});
    s.v.assign(L.size(), {});
    for (size_t il = 0; il < L.size(); ++il) {
        if (!L[il].own_kv) continue;
        s.k[il].clear();
        s.v[il].clear();
    }
}

// ------------------------------------------------------------------ forward (CPU)
bool Gemma4Model::forward(State& s, const int* ids, int T, float* logits, std::string* err) {
    if (gpu) return forward_gpu(s, ids, T, logits, err);
    try {
        const int d = n_embd, pl = n_embd_pl;
        const int pos0 = s.n;
        static const bool tr = getenv("G4_TRACE") != nullptr;
        auto TR = [&](const char* m){ if (tr) fprintf(stderr, "[g4] %s\n", m); };
        std::vector<float> x((size_t)T * d);
        TR("emb-start");
        // embeddings
        for (int t = 0; t < T; ++t)
            q4_0_row(gguf, *tok_embd_raw, ids[t], x.data() + (size_t)t * d);
        const float emb_scale = sqrtf((float)d);
        for (auto& v : x) v *= emb_scale;

        TR("emb-done");
        // per-layer embeddings [T][n_layer][pl]
        std::vector<float> ple((size_t)T * n_layer * pl, 0.f);
        if (pl > 0) {
            std::vector<float> prot((size_t)T * n_layer * pl);
            mm(pl_model_proj, x.data(), T, prot.data(), n_layer * pl);
            const float proj_scale = 1.f / sqrtf((float)d);
            std::vector<float> tokrow((size_t)n_layer * pl);
            for (int t = 0; t < T; ++t) {
                q4_0_row(gguf, *ple_tok_raw, ids[t], tokrow.data());
                const float tok_scale = sqrtf((float)pl);
                for (int i = 0; i < n_layer * pl; ++i) tokrow[i] *= tok_scale;
                for (int il = 0; il < n_layer; ++il) {
                    float* dst = ple.data() + ((size_t)t * n_layer + il) * pl;
                    const float* src = prot.data() + (size_t)t * n_layer * pl + (size_t)il * pl;
                    float tmp[4096];
                    for (int i = 0; i < pl; ++i) tmp[i] = src[i] * proj_scale;
                    rmsnorm(tmp, pl_proj_norm.data(), pl, rms_eps, dst);
                    const float* tokp = tokrow.data() + (size_t)il * pl;
                    for (int i = 0; i < pl; ++i) dst[i] = (dst[i] + tokp[i]) * 0.70710678f;
                }
            }
        }

        g4_dump("inp_scaled", x.data(), (size_t)T * d);
        g4_dump("inp_per_layer", ple.data(), ple.size());
        TR("ple-done");
        // scratch
        const int qdim_max = n_head * 512, kv_max = 512;
        std::vector<float> cur;                      // [T][d] residual stream
        std::vector<float> xn((size_t)T * d), q((size_t)T * qdim_max), k((size_t)T * kv_max), v((size_t)T * kv_max);
        std::vector<float> attn((size_t)T * qdim_max), proj_out((size_t)T * std::max(d, 12288));
        std::vector<float> scores;

        for (int il = 0; il < n_layer; ++il) {
            const G4Layer& l = L[il];
            const int hd = l.hd, nh = n_head, kvh = n_kv;
            const int qdim = nh * hd, kvdim = kvh * hd;
            std::vector<float> attn_out((size_t)T * d);

            // ---- attention
            for (int t = 0; t < T; ++t) {
                float* xn_t = xn.data() + (size_t)t * d;
                rmsnorm(x.data() + (size_t)t * d, l.attn_norm.data(), d, rms_eps, xn_t);
            }
            if (tr) fprintf(stderr, "[g4] L%d swa=%d ownkv=%d qdim=%d ff=%d\n", il, (int)l.swa, (int)l.own_kv, qdim, l.ffn_gate.rows);
            mm(l.wq, xn.data(), T, q.data(), qdim);
            if (l.own_kv) {
                mm(l.wk, xn.data(), T, k.data(), kvdim);
                if (!l.wv.empty()) mm(l.wv, xn.data(), T, v.data(), kvdim);
                else memcpy(v.data(), k.data(), (size_t)T * kvdim * 4);
            }
            // per-head norms + rope
            const float rb = l.swa ? rope_base_swa : rope_base;
            for (int t = 0; t < T; ++t) {
                const int pos = pos0 + t;
                for (int h = 0; h < nh; ++h) {
                    float* qh = q.data() + (size_t)t * qdim + (size_t)h * hd;
                    rmsnorm(qh, l.q_norm.data(), hd, rms_eps, qh);
                    for (int i = 0; i < hd / 2; ++i) {          // NEOX rotate-half (gemma4 rope type)
                        float th = powf(rb, -2.f * i / hd);
                        if (!l.swa && !rope_freqs.empty()) th /= rope_freqs[i];
                        const float ang = pos * th;
                        const float c = cosf(ang), sn = sinf(ang);
                        const float a = qh[i], bq = qh[i + hd / 2];
                        qh[i] = a * c - bq * sn;
                        qh[i + hd / 2] = a * sn + bq * c;
                    }
                }
                if (l.own_kv) {
                    for (int h = 0; h < kvh; ++h) {
                        float* kh = k.data() + (size_t)t * kvdim + (size_t)h * hd;
                        float* vh = v.data() + (size_t)t * kvdim + (size_t)h * hd;
                        rmsnorm(kh, l.k_norm.data(), hd, rms_eps, kh);
                        rmsnorm(vh, nullptr, hd, rms_eps, vh);       // V norm without weights
                        for (int i = 0; i < hd / 2; ++i) {      // NEOX rotate-half
                            float th = powf(rb, -2.f * i / hd);
                            if (!l.swa && !rope_freqs.empty()) th /= rope_freqs[i];
                            const float ang = pos * th;
                            const float c = cosf(ang), sn = sinf(ang);
                            const float a = kh[i], bk = kh[i + hd / 2];
                            kh[i] = a * c - bk * sn;
                            kh[i + hd / 2] = a * sn + bk * c;
                        }
                    }
                }
            }
            // append to the cache of the owning layer; shared layers read their source cache
            const int src = l.kv_src;
            if (l.own_kv) {
                s.k[il].insert(s.k[il].end(), k.data(), k.data() + (size_t)T * kvdim);
                s.v[il].insert(s.v[il].end(), v.data(), v.data() + (size_t)T * kvdim);
            }
            const std::vector<float>& K = s.k[src];
            const std::vector<float>& V = s.v[src];
            for (int t = 0; t < T; ++t) {
                const int p = pos0 + t;                       // absolute position of this query (cache holds 0..p)
                const int k0 = l.swa ? std::max(0, p - swa_win + 1) : 0;
                const int kn = p - k0 + 1;
                scores.resize((size_t)kn);
                for (int h = 0; h < nh; ++h) {
                    const float* qh = q.data() + (size_t)t * qdim + (size_t)h * hd;
                    float mx = -1e30f;
                    for (int j = 0; j < kn; ++j) {
                        const float* kj = K.data() + (size_t)(k0 + j) * kvdim;   // MQA: single kv head
                        float sdot = 0.f;
                        for (int i = 0; i < hd; ++i) sdot += qh[i] * kj[i];
                        scores[j] = sdot;
                        mx = std::fmax(mx, sdot);
                    }
                    float sum = 0.f;
                    for (int j = 0; j < kn; ++j) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
                    float* oh = attn.data() + (size_t)t * qdim + (size_t)h * hd;
                    for (int i = 0; i < hd; ++i) oh[i] = 0.f;
                    for (int j = 0; j < kn; ++j) {
                        const float w = scores[j] / sum;
                        const float* vj = V.data() + (size_t)(k0 + j) * kvdim;
                        for (int i = 0; i < hd; ++i) oh[i] += w * vj[i];
                    }
                }
            }
            mm(l.wo, attn.data(), T, attn_out.data(), d);
            { char nm[32]; snprintf(nm, sizeof nm, "attn_wo-%d", il); g4_dump(nm, attn_out.data(), (size_t)T * d); }

            // ---- residual + post-attention norm
            std::vector<float> r1((size_t)T * d);
            for (int t = 0; t < T; ++t) {
                float* tmp = proj_out.data() + (size_t)t * d;
                rmsnorm(attn_out.data() + (size_t)t * d, l.post_attn_norm.data(), d, rms_eps, tmp);
                const float* xt = x.data() + (size_t)t * d;
                for (int i = 0; i < d; ++i) r1[(size_t)t * d + i] = tmp[i] + xt[i];
            }

            { char nm[32]; snprintf(nm, sizeof nm, "attn_out-%d", il); g4_dump(nm, r1.data(), (size_t)T * d); }
            // ---- FFN (gelu-gated SwiGLU): norm -> gate*up -> down -> post norm -> residual
            const int ff = l.ffn_gate.rows;
            std::vector<float> g((size_t)T * ff), u((size_t)T * ff);
            for (int t = 0; t < T; ++t)
                rmsnorm(r1.data() + (size_t)t * d, l.ffn_norm.data(), d, rms_eps, xn.data() + (size_t)t * d);
            mm(l.ffn_gate, xn.data(), T, g.data(), ff);
            mm(l.ffn_up,   xn.data(), T, u.data(), ff);
            for (size_t i = 0; i < (size_t)T * ff; ++i) g[i] = gelu_tanh(g[i]) * u[i];
            mm(l.ffn_down, g.data(), T, proj_out.data(), d);
            { char nm[32]; snprintf(nm, sizeof nm, "ffn_out-%d", il); g4_dump(nm, proj_out.data(), (size_t)T * d); }
            std::vector<float> r2((size_t)T * d);
            for (int t = 0; t < T; ++t) {
                float* tmp = xn.data() + (size_t)t * d;
                rmsnorm(proj_out.data() + (size_t)t * d, l.post_ffw_norm.data(), d, rms_eps, tmp);
                const float* r1t = r1.data() + (size_t)t * d;
                for (int i = 0; i < d; ++i) r2[(size_t)t * d + i] = tmp[i] + r1t[i];
            }

            { char nm[32]; snprintf(nm, sizeof nm, "pre_ple-%d", il); g4_dump(nm, r2.data(), (size_t)T * d); }
            // ---- per-layer embedding injection
            std::vector<float> out((size_t)T * d);
            if (pl > 0) {
                std::vector<float> pg((size_t)T * pl), pp((size_t)T * d);
                mm(l.inp_gate, r2.data(), T, pg.data(), pl);
                for (auto& z : pg) z = gelu_tanh(z);
                for (int t = 0; t < T; ++t) {
                    const float* pt = ple.data() + ((size_t)t * n_layer + il) * pl;
                    for (int i = 0; i < pl; ++i) pg[(size_t)t * pl + i] *= pt[i];
                }
                mm(l.proj, pg.data(), T, pp.data(), d);
                for (int t = 0; t < T; ++t) {
                    float* tmp = xn.data() + (size_t)t * d;
                    rmsnorm(pp.data() + (size_t)t * d, l.post_norm.data(), d, rms_eps, tmp);
                    const float* r2t = r2.data() + (size_t)t * d;
                    for (int i = 0; i < d; ++i) out[(size_t)t * d + i] = tmp[i] + r2t[i];
                }
            } else out = r2;

            const float osc = l.out_scale.empty() ? 1.f : l.out_scale[0];
            for (int t = 0; t < T; ++t) {
                const float* ot = out.data() + (size_t)t * d;
                float* xt = x.data() + (size_t)t * d;
                for (int i = 0; i < d; ++i) xt[i] = ot[i] * osc;
            }
            { char nm[32]; snprintf(nm, sizeof nm, "l_out-%d", il); g4_dump(nm, x.data(), (size_t)T * d); }
        }

        TR("layers-done");
        s.n = pos0 + T;
        if (logits) {                                    // next-token logits of the LAST position
            rmsnorm(x.data() + (size_t)(T - 1) * d, out_norm.data(), d, rms_eps, xn.data());
            g4_dump("result_norm", xn.data(), (size_t)d);
            mm(tok_embd, xn.data(), 1, logits, n_vocab);
            g4_dump("result_output_pre", logits, (size_t)n_vocab);
            if (softcap > 0.f)
                for (int i = 0; i < n_vocab; ++i) logits[i] = softcap * tanhf(logits[i] / softcap);
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

}  // namespace ferro
