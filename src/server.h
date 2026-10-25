// server.h - `ferro-embed serve`: a resident, OpenAI-compatible HTTP server.
//   * embeddings: EmbeddingGemma 2, text / image / audio / video        (--model [--mmproj])
//   * chat / completions: LiquidAI LFM2 / LFM2.5 language models, streaming (--lm)
//   * decisions: LiquidAI d1-omni typed answers over text / image / audio states, llama.cpp's POST /v1/systemone contract (--d1 [--d1-mmproj])
// Either or both can be loaded; they share the iGPU through one worker thread.
#pragma once
#include <string>
#include <vector>

namespace ferro {

struct ServerOptions {
    std::string model, mmproj;                 // EG2 text GGUF (+ optional mmproj for image / audio / video inputs); empty = no embeddings
    std::string lm;                            // LFM2 / LFM2.5 language-model GGUF (chat + completions); empty = no generation
    std::string d1, d1_mmproj;                 // d1-omni decision GGUF (+ its mmproj for image / audio states; found next to the model when omitted); empty = no /v1/systemone
    int d1_batch_tokens = 2048;                // queued decision requests share one forward pass up to this many tokens
    std::string lm_name;                       // id reported for the language model (default: file name)
    int lm_ctx = 4096;                         // context per sequence slot; the KV cache is allocated up front (LFM2.5-1.2B: ~25 MB per 1k tokens per slot)
    int prefix_cache_mb = 128;                 // LM prompt-prefix cache budget (device memory, LRU); 0 = off. A snapshot costs ~25 KB per cached token (LFM2.5, 6 attention layers)
    std::string snapshot_dir;                  // --snapshot-dir: persist prompt-prefix snapshots here (survive restarts; also the spill target when the device cache evicts)
    int snapshot_mb = 2048;                    // disk budget for the snapshot directory
    int block_cache = 0;                       // --block-cache N: store the prompt cache as N-token blocks (0 = whole-prefix snapshots)                    // disk budget for the snapshot directory (oldest files go first)
    bool plan_only = false;                    // `ferro-embed plan`: print the memory plan and exit
    bool no_memory_check = false;              // start even when the plan says the configuration does not fit the container limit
    int kv_dtype = 0;                           // --kv-dtype: 0 f32, 1 int8 (+ one fp32 scale per 32 values, 3.6x smaller, small accuracy cost), 2 f16 (half storage, fp32 arithmetic, 1.8x smaller)
    int spec_draft = 6;                        // speculative decoding (prompt lookup, verified exactly): longest draft per step; 0 = off. Streams start at 3 (a 4-row window costs ~1.3x a plain step) and use up to N once >85% of their drafts are accepted. Request field "draft": false opts out
    int thinking_budget = -1;                  // default cap on reasoning tokens per reply (-1 = none); the reply field `thinking_budget` overrides it
    int prefill_slice_ms = 250;                // while other work waits (decode streams, d1, embeddings) a long prompt is prefilled in slices of about this many ms
    int decode_share = 40;                     // percent of the time decode steps get while a prompt is filling (decode runs ~slice*share/(100-share) ms after each slice)
    int prefix_min_tokens = 32;                // do not cache prefixes shorter than this (the saving would not pay for the copy)
    bool prefix_tail = true;                   // --no-prefix-tail sets false: chats with several messages or tools also leave a snapshot at the end of the prompt (the next turn of an agent loop extends exactly that prompt)
    int lm_slots = 4;                          // concurrent generations (decode of all active slots runs as one batched pass)
    std::string reasoning_format = "deepseek"; // "deepseek": <think>..</think> -> reasoning_content; "none": keep it inline in content
    float repeat_penalty = 0.f;                // --repeat-penalty: server default for repetition_penalty (<= 0: the sampler's 1.05, LiquidAI's recommendation)
    int repeat_last_n = -1;                    // --repeat-last-n: default penalty window (-1 whole context, 0 off, N = last N tokens); request field repeat_last_n / penalty_last_n
    bool penalize_prompt = true;               // --penalty-scope generated sets false: prompt tokens are never penalized (agents copy ids / paths / JSON keys); request field penalize_prompt
    std::string host = "127.0.0.1";            // bind address (use 0.0.0.0 to expose it)
    int port = 8080;
    std::vector<std::string> api_keys;         // repeatable --api-key; each key may be "label: key"
    std::string api_key_file;                  // --api-key-file: one "label: key" (or bare key) per line, '#' comments; re-read on SIGHUP
    bool metrics_open = false;                 // --metrics-open: /metrics without a key (it needs one when keys are set)
    std::string api_key;                       // non-empty: require "Authorization: Bearer <key>" (everything but /health)
    std::string model_name;                    // id reported by /v1/models and echoed in responses (default: GGUF general.name)
    int threads = 2;
    int max_batch_tokens = 4096;               // text requests that arrive together share forward passes up to this many tokens
    int batch_wait_ms = 4;                     // how long the scheduler waits for in-flight requests to join a batch
    int max_queue = 64;                        // queued jobs beyond this are answered 503
    int max_connections = 256;
    size_t max_body_mb = 512;                  // base64 media inflates 4/3; a 30 s clip is ~1 MB, a 1080p PNG ~5 MB
    float video_fps = 4.0f;
};

// Blocks until SIGINT / SIGTERM. Returns a process exit code.
int run_server(const ServerOptions& o);

}  // namespace ferro
