// server.h - `ferro-embed serve`: a resident, OpenAI-compatible HTTP server.
//   * embeddings: EmbeddingGemma 2, text / image / audio / video        (--model [--mmproj])
//   * chat / completions: LiquidAI LFM2 / LFM2.5 language models, streaming (--lm)
//   * decisions: LiquidAI d1-omni typed answers over text / image / audio states, llama.cpp's POST /v1/systemone contract (--d1 [--d1-mmproj])
// Either or both can be loaded; they share the iGPU through one worker thread.
#pragma once
#include <string>

namespace ferro {

struct ServerOptions {
    std::string model, mmproj;                 // EG2 text GGUF (+ optional mmproj for image / audio / video inputs); empty = no embeddings
    std::string lm;                            // LFM2 / LFM2.5 language-model GGUF (chat + completions); empty = no generation
    std::string d1, d1_mmproj;                 // d1-omni decision GGUF (+ its mmproj for image / audio states; found next to the model when omitted); empty = no /v1/systemone
    int d1_batch_tokens = 2048;                // queued decision requests share one forward pass up to this many tokens
    std::string lm_name;                       // id reported for the language model (default: file name)
    int lm_ctx = 4096;                         // context per sequence slot; the KV cache is allocated up front (LFM2.5-1.2B: ~25 MB per 1k tokens per slot)
    int lm_slots = 4;                          // concurrent generations (decode of all active slots runs as one batched pass)
    std::string reasoning_format = "deepseek"; // "deepseek": <think>..</think> -> reasoning_content; "none": keep it inline in content
    std::string host = "127.0.0.1";            // bind address (use 0.0.0.0 to expose it)
    int port = 8080;
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
