// server.h - `ferro-embed serve`: a resident, OpenAI-compatible HTTP embedding server (EmbeddingGemma 2, text / image / audio / video).
#pragma once
#include <string>

namespace ferro {

struct ServerOptions {
    std::string model, mmproj;                 // EG2 text GGUF (+ optional mmproj for image / audio / video inputs)
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
