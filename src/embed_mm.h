// embed_mm.h — multimodal EmbeddingGemma 2: an ordered list of text / image / audio / video items -> one 768-d embedding.
// Sequence layout (identical to llama.cpp mtmd + the EmbeddingGemma 2 graph):
//   BOS, [text tokens | <|image> image-embeddings <image|> | <|audio> audio-embeddings <audio|>]..., EOS
// Video = ffmpeg frames (default 4 fps) as consecutive images after the text "Video:", with "[XmY.YYs]" text every 5 s.
#pragma once
#include "gguf.h"
#include "mm_gemma4.h"
#include "model_eg2.h"
#include "tokenizer.h"
#include <memory>
#include <string>
#include <vector>

namespace ferro {

struct EmbedItem {
    enum Kind { TEXT, IMAGE, AUDIO, VIDEO } kind = TEXT;
    std::string value;                 // TEXT: the text; others: file path
    std::vector<uint8_t> bytes;        // optional in-memory media (takes precedence over the path)
};
struct MmEmbedOptions {
    float video_fps = 4.0f;            // llama.cpp --video-fps default
    int video_ts_interval_ms = 5000;   // llama.cpp --video-timestamp-interval default
    int max_image_tokens = 0;          // 0 = model default (1120)
};

class MmEmbedder {
public:
    bool init(const std::string& model_path, const std::string& mmproj_path, int threads, std::string* err);
    // out: L2-normalized embedding [dim_out]; n_tokens (optional) receives the sequence length
    bool embed(const std::vector<EmbedItem>& items, const MmEmbedOptions& opt, std::vector<float>& out, std::string* err,
               int* n_tokens = nullptr);
    int dim_out() const { return eng_.dim_out(); }
private:
    Engine eng_;
    GgufFile tokf_;
    FerroTokenizer tok_;
    Gemma4Proj proj_;
};

// ffmpeg frame source: decodes a video (file path or in-memory bytes) to RGB frames at fps_target
struct VideoFrames {
    int width = 0, height = 0;
    float fps = 0;
    std::vector<std::vector<uint8_t>> rgb;     // each width*height*3
};
bool decode_video_frames(const std::string& path, const std::vector<uint8_t>& bytes, float fps_target, VideoFrames& out, std::string* err);

}  // namespace ferro
