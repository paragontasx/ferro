// decision.h — TypeSafe "systemone" decision protocol for LFM2 d1-omni (text, image and audio states).
// Native port of llama.cpp tools/server/server-decision.cpp (LFM2_D1_OMNI path) and the
// GGUF "systemone" jinja template: request parsing, prompt rendering + token budgets,
// calibrated answers. One sequence per question; the model scores one <|mask|> per option.
// Media (images, or one audio clip) are a prefix of every question sequence of the request.
#pragma once
#include "gguf.h"
#include "json.h"
#include "tokenizer_bpe.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ferro {

enum DecisionType { DQ_CHOICE = 0, DQ_SCORE = 1, DQ_NOUL = 2 };   // = model output column

struct DecisionOption { std::string key; Json description; };
struct DecisionQuestion {
    std::string id;
    DecisionType type;
    Json instructions;
    std::vector<DecisionOption> options;   // in model output order
};
// A region of an image, so callers can send the whole frame plus a detector box instead of encoding a crop themselves.
//   {"x":..,"y":..,"w":..,"h":..} or {"x1":..,"y1":..,"x2":..,"y2":..}; "unit": "px" (default) | "norm" (0..1 of the width/height);
//   "pad": fraction of the box size added on every side (default 0). The box is clipped to the image.
struct CropSpec {
    double x = 0, y = 0, w = 0, h = 0;   // x,y = top-left; w,h = size (already converted from x1,y1,x2,y2)
    bool norm = false;
    double pad = 0;
};
// what one entry of "files" / one image_url part turns into: the full image and/or one image per crop
struct FileSpec {
    bool full = true;
    std::vector<CropSpec> crops;
    int n_images() const { return (full ? 1 : 0) + (int)crops.size(); }
};
// crops one RGB image; returns false (and sets err) when the box misses the image. out is ox x oy, row-major RGB.
bool crop_rgb(const uint8_t* rgb, int nx, int ny, const CropSpec& c, std::vector<uint8_t>& out, int& ox, int& oy, std::string* err);

struct DecisionRequest {
    Json state;                                  // media parts removed
    std::vector<DecisionQuestion> questions;
    std::vector<std::vector<uint8_t>> files;     // encoded media (images or one audio), in request order
    std::vector<FileSpec> file_specs;            // parallel to files (audio: always the default)
};
// what the media of a request turned into (filled by the caller after decoding/encoding)
struct DecisionMediaInfo {
    int n_tokens = 0;        // media positions in each question sequence (prefix)
    bool is_audio = false;
    int n_files = 0;
};
// one model sequence (= one question)
struct DecisionSeq {
    std::vector<int> tokens;    // text tokens (after the media prefix)
    std::vector<int> markers;   // positions of the option <|mask|> tokens within tokens, in option order
    DecisionType type;
    int media = -1;             // index of the media prefix (DecisionEngine::score media list), -1 = none
    int n_media = 0;            // media prefix length
};

class DecisionPrompter {
public:
    static const size_t MAX_MEDIA_FILES = 8;   // DECISION_MAX_IMAGES
    bool init(const GgufFile& g, const BpeTokenizer* tok, std::string* err);
    DecisionRequest parse(const Json& body) const;                       // throws std::invalid_argument
    DecisionSeq build(const DecisionRequest& req, const DecisionQuestion& q, const DecisionMediaInfo& media = {}) const;
    // has_media: d1-omni image/audio answers are not calibrated (temperature 1), as in server-decision format_answer
    Json answer(const DecisionQuestion& q, const std::vector<float>& scores, bool has_media = false) const;
    std::string render_debug(const DecisionRequest& req, const DecisionQuestion& q, const DecisionMediaInfo& media = {}) const;

private:
    std::vector<std::string> render_pieces(const DecisionRequest& req, const DecisionQuestion& q, const DecisionMediaInfo& media) const;
    float temperature(const DecisionQuestion& q) const;
    const BpeTokenizer* tok_ = nullptr;
    std::map<std::string, float> temps_;
    std::map<size_t, std::vector<float>> score_bias_;
};

}  // namespace ferro
