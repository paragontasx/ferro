// mm_preproc.h — d1-omni media preprocessing (exact ports of llama.cpp mtmd):
//  * images: LFM2-VL tiling (mtmd_image_preprocessor_lfm2, d1omni_v: tiles stretched, no padding),
//    Pillow-compatible bilinear resize, normalize (x/255 - mean)/std, planar CHW
//  * audio : NeMo-style log-mel (mtmd_audio_preprocessor_d1omni / _conformer): 16 kHz, n_fft 512,
//    win 400 (symmetric Hann), hop 160, preemph 0.97, natural log, per-feature normalization, 30 s cap
#pragma once
#include <cstdint>
#include <vector>

namespace ferro {

struct VisionPreParams {
    int patch_size = 16, n_merge = 2;
    int min_pixels = 64 * 1024, max_pixels = 256 * 1024;   // set_limit_image_tokens(64, 256) * (16*2)^2
    float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
};
// one encoder input (a tile or the thumbnail): planar [3][ny][nx], normalized
struct VisionInput { int nx = 0, ny = 0; std::vector<float> chw; };
// tiles row by row, then the thumbnail; or one resized image when no tiling is needed
std::vector<VisionInput> d1omni_preprocess_image(const uint8_t* rgb, int nx, int ny, const VisionPreParams& p);
// number of LM tokens the encoder emits for one input
int vision_tokens(const VisionInput& v, const VisionPreParams& p);

struct Mel { int n_mel = 0, n_len = 0; std::vector<float> data; };   // data[mel * n_len + frame]
bool d1omni_audio_mel(const float* pcm, size_t n, int n_mel, Mel& out);
int audio_tokens(const Mel& m);   // ceil(ceil(ceil(frames/2)/2)/2)

// Gemma 4 / EmbeddingGemma 2 audio (mtmd_audio_preprocessor_gemma4a): 16 kHz, n_fft 512, 20 ms periodic Hann (320, left-aligned
// in the FFT frame), hop 160, semicausal padding, magnitude spectrum, HTK mel (no area norm), ln(max(x, 1e-3)), no
// normalization. Audio longer than 30 s is split into 30 s pieces: one Mel per piece (data[mel * n_len + frame]).
bool gemma4a_audio_mel(const float* pcm, size_t n, int n_mel, std::vector<Mel>& out);
// encoder tokens for a Mel of n_len frames: two stride-2 3x3 convs (pad 1): n -> (n-1)/2+1, twice
inline int gemma4a_tokens(int n_len) { int n = n_len; for (int i = 0; i < 2; ++i) n = (n - 1) / 2 + 1; return n; }

// Pillow-compatible resize (bilinear), exposed for tests
void resize_pillow_bilinear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);
void resize_pillow_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);

// Gemma 4 / EmbeddingGemma 2 vision input (mtmd_image_preprocessor_dyn_size): one image, aspect preserved, edges
// aligned to patch*merge, token budget [min_tokens, max_tokens] (tokens = patch^2 * merge^2 pixels each), bicubic.
struct Gemma4VisionPre { int patch_size = 16, n_merge = 3, min_tokens = 70, max_tokens = 1120; };
VisionInput gemma4_preprocess_image(const uint8_t* rgb, int nx, int ny, const Gemma4VisionPre& p);

}  // namespace ferro
