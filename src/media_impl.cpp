// media_impl.cpp — the third-party decoders (own translation unit: miniaudio is ~95k lines).
// Same configuration as llama.cpp tools/mtmd/mtmd-helper.cpp.
#include <vector>
#include <cstddef>

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "../third_party/miniaudio/miniaudio.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include "../third_party/stb/stb_image.h"

namespace ferro_media_impl {

unsigned char* image_load(const unsigned char* buf, int len, int* nx, int* ny) {
    int nc = 0;
    return stbi_load_from_memory(buf, len, nx, ny, &nc, 3);
}
void image_free(unsigned char* p) { stbi_image_free(p); }

bool audio_decode(const unsigned char* buf, size_t len, int sample_rate, std::vector<float>& out) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, (ma_uint32)sample_rate);
    ma_decoder dec;
    if (ma_decoder_init_memory(buf, len, &cfg, &dec) != MA_SUCCESS) return false;
    ma_uint64 n = 0, got = 0;
    if (ma_decoder_get_length_in_pcm_frames(&dec, &n) != MA_SUCCESS) { ma_decoder_uninit(&dec); return false; }
    out.resize((size_t)n);
    if (ma_decoder_read_pcm_frames(&dec, out.data(), n, &got) != MA_SUCCESS) { ma_decoder_uninit(&dec); return false; }
    ma_decoder_uninit(&dec);
    return true;
}

}  // namespace ferro_media_impl
