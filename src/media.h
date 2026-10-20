// media.h — decision-request media: data URLs / raw base64 -> decoded image (RGB8) or audio (f32 mono).
// Mirrors llama.cpp tools/server (handle_media, base64_decode) and mtmd-helper (stb_image for images,
// miniaudio for WAV/MP3/FLAC resampled to the encoder rate).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ferro {

struct MediaItem {
    bool is_audio = false;
    int nx = 0, ny = 0;               // image size
    std::vector<uint8_t> rgb;         // image: nx*ny*3, row-major RGB
    std::vector<float> pcm;           // audio: mono f32 at the requested sample rate
    std::string sha256;               // of the encoded bytes (cache key)
};

// "data:image/...;base64,..." / "data:audio/...;base64,..." or raw base64 -> bytes. Throws std::invalid_argument.
std::vector<uint8_t> media_bytes_from_url(const std::string& url);
std::vector<uint8_t> base64_decode(const std::string& s);   // llama.cpp semantics (stops at '=' / non-base64)
bool is_audio_bytes(const uint8_t* buf, size_t len);         // WAV (RIFF/WAVE), MP3 (ID3 / sync), FLAC
// decode bytes; audio is resampled to sample_rate mono. Throws std::invalid_argument on undecodable data.
MediaItem decode_media(const std::vector<uint8_t>& bytes, int sample_rate);
std::string sha256_hex(const uint8_t* data, size_t len);

}  // namespace ferro
