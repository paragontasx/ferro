// media.cpp — see media.h. stb_image / miniaudio implementations live in media_impl.cpp.
#include "media.h"

#include <cctype>
#include <cstring>
#include <stdexcept>

// decoders (media_impl.cpp)
namespace ferro_media_impl {
unsigned char* image_load(const unsigned char* buf, int len, int* nx, int* ny);   // RGB8, free with image_free
void image_free(unsigned char* p);
bool audio_decode(const unsigned char* buf, size_t len, int sample_rate, std::vector<float>& out);
}

namespace ferro {

static const char* B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static inline bool is_b64(uint8_t c) { return std::isalnum(c) || c == '+' || c == '/'; }

std::vector<uint8_t> base64_decode(const std::string& s) {
    // same as llama.cpp server-common.cpp base64_decode
    int8_t lut[256];
    memset(lut, 0, sizeof(lut));
    for (int i = 0; i < 64; ++i) lut[(uint8_t)B64[i]] = (int8_t)i;
    std::vector<uint8_t> ret;
    ret.reserve(s.size() / 4 * 3 + 3);
    uint8_t a4[4], a3[3];
    int i = 0;
    size_t in = 0;
    while (in < s.size() && s[in] != '=' && is_b64((uint8_t)s[in])) {
        a4[i++] = (uint8_t)s[in++];
        if (i == 4) {
            for (int k = 0; k < 4; ++k) a4[k] = (uint8_t)lut[a4[k]];
            a3[0] = (uint8_t)((a4[0] << 2) + ((a4[1] & 0x30) >> 4));
            a3[1] = (uint8_t)(((a4[1] & 0xf) << 4) + ((a4[2] & 0x3c) >> 2));
            a3[2] = (uint8_t)(((a4[2] & 0x3) << 6) + a4[3]);
            ret.insert(ret.end(), a3, a3 + 3);
            i = 0;
        }
    }
    if (i) {
        for (int k = i; k < 4; ++k) a4[k] = 0;
        for (int k = 0; k < 4; ++k) a4[k] = (uint8_t)lut[a4[k]];
        a3[0] = (uint8_t)((a4[0] << 2) + ((a4[1] & 0x30) >> 4));
        a3[1] = (uint8_t)(((a4[1] & 0xf) << 4) + ((a4[2] & 0x3c) >> 2));
        a3[2] = (uint8_t)(((a4[2] & 0x3) << 6) + a4[3]);
        for (int k = 0; k < i - 1; ++k) ret.push_back(a3[k]);
    }
    return ret;
}

static bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

std::vector<uint8_t> media_bytes_from_url(const std::string& url) {
    if (starts_with(url, "http") || starts_with(url, "file://"))
        throw std::invalid_argument("media must be inline base64 data (data:image/...;base64,... or data:audio/...;base64,...); remote and file URLs are not fetched");
    if (starts_with(url, "data:")) {
        const size_t comma = url.find(',');
        if (comma == std::string::npos || url.find(',', comma + 1) != std::string::npos)
            throw std::invalid_argument("Invalid uri-encoded base64 value");
        const std::string head = url.substr(0, comma);
        if (!starts_with(head, "data:image/") && !starts_with(head, "data:video/") && !starts_with(head, "data:audio/"))
            throw std::invalid_argument("Invalid uri format: " + head);
        if (head.size() < 6 || head.compare(head.size() - 6, 6, "base64") != 0)
            throw std::invalid_argument("uri must be base64 encoded");
        return base64_decode(url.substr(comma + 1));
    }
    std::vector<uint8_t> b = base64_decode(url);
    if (b.empty()) throw std::invalid_argument("Invalid base64 value");
    return b;
}

bool is_audio_bytes(const uint8_t* buf, size_t len) {
    if (len < 12) return false;
    const bool wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    const bool mp3 = memcmp(buf, "ID3", 3) == 0 || (buf[0] == 0xFF && (buf[1] & 0xE0) == 0xE0);
    const bool flac = memcmp(buf, "fLaC", 4) == 0;
    return wav || mp3 || flac;
}

MediaItem decode_media(const std::vector<uint8_t>& bytes, int sample_rate) {
    MediaItem m;
    m.sha256 = sha256_hex(bytes.data(), bytes.size());
    if (is_audio_bytes(bytes.data(), bytes.size())) {
        m.is_audio = true;
        if (!ferro_media_impl::audio_decode(bytes.data(), bytes.size(), sample_rate, m.pcm))
            throw std::invalid_argument("unable to decode the audio file (WAV, MP3 and FLAC are supported)");
        return m;
    }
    int nx = 0, ny = 0;
    unsigned char* p = ferro_media_impl::image_load(bytes.data(), (int)bytes.size(), &nx, &ny);
    if (!p) throw std::invalid_argument("unable to decode the media file (images: JPEG, PNG, BMP, GIF, TGA, PSD, HDR, PNM; audio: WAV, MP3, FLAC)");
    m.nx = nx; m.ny = ny;
    m.rgb.assign(p, p + (size_t)nx * ny * 3);
    ferro_media_impl::image_free(p);
    return m;
}

// ---- SHA-256 (FIPS 180-4), for media cache keys
std::string sha256_hex(const uint8_t* data, size_t len) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    std::vector<uint8_t> msg(data, data + len);
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    const uint64_t bits = (uint64_t)len * 8;
    for (int i = 7; i >= 0; --i) msg.push_back((uint8_t)(bits >> (i * 8)));
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t)msg[off + 4 * i] << 24 | (uint32_t)msg[off + 4 * i + 1] << 16 | (uint32_t)msg[off + 4 * i + 2] << 8 | msg[off + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static const char* hx = "0123456789abcdef";
    std::string out;
    for (uint32_t v : h) for (int i = 7; i >= 0; --i) out += hx[(v >> (i * 4)) & 15];
    return out;
}

}  // namespace ferro
