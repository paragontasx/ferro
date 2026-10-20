// test_media.cpp — unit tests for the d1-omni media front end: data URLs / base64 (llama.cpp semantics),
// container sniffing, WAV decode + resample, log-mel frame and token counts, LFM2-VL tiling geometry.
// The expected geometry/token counts are the ones the llama.cpp mtmd oracle produced (bench/mm/mm_ref).
// Files under bench/d1/media are used when present.
#include "media.h"
#include "mm_preproc.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ferro;
static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { ++fails; fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, msg); } } while (0)

static std::vector<uint8_t> wav16(const std::vector<int16_t>& s, int rate, int ch) {
    std::vector<uint8_t> b;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 255); };
    auto u16 = [&](uint16_t v) { b.push_back(v & 255); b.push_back(v >> 8); };
    const uint32_t data = (uint32_t)s.size() * 2;
    b.insert(b.end(), {'R', 'I', 'F', 'F'}); u32(36 + data); b.insert(b.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    u32(16); u16(1); u16((uint16_t)ch); u32(rate); u32(rate * ch * 2); u16((uint16_t)(ch * 2)); u16(16);
    b.insert(b.end(), {'d', 'a', 't', 'a'}); u32(data);
    for (int16_t v : s) u16((uint16_t)v);
    return b;
}

static std::string b64(const std::vector<uint8_t>& in) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t v = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        o += T[v >> 18]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += T[v & 63];
    }
    if (i + 1 == in.size()) { uint32_t v = in[i] << 16; o += T[v >> 18]; o += T[(v >> 12) & 63]; o += "=="; }
    else if (i + 2 == in.size()) { uint32_t v = (in[i] << 16) | (in[i + 1] << 8); o += T[v >> 18]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += '='; }
    return o;
}

static void test_base64_url() {
    CHECK(base64_decode("aGVsbG8=") == std::vector<uint8_t>({'h', 'e', 'l', 'l', 'o'}), "base64 with padding");
    CHECK(base64_decode("aGVsbG8") == std::vector<uint8_t>({'h', 'e', 'l', 'l', 'o'}), "base64 without padding");
    CHECK(base64_decode("aGVs bG8=").size() == 3, "stops at a non-base64 character (llama.cpp semantics)");
    CHECK(media_bytes_from_url("data:image/png;base64,aGVsbG8=").size() == 5, "data URL");
    bool threw = false;
    try { media_bytes_from_url("https://example.com/cat.jpg"); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "remote URLs are rejected");
}

static void test_audio() {
    // 1 s 440 Hz stereo at 44.1 kHz -> mono 16 kHz
    std::vector<int16_t> s;
    for (int i = 0; i < 44100; ++i) { int16_t v = (int16_t)(12000 * std::sin(2 * M_PI * 440 * i / 44100.0)); s.push_back(v); s.push_back(v); }
    std::vector<uint8_t> w = wav16(s, 44100, 2);
    CHECK(is_audio_bytes(w.data(), w.size()), "WAV sniffed as audio");
    MediaItem m = decode_media(media_bytes_from_url("data:audio/wav;base64," + b64(w)), 16000);
    CHECK(m.is_audio, "decoded as audio");
    CHECK(std::abs((long)m.pcm.size() - 16000) <= 2, "resampled to 16 kHz mono");
    double rms = 0; for (float v : m.pcm) rms += v * v; rms = std::sqrt(rms / std::max<size_t>(1, m.pcm.size()));
    CHECK(std::abs(rms - 12000 / 32768.0 / std::sqrt(2.0)) < 0.02, "amplitude preserved");
    // frame / token counts: 11 s (jfk.wav) = 1100 frames -> 138 tokens; > 30 s is cut to 3000 frames -> 375 tokens
    Mel mel;
    std::vector<float> pcm(176000, 0.01f);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = 0.1f * std::sin(i * 0.05f);
    CHECK(d1omni_audio_mel(pcm.data(), pcm.size(), 128, mel) && mel.n_len == 1100 && audio_tokens(mel) == 138, "11 s -> 1100 frames -> 138 tokens");
    pcm.assign(704000, 0.0f);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = 0.1f * std::sin(i * 0.05f);
    CHECK(d1omni_audio_mel(pcm.data(), pcm.size(), 128, mel) && mel.n_len == 3000 && audio_tokens(mel) == 375, "44 s capped at 30 s -> 375 tokens");
    for (float v : mel.data) if (!std::isfinite(v)) { CHECK(false, "mel finite"); break; }
    const uint8_t png[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    CHECK(!is_audio_bytes(png, 8), "PNG is not audio");
}

static void test_image_geometry() {
    VisionPreParams p;
    auto run = [&](int nx, int ny) { std::vector<uint8_t> rgb((size_t)nx * ny * 3, 128); return d1omni_preprocess_image(rgb.data(), nx, ny, p); };
    auto v = run(640, 480);   // cats.jpg
    CHECK(v.size() == 1 && v[0].nx == 576 && v[0].ny == 416 && vision_tokens(v[0], p) == 234, "640x480 -> 576x416, 234 tokens");
    v = run(1920, 1080);      // big.png: 4x2 tiles of 512x512 + a 672x384 thumbnail
    bool tiles = v.size() == 9;
    for (int i = 0; tiles && i < 8; ++i) tiles = v[i].nx == 512 && v[i].ny == 512 && vision_tokens(v[i], p) == 256;
    CHECK(tiles && v[8].nx == 672 && v[8].ny == 384 && vision_tokens(v[8], p) == 252, "1920x1080 -> 8 tiles + thumbnail (thumbnail last)");
    v = run(100, 80);         // tiny.png: upscaled to the minimum token count
    int tok = 0; for (auto& x : v) tok += vision_tokens(x, p);
    CHECK(v.size() == 1 && tok == 72, "100x80 -> 72 tokens");
    // Pillow bilinear: a constant image stays constant, a 2x2 -> 1x1 downscale averages
    std::vector<uint8_t> src = {0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255}, dst(3);
    resize_pillow_bilinear(src.data(), 2, 2, dst.data(), 1, 1);
    CHECK(dst[0] == 128 && dst[1] == 128 && dst[2] == 128, "Pillow bilinear 2x2 -> 1x1 averages (round half up)");
}

static void test_gemma4_geometry() {
    Gemma4VisionPre p;
    auto toks = [&](int nx, int ny) {
        std::vector<uint8_t> rgb((size_t)nx * ny * 3, 100);
        VisionInput v = gemma4_preprocess_image(rgb.data(), nx, ny, p);
        CHECK(v.nx % 48 == 0 && v.ny % 48 == 0 && v.chw.size() == (size_t)3 * v.nx * v.ny, "aligned to patch*merge");
        return (v.nx / 48) * (v.ny / 48);
    };
    // token counts of the llama.cpp mtmd oracle (bench/mm/mm_ref on mmproj-embeddinggemma-2)
    CHECK(toks(640, 480) == 130, "640x480 -> 130 tokens");
    CHECK(toks(1920, 1080) == 920, "1920x1080 -> 920 tokens");
    CHECK(toks(100, 80) == 80, "100x80 -> 80 tokens (min budget)");
    CHECK(toks(8000, 6000) <= 1120, "huge image capped at 1120 tokens");
    // bicubic: constant image stays constant; a 4x1 ramp downscaled keeps its mean
    std::vector<uint8_t> flat(4 * 4 * 3, 77), out(2 * 2 * 3);
    resize_pillow_bicubic(flat.data(), 4, 4, out.data(), 2, 2);
    bool ok = true; for (uint8_t v : out) ok = ok && v == 77;
    CHECK(ok, "bicubic keeps a constant image constant");
}

static void test_files() {
    auto slurp = [](const char* path) { std::ifstream f(path, std::ios::binary); return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {}); };
    std::vector<uint8_t> b = slurp("bench/d1/media/jfk.wav");
    if (b.empty()) { fprintf(stderr, "skip: bench/d1/media not present\n"); return; }
    MediaItem m = decode_media(b, 16000);
    CHECK(m.is_audio && m.pcm.size() == 176000, "jfk.wav: 176000 samples at 16 kHz");
    for (const char* f : {"bench/d1/media/jfk.mp3", "bench/d1/media/jfk_44k.flac"}) {
        b = slurp(f);
        if (b.empty()) continue;
        m = decode_media(b, 16000);
        CHECK(m.is_audio && std::abs((long)m.pcm.size() - 176000) < 2000, f);
    }
    b = slurp("bench/d1/media/cats.jpg");
    if (!b.empty()) { m = decode_media(b, 16000); CHECK(!m.is_audio && m.nx == 640 && m.ny == 480 && m.rgb.size() == 640u * 480 * 3, "cats.jpg decodes to 640x480 RGB"); }
    bool threw = false;
    try { decode_media({'n', 'o', 't', ' ', 'm', 'e', 'd', 'i', 'a'}, 16000); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "undecodable bytes are rejected");
}

int main() {
    test_base64_url();
    test_audio();
    test_image_geometry();
    test_gemma4_geometry();
    test_files();
    if (fails) { fprintf(stderr, "test_media: %d FAILED\n", fails); return 1; }
    printf("test_media: ALL PASS\n");
    return 0;
}
