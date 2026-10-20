// mel_ref.cpp - llama.cpp oracle for the Gemma 4 audio mel frontend (mtmd_audio_preprocessor_gemma4a), no model needed.
// Usage: mel_ref in.f32 out.bin     (in.f32 = raw 16 kHz mono float32 PCM)
// Output: int32 n_chunks, then per chunk: int32 n_mel, int32 n_len, f32 data[n_mel * n_len]   (data[mel * n_len + frame])
//
// The preprocessor class is UPSTREAM's, compiled unmodified from tools/mtmd/mtmd-audio.cpp. It reads its parameters
// through clip_get_hparams(ctx) and logs through clip.cpp's logger; clip_ctx is private to clip.cpp, so the harness
// provides those few symbols itself and hands the class an opaque pointer that is only ever passed back to
// clip_get_hparams. No model, no clip.cpp and no vendored-file edit are needed.
#include "mtmd-audio.h"
#include "clip-model.h"
#include "clip-impl.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

static clip_hparams g_hp;
const clip_hparams* clip_get_hparams(const struct clip_ctx*) { return &g_hp; }
struct clip_logger_state g_logger_state = {nullptr, nullptr};
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    fprintf(stderr, "ggml_abort at %s:%d: %s\n", file, line, fmt ? fmt : ""); abort();
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: mel_ref in.f32 out.bin\n"); return 1; }
    FILE* f = fopen(argv[1], "rb"); if (!f) return 2;
    std::vector<float> pcm; float v; while (fread(&v, 4, 1, f) == 1) pcm.push_back(v); fclose(f);
    // the values clip.cpp sets for PROJECTOR_TYPE_GEMMA4A (+ n_mel_bins from the mmproj: clip.audio.num_mel_bins = 128)
    g_hp.n_mel_bins = 128;
    g_hp.audio_chunk_len = 0;
    g_hp.audio_sample_rate = 16000;
    g_hp.audio_n_fft = 512;
    g_hp.audio_window_len = 320;
    g_hp.audio_hop_len = 160;
    mtmd_audio_preprocessor_gemma4a pre(reinterpret_cast<const clip_ctx*>(&g_hp));
    pre.initialize();                                  // builds the Hann window + HTK filterbank; without it the output is garbage
    std::vector<mtmd_audio_mel> out;
    if (!pre.preprocess(pcm.data(), pcm.size(), out)) { fprintf(stderr, "preprocess failed\n"); return 3; }
    FILE* o = fopen(argv[2], "wb");
    int32_t nc = (int32_t)out.size(); fwrite(&nc, 4, 1, o);
    for (auto& m : out) {
        int32_t nm = (int32_t)m.n_mel, nl = (int32_t)m.n_len;
        fwrite(&nm, 4, 1, o); fwrite(&nl, 4, 1, o); fwrite(m.data.data(), 4, (size_t)nm * nl, o);
        fprintf(stderr, "chunk: n_mel=%d n_len=%d\n", nm, nl);
    }
    fclose(o);
    return 0;
}
