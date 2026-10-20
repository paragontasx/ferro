// embed_mm.cpp — see embed_mm.h
#include "embed_mm.h"
#include "media.h"
#include "threadpool.h"
#include "gpu_cl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <unistd.h>

namespace ferro {

static std::string shell_quote(const std::string& s) {
    std::string o = "'";
    for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; }
    return o + "'";
}

bool decode_video_frames(const std::string& path_in, const std::vector<uint8_t>& bytes, float fps_target, VideoFrames& out, std::string* err) {
    std::string path = path_in, tmp;
    if (!bytes.empty()) {                                   // in-memory video: spill to a scratch file for ffprobe/ffmpeg
        const char* td = getenv("TMPDIR");
        char buf[512]; snprintf(buf, sizeof buf, "%s/ferro_video_XXXXXX", td ? td : "/tmp");
        int fd = mkstemp(buf);
        if (fd < 0) { *err = "cannot create a temporary file for the video"; return false; }
        size_t off = 0; while (off < bytes.size()) { ssize_t n = write(fd, bytes.data() + off, bytes.size() - off); if (n <= 0) break; off += (size_t)n; }
        close(fd); tmp = buf; path = tmp;
    }
    struct Cleanup { std::string f; ~Cleanup() { if (!f.empty()) unlink(f.c_str()); } } cleanup{tmp};
    // probe: width/height/r_frame_rate (same fields as llama.cpp's helper)
    int w = 0, h = 0; float ofps = 0;
    {
        std::string cmd = "ffprobe -v quiet -show_entries stream=width,height,r_frame_rate -select_streams v:0 -of default=noprint_wrappers=1 " + shell_quote(path) + " 2>/dev/null";
        FILE* p = popen(cmd.c_str(), "r");
        if (!p) { *err = "cannot run ffprobe (is ffmpeg installed?)"; return false; }
        char line[256];
        while (fgets(line, sizeof line, p)) {
            char* eq = strchr(line, '='); if (!eq) continue; *eq = 0;
            const char* k = line; const char* v = eq + 1;
            if (!strcmp(k, "width")) w = atoi(v);
            else if (!strcmp(k, "height")) h = atoi(v);
            else if (!strcmp(k, "r_frame_rate")) { int n = 0, d = 1; if (sscanf(v, "%d/%d", &n, &d) == 2 && d > 0) ofps = (float)n / d; else ofps = (float)atof(v); }
        }
        pclose(p);
    }
    if (w <= 0 || h <= 0 || ofps <= 0) { *err = "could not read the video stream (ffprobe failed)"; return false; }
    out.width = w; out.height = h; out.fps = fps_target > 0 ? fps_target : ofps;
    char fpsarg[64]; snprintf(fpsarg, sizeof fpsarg, "fps=%.6f", out.fps);
    std::string cmd = "ffmpeg -nostdin -i " + shell_quote(path) + " -vf " + fpsarg + " -f rawvideo -pix_fmt rgb24 pipe:1 -loglevel error";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) { *err = "cannot run ffmpeg"; return false; }
    const size_t fs = (size_t)w * h * 3;
    for (;;) {
        std::vector<uint8_t> f(fs);
        size_t n = fread(f.data(), 1, fs, p);
        if (n < fs) break;                                  // clean EOF (a partial frame is dropped)
        out.rgb.push_back(std::move(f));
    }
    pclose(p);
    if (out.rgb.empty()) { *err = "no video frames decoded"; return false; }
    return true;
}

bool MmEmbedder::init(const std::string& model_path, const std::string& mmproj_path, int threads, std::string* err) {
    if (!eng_.init(model_path, threads, err)) return false;
    if (!tokf_.open_file(model_path.c_str(), err) || !tok_.load(tokf_, err)) return false;
    if (!mmproj_path.empty() && !proj_.load(mmproj_path, err)) return false;
    return true;
}

bool MmEmbedder::embed(const std::vector<EmbedItem>& items, const MmEmbedOptions& opt, std::vector<float>& out, std::string* err, int* n_tokens) {
    const int E = eng_.dim_in();
    std::vector<int> ids, midx;
    std::vector<float> rows;                                // media rows [n][E]
    auto push_text = [&](const std::string& t) {
        for (int id : tok_.tokenize(t, false)) { ids.push_back(id); midx.push_back(-1); }
    };
    auto push_tok = [&](int id) { ids.push_back(id); midx.push_back(-1); };
    auto push_media = [&](const std::vector<float>& emb, int n_tok, int beg, int end) {
        if (emb.size() != (size_t)n_tok * E) { *err = "media embedding width does not match the text model"; return false; }
        push_tok(beg);
        const int base = (int)(rows.size() / E);
        rows.insert(rows.end(), emb.begin(), emb.end());
        for (int i = 0; i < n_tok; ++i) { ids.push_back(0); midx.push_back(base + i); }
        push_tok(end);
        return true;
    };
    constexpr int IMG_BEG = 255999, IMG_END = 258882;
    auto image = [&](const uint8_t* rgb, int nx, int ny) {
        if (!proj_.has_vision()) { *err = "no vision encoder loaded (pass --mmproj)"; return false; }
        VisionInput in = gemma4_preprocess_image(rgb, nx, ny, proj_.vision_params());
        return push_media(proj_.encode_image(in), (in.nx / 48) * (in.ny / 48), IMG_BEG, IMG_END);
    };
    try {
        if (tok_.bos_id() >= 0) push_tok(tok_.bos_id());
        for (const EmbedItem& it : items) {
            if (it.kind == EmbedItem::TEXT) { push_text(it.value); continue; }
            std::vector<uint8_t> bytes = it.bytes;
            if (it.kind != EmbedItem::VIDEO && bytes.empty()) {
                std::ifstream f(it.value, std::ios::binary);
                if (!f) { *err = "cannot read " + it.value; return false; }
                bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }
            if (it.kind == EmbedItem::IMAGE) {
                MediaItem m = decode_media(bytes, 16000);
                if (m.is_audio) { *err = "expected an image: " + it.value; return false; }
                if (!image(m.rgb.data(), m.nx, m.ny)) return false;
            } else if (it.kind == EmbedItem::AUDIO) {
                *err = "audio input is not implemented yet"; return false;
            } else {
                VideoFrames vf;
                if (!decode_video_frames(it.value, it.bytes, opt.video_fps, vf, err)) return false;
                push_text("Video:");
                float next_ts = 0.0f;
                for (size_t i = 0; i < vf.rgb.size(); ++i) {
                    if (!image(vf.rgb[i].data(), vf.width, vf.height)) return false;
                    if (opt.video_ts_interval_ms > 0) {
                        const float elapsed_ms = (float)i / vf.fps * 1000.0f;
                        if (elapsed_ms >= next_ts) {
                            const float es = elapsed_ms / 1000.0f; const int mins = (int)(es / 60); const float secs = es - mins * 60.0f;
                            char b[32]; snprintf(b, sizeof b, "[%dm%.2fs]", mins, secs);
                            push_text(b);
                            next_ts += (float)opt.video_ts_interval_ms;
                        }
                    }
                }
            }
        }
        if (tok_.eos_id() >= 0) push_tok(tok_.eos_id());
    } catch (const std::exception& e) { *err = e.what(); return false; }
    const int T = (int)ids.size();
    if (n_tokens) *n_tokens = T;
    if (T > 8192) { *err = "input is " + std::to_string(T) + " tokens; the model context is 8192"; return false; }
    std::vector<int> seg(T, 0), pos(T);
    for (int t = 0; t < T; ++t) pos[t] = t;
    out.assign(eng_.dim_out(), 0.0f);
    eng_.encode_mm(ids.data(), seg.data(), pos.data(), T, 1, out.data(), rows.data(), midx.data());
    return true;
}

}  // namespace ferro
