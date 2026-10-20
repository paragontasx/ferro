// main.cpp — ferro-embed CLI
//   ferro-embed embed  --model M.gguf (--in file.txt | --tokens tokens.txt) [opts]
//   ferro-embed bench  --model M.gguf --in corpus.txt --reps N [opts]
//   ferro-embed decide --model d1.gguf --request req.json [--out answers.json] [--reps N]
// tokens.txt format: one sentence per line; each line = space-separated token ids.
// req.json: one systemone request {"state":..., "questions":{...}} or a JSON array of them.
#include "model_eg2.h"
#include "model_lfm2.h"
#include "media.h"
#include "mm_encoder.h"
#include "mm_gemma4.h"
#include "embed_mm.h"
#include "gpu_cl.h"
#include "tokenizer.h"
#include "threadpool.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <string>
#include <vector>

using namespace ferro;
using clk = std::chrono::steady_clock;

static void usage() {
    fprintf(stderr,
        "usage: ferro-embed embed|bench --model EMB.gguf [--in f|--tokens f] [--out f] [--threads N] [--max-tokens T] [--reps R] [--int8] [--cpu|--gpu|--hetero]\n"
        "       ferro-embed decide --model D1.gguf --request req.json|- [--mmproj P.gguf] [--out f] [--threads N] [--reps R] [--cpu|--gpu]\n");
    exit(2);
}

// decision models (LFM2 d1-omni): typed answers over a state, one forward pass, no generation
static int run_decide(const std::string& model, const std::string& reqf, const std::string& outp, int threads, int reps,
                      std::string mmproj_path) {
    struct PoolGuard { ~PoolGuard() { Pool::get().shutdown(); } } pool_guard;   // join workers on every exit path
    auto t0 = clk::now();
    DecisionEngine eng;
    std::string err;
    if (!eng.init(model, threads, &err)) { fprintf(stderr, "init failed: %s\n", err.c_str()); return 1; }
    BpeTokenizer tok;
    if (!tok.load(eng.gguf(), &err)) { fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
    DecisionPrompter dp;
    if (!dp.init(eng.gguf(), &tok, &err)) { fprintf(stderr, "decision: %s\n", err.c_str()); return 1; }
    fprintf(stderr, "[ferro] decision model %s loaded in %.2fs isa=%d threads=%d device=%s\n",
            eng.gguf().get_str("general.name").c_str(), std::chrono::duration<double>(clk::now() - t0).count(),
            isa_level(), threads, eng.on_gpu() ? "igpu" : "cpu");

    std::string text;
    if (reqf == "-") { std::stringstream ss; ss << std::cin.rdbuf(); text = ss.str(); }
    else { std::ifstream f(reqf); if (!f) { fprintf(stderr, "cannot read %s\n", reqf.c_str()); return 1; } std::stringstream ss; ss << f.rdbuf(); text = ss.str(); }

    std::vector<DecisionRequest> reqs;
    bool batch = false;
    std::vector<DecisionSeq> seqs;
    // media: decoded + preprocessed per request (encoder inputs), encoded per run
    struct ReqMedia { bool is_audio = false; int n_files = 0; std::vector<VisionInput> vin; Mel mel; int n_tok = 0; };
    std::vector<ReqMedia> rmedia;
    std::vector<int> req_media;     // request -> index into rmedia (-1 = text only)
    MmProj mm;
    double t_pre = 0;
    try {
        Json body = json_parse(text);
        batch = body.is_array();
        if (batch) for (auto& r : body.arr) reqs.push_back(dp.parse(r));
        else reqs.push_back(dp.parse(body));
        bool any_media = false;
        for (auto& r : reqs) any_media = any_media || !r.files.empty();
        if (any_media) {
            if (mmproj_path.empty()) mmproj_path = find_mmproj_near(model);
            if (mmproj_path.empty()) throw std::invalid_argument("this request has images/audio but no mmproj GGUF was given (--mmproj) or found next to the model");
            std::string merr;
            auto tl = clk::now();
            if (!mm.load(mmproj_path, &merr)) throw std::runtime_error("mmproj " + mmproj_path + ": " + merr);
            if (eng.on_gpu()) mm.upload_gpu();
            fprintf(stderr, "[ferro] mmproj %s loaded in %.2fs (vision %s, audio %s)\n", mmproj_path.c_str(),
                    std::chrono::duration<double>(clk::now() - tl).count(), mm.has_vision() ? "yes" : "no", mm.has_audio() ? "yes" : "no");
        }
        auto tp = clk::now();
        for (auto& r : reqs) {
            if (r.files.empty()) { req_media.push_back(-1); continue; }
            ReqMedia rm;
            rm.n_files = (int)r.files.size();
            std::vector<MediaItem> items;
            for (auto& f : r.files) items.push_back(decode_media(f, 16000));
            for (auto& it : items) rm.is_audio = rm.is_audio || it.is_audio;
            if (rm.is_audio && items.size() > 1) throw std::invalid_argument("a request has images or one audio clip, not both");
            if (rm.is_audio) {
                if (!mm.has_audio()) throw std::invalid_argument("this mmproj has no audio encoder");
                if (!d1omni_audio_mel(items[0].pcm.data(), items[0].pcm.size(), mm.n_mel(), rm.mel)) throw std::invalid_argument("the audio clip is empty");
                rm.n_tok = audio_tokens(rm.mel);
            } else {
                if (!mm.has_vision()) throw std::invalid_argument("this mmproj has no vision encoder");
                for (auto& it : items) {
                    auto v = d1omni_preprocess_image(it.rgb.data(), it.nx, it.ny, mm.vision_params());
                    for (auto& x : v) { rm.n_tok += vision_tokens(x, mm.vision_params()); rm.vin.push_back(std::move(x)); }
                }
            }
            req_media.push_back((int)rmedia.size());
            rmedia.push_back(std::move(rm));
        }
        t_pre = std::chrono::duration<double>(clk::now() - tp).count();
        for (size_t ri = 0; ri < reqs.size(); ++ri) {
            DecisionMediaInfo mi;
            if (req_media[ri] >= 0) { const auto& rm = rmedia[req_media[ri]]; mi.n_tokens = rm.n_tok; mi.is_audio = rm.is_audio; mi.n_files = rm.n_files; }
            for (auto& q : reqs[ri].questions) {
                DecisionSeq sq = dp.build(reqs[ri], q, mi);
                sq.media = req_media[ri];
                seqs.push_back(std::move(sq));
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "bad request: %s\n", e.what());
        printf("{\"error\":{\"code\":400,\"message\":%s,\"type\":\"invalid_request_error\"}}\n", json_dump(Json::str(e.what())).c_str());
        return 3;
    }
    size_t n_tok = 0, n_media_tok = 0;
    for (auto& s : seqs) n_tok += s.tokens.size() + s.n_media;
    for (auto& rm : rmedia) n_media_tok += rm.n_tok;
    if (!rmedia.empty())
        fprintf(stderr, "[ferro] %zu media request(s), %zu media tokens (decode+preprocess %.1f ms)\n", rmedia.size(), n_media_tok, t_pre * 1e3);
    if (getenv("FERRO_DECISION_DUMP")) {
        size_t k = 0;
        for (size_t ri = 0; ri < reqs.size(); ++ri) for (auto& q : reqs[ri].questions) {
            auto& r = reqs[ri];
            DecisionMediaInfo mi;
            if (req_media[ri] >= 0) { mi.n_tokens = rmedia[req_media[ri]].n_tok; mi.is_audio = rmedia[req_media[ri]].is_audio; mi.n_files = rmedia[req_media[ri]].n_files; }
            fprintf(stderr, "[prompt %s] (media %d) %s\n[tokens]", q.id.c_str(), mi.n_tokens, dp.render_debug(r, q, mi).c_str());
            for (int t : seqs[k].tokens) fprintf(stderr, " %d", t);
            fprintf(stderr, "\n"); ++k;
        }
    }
    fprintf(stderr, "[ferro] %zu request(s), %zu question sequences, %zu tokens\n", reqs.size(), seqs.size(), n_tok);

    std::vector<std::vector<float>> scores;
    double best = 1e30, total = 0;
    reps = std::max(1, reps);
    std::vector<std::vector<float>> membd(rmedia.size());
    std::vector<DecisionMediaEmbd> media(rmedia.size());
    for (int r = 0; r < reps; ++r) {
        auto s = clk::now();
        // media encoders (part of the request latency)
        for (size_t i = 0; i < rmedia.size(); ++i) {
            membd[i] = rmedia[i].is_audio ? mm.encode_audio(rmedia[i].mel) : mm.encode_vision(rmedia[i].vin);
            if (const char* ov = getenv("FERRO_MEDIA_OVERRIDE")) {
                // debug: replace the encoder output with a dump (bench/mm/mm_ref format) of the same length
                FILE* f = fopen(ov, "rb");
                int32_t nc = 0; std::vector<float> all;
                if (f && fread(&nc, 4, 1, f) == 1)
                    for (int c = 0; c < nc; ++c) {
                        int32_t n = 0, ne = 0;
                        if (fread(&n, 4, 1, f) != 1 || fread(&ne, 4, 1, f) != 1) break;
                        std::vector<float> v((size_t)n * ne);
                        if (fread(v.data(), 4, v.size(), f) != v.size()) break;
                        all.insert(all.end(), v.begin(), v.end());
                    }
                if (f) fclose(f);
                if (all.size() == membd[i].size()) { membd[i] = all; fprintf(stderr, "[debug] media %zu: encoder output replaced by %s\n", i, ov); }
            }
            media[i].embd = membd[i].data();
            media[i].n_tok = (int)(membd[i].size() / eng.model().n_embd);
            if (media[i].n_tok != rmedia[i].n_tok) { fprintf(stderr, "media encoder produced %d tokens, expected %d\n", media[i].n_tok, rmedia[i].n_tok); return 1; }
        }
        if (!rmedia.empty() && (reps > 1 || getenv("FERRO_VERBOSE")))
            fprintf(stderr, "[ferro] media encode %.1f ms\n", std::chrono::duration<double, std::milli>(clk::now() - s).count());
        static const bool sequential = getenv("FERRO_DECISION_SEQUENTIAL") != nullptr;   // one request per forward (latency mode)
        if (!sequential) eng.score(seqs, scores, media);
        else {
            scores.clear();
            size_t k0 = 0;
            for (auto& rq : reqs) {
                std::vector<DecisionSeq> part(seqs.begin() + k0, seqs.begin() + k0 + rq.questions.size());
                std::vector<std::vector<float>> sc;
                eng.score(part, sc, media);
                scores.insert(scores.end(), sc.begin(), sc.end());
                k0 += rq.questions.size();
            }
        }
        double dt = std::chrono::duration<double>(clk::now() - s).count();
        if (r > 0 || reps == 1) { best = std::min(best, dt); total += dt; }
        if (reps > 1) fprintf(stderr, "[ferro] rep %d: %.1f ms  (%.0f tok/s)\n", r, dt * 1e3, n_tok / dt);
    }
    if (reps > 1)
        fprintf(stderr, "[ferro] BEST %.1f ms | avg(excl warmup) %.1f ms | %.0f tok/s\n",
                best * 1e3, total / (reps - 1) * 1e3, n_tok / (total / (reps - 1)));

    Json all = Json::array();
    size_t k = 0;
    try {
        for (size_t ri = 0; ri < reqs.size(); ++ri) {
            auto& r = reqs[ri];
            Json answers = Json::object();
            int64_t ntok = 0;
            for (auto& q : r.questions) { ntok += (int64_t)(seqs[k].tokens.size() + seqs[k].n_media); answers.set(q.id, dp.answer(q, scores[k], req_media[ri] >= 0)); ++k; }
            Json res = Json::object();
            res.set("model", Json::str(eng.gguf().get_str("general.name")));
            res.set("answers", answers);
            Json usage = Json::object();
            usage.set("input_tokens", Json::integer(ntok));
            usage.set("output_tokens", Json::integer(0));
            res.set("usage", usage);
            all.push(res);
        }
    } catch (const std::exception& e) { fprintf(stderr, "decision failed: %s\n", e.what()); return 1; }
    const std::string js = json_dump(batch ? all : all.arr[0]);
    if (!outp.empty()) {
        FILE* f = fopen(outp.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", outp.c_str()); return 1; }
        fprintf(f, "%s\n", js.c_str()); fclose(f);
        fprintf(stderr, "[ferro] wrote %s\n", outp.c_str());
    } else if (reps <= 1) {
        printf("%s\n", js.c_str());      // bench mode (reps > 1) prints answers only with --out
    }
    return 0;
}

// debug/oracle: media embeddings from the mmproj (same binary layout as bench/mm/mm_ref)
// EmbeddingGemma 2 projector (gemma4v / gemma4a): same dump format as the d1 path
static int run_mm_embed_g4(const std::string& mmproj, const std::string& media, const std::string& outp, int threads) {
    struct PoolGuard { ~PoolGuard() { Pool::get().shutdown(); } } pool_guard;
    Pool::get().init(threads);
    Gemma4Proj mm;
    std::string err;
    if (!mm.load(mmproj, &err)) { fprintf(stderr, "mmproj: %s\n", err.c_str()); return 1; }
    const char* ge = getenv("FERRO_GPU");
    if ((!ge || atoi(ge) != 0) && gpu::init(getenv("FERRO_VERBOSE") != nullptr)) mm.upload_gpu();   // vision tower on the iGPU
    std::ifstream f(media, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", media.c_str()); return 1; }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    MediaItem it = decode_media(bytes, 16000);
    std::vector<std::vector<float>> chunks;
    if (it.is_audio) {
        if (!mm.has_audio()) { fprintf(stderr, "mmproj has no audio encoder\n"); return 1; }
        std::vector<Mel> mels;
        if (!gemma4a_audio_mel(it.pcm.data(), it.pcm.size(), 128, mels)) { fprintf(stderr, "audio mel failed\n"); return 1; }
        auto ta = clk::now();
        for (auto& m : mels) {
            chunks.push_back(mm.encode_audio(m));
            fprintf(stderr, "[mm] audio chunk: %d frames -> %d tokens\n", m.n_len, gemma4a_tokens(m.n_len));
        }
        fprintf(stderr, "[mm] audio encoded in %.1f ms\n", std::chrono::duration<double, std::milli>(clk::now() - ta).count());
        FILE* o = fopen(outp.c_str(), "wb");
        if (!o) return 1;
        int32_t nc = (int32_t)chunks.size(), ne = mm.n_embd_out();
        fwrite(&nc, 4, 1, o);
        for (auto& c : chunks) { int32_t n = (int32_t)(c.size() / ne); fwrite(&n, 4, 1, o); fwrite(&ne, 4, 1, o); fwrite(c.data(), 4, c.size(), o); }
        fclose(o);
        return 0;
    }
    VisionInput in = gemma4_preprocess_image(it.rgb.data(), it.nx, it.ny, mm.vision_params());
    if (getenv("FERRO_MM_WARM")) {                       // steady-state timing: first call pays kernel build + buffer allocation
        auto w0 = clk::now(); (void)mm.encode_image(in);
        fprintf(stderr, "[mm] warm-up encode %.1f ms (vision on %s)\n", std::chrono::duration<double, std::milli>(clk::now() - w0).count(), mm.vision_on_gpu() ? "iGPU" : "CPU");
    }
    auto t0 = clk::now();
    chunks.push_back(mm.encode_image(in));
    fprintf(stderr, "[mm] image %dx%d input %dx%d -> %d tokens\n", it.nx, it.ny, in.nx, in.ny, (in.nx / 48) * (in.ny / 48));
    fprintf(stderr, "[mm] encoded in %.1f ms\n", std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    FILE* o = fopen(outp.c_str(), "wb");
    if (!o) return 1;
    int32_t nc = (int32_t)chunks.size(), ne = mm.n_embd_out();
    fwrite(&nc, 4, 1, o);
    for (auto& c : chunks) { int32_t n = (int32_t)(c.size() / ne); fwrite(&n, 4, 1, o); fwrite(&ne, 4, 1, o); fwrite(c.data(), 4, c.size(), o); }
    fclose(o);
    return 0;
}

static int run_mm_embed(const std::string& mmproj, const std::string& media, const std::string& outp, int threads) {
    struct PoolGuard { ~PoolGuard() { Pool::get().shutdown(); } } pool_guard;
    Pool::get().init(threads);
    MmProj mm;
    std::string err;
    if (!mm.load(mmproj, &err)) { fprintf(stderr, "mmproj: %s\n", err.c_str()); return 1; }
    const char* ge = getenv("FERRO_GPU");
    if ((!ge || atoi(ge) != 0) && gpu::init(getenv("FERRO_VERBOSE") != nullptr)) mm.upload_gpu();
    std::ifstream f(media, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", media.c_str()); return 1; }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    MediaItem it = decode_media(bytes, 16000);
    std::vector<std::vector<float>> chunks;
    auto t0 = clk::now();
    if (it.is_audio) {
        Mel mel;
        if (!d1omni_audio_mel(it.pcm.data(), it.pcm.size(), mm.n_mel(), mel)) { fprintf(stderr, "audio too short\n"); return 1; }
        chunks.push_back(mm.encode_audio(mel));
        fprintf(stderr, "[mm] audio %zu samples -> %d mel frames -> %d tokens\n", it.pcm.size(), mel.n_len, audio_tokens(mel));
    } else {
        auto in = d1omni_preprocess_image(it.rgb.data(), it.nx, it.ny, mm.vision_params());
        std::vector<float> all = mm.encode_vision(in);
        size_t o = 0;
        for (auto& v : in) {
            const size_t n = (size_t)vision_tokens(v, mm.vision_params()) * mm.n_embd_out();
            chunks.emplace_back(all.begin() + o, all.begin() + o + n);
            o += n;
            fprintf(stderr, "[mm] image %dx%d input %dx%d -> %d tokens\n", it.nx, it.ny, v.nx, v.ny, vision_tokens(v, mm.vision_params()));
        }
    }
    fprintf(stderr, "[mm] encoded in %.1f ms\n", std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    FILE* o = fopen(outp.c_str(), "wb");
    if (!o) return 1;
    int32_t nc = (int32_t)chunks.size(), ne = mm.n_embd_out();
    fwrite(&nc, 4, 1, o);
    for (auto& c : chunks) { int32_t n = (int32_t)(c.size() / ne); fwrite(&n, 4, 1, o); fwrite(&ne, 4, 1, o); fwrite(c.data(), 4, c.size(), o); }
    fclose(o);
    return 0;
}

static std::vector<std::string> read_lines(const char* path) {
    std::vector<std::string> out; std::ifstream f(path); std::string s;
    while (std::getline(f, s)) { if (!s.empty() && s.back() == '\r') s.pop_back(); if (!s.empty()) out.push_back(s); }
    return out;
}

int main(int argc, char** argv) {
    if (getenv("FERRO_SCALAR")) set_isa_level(0);
    if (argc < 2) usage();
    std::string cmd = argv[1];
    std::string model, inp, tokf, outp, reqf, mmproj, media;
    std::vector<std::string> items;   // embed-media: ordered text:/image:/audio:/video: items
    float video_fps = 4.0f;
    int threads = 0, maxtok = 2048, reps = 5;  // 0 = auto: power-limited SoCs run best on 2 cores
    bool int8 = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
            return argv[++i];
        };
        if (a == "--model") model = next("--model");
        else if (a == "--request") reqf = next("--request");
        else if (a == "--mmproj") mmproj = next("--mmproj");
        else if (a == "--media") media = next("--media");
        else if (a == "--in") inp = next("--in");
        else if (a == "--tokens") tokf = next("--tokens");
        else if (a == "--out") outp = next("--out");
        else if (a == "--item") items.push_back(next("--item"));
        else if (a == "--fps") video_fps = (float)atof(next("--fps"));
        else if (a == "--threads") threads = atoi(next("--threads"));
        else if (a == "--max-tokens") maxtok = atoi(next("--max-tokens"));
        else if (a == "--reps") reps = atoi(next("--reps"));
        else if (a == "--int8") int8 = true;
        else if (a == "--cpu") setenv("FERRO_GPU", "0", 1);            // force CPU path
        else if (a == "--gpu") setenv("FERRO_GPU", "1", 1);            // iGPU (default when available)
        else if (a == "--hetero") { setenv("FERRO_GPU", "1", 1); setenv("FERRO_HETERO", "1", 1); }  // CPU+iGPU split
        else usage();
    }
    if (cmd == "embed-media") {
        if (model.empty() || items.empty()) { fprintf(stderr, "embed-media --model M.gguf [--mmproj P.gguf] --item text:STR|image:F|audio:F|video:F ... [--fps N] [--out f.bin]\n"); return 2; }
        struct PoolGuard { ~PoolGuard() { Pool::get().shutdown(); } } pool_guard;
        MmEmbedder me; std::string err;
        if (!me.init(model, mmproj, threads > 0 ? threads : 2, &err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
        std::vector<EmbedItem> its;
        for (auto& s : items) {
            const size_t c = s.find(':');
            if (c == std::string::npos) { fprintf(stderr, "bad --item %s (want kind:value)\n", s.c_str()); return 2; }
            const std::string k = s.substr(0, c), v = s.substr(c + 1);
            EmbedItem it; it.value = v;
            if (k == "text") it.kind = EmbedItem::TEXT; else if (k == "image") it.kind = EmbedItem::IMAGE;
            else if (k == "audio") it.kind = EmbedItem::AUDIO; else if (k == "video") it.kind = EmbedItem::VIDEO;
            else { fprintf(stderr, "unknown item kind %s\n", k.c_str()); return 2; }
            its.push_back(std::move(it));
        }
        MmEmbedOptions opt; opt.video_fps = video_fps;
        std::vector<float> vec; int nt = 0;
        auto t0 = clk::now();
        if (!me.embed(its, opt, vec, &err, &nt)) { fprintf(stderr, "embed: %s\n", err.c_str()); return 1; }
        fprintf(stderr, "[embed-media] %d tokens, %.2f s\n", nt, std::chrono::duration<double>(clk::now() - t0).count());
        if (!outp.empty()) { FILE* o = fopen(outp.c_str(), "wb"); if (!o) return 1; fwrite(vec.data(), 4, vec.size(), o); fclose(o); }
        else { for (size_t i = 0; i < vec.size(); ++i) printf("%s%.6f", i ? " " : "", vec[i]); printf("\n"); }
        return 0;
    }
    if (cmd == "mel-dump") {          // Gemma 4 audio frontend only: raw f32 16 kHz mono PCM -> mel chunks (bench/mm/mel_ref format)
        if (media.empty() || outp.empty()) { fprintf(stderr, "mel-dump --media in.f32 --out mel.bin\n"); return 2; }
        std::ifstream f(media, std::ios::binary);
        std::vector<float> pcm; { float v; while (f.read((char*)&v, 4)) pcm.push_back(v); }
        std::vector<Mel> mels;
        if (!gemma4a_audio_mel(pcm.data(), pcm.size(), 128, mels)) { fprintf(stderr, "mel failed\n"); return 1; }
        FILE* o = fopen(outp.c_str(), "wb"); if (!o) return 1;
        int32_t nc = (int32_t)mels.size(); fwrite(&nc, 4, 1, o);
        for (auto& m : mels) { int32_t nm = m.n_mel, nl = m.n_len; fwrite(&nm, 4, 1, o); fwrite(&nl, 4, 1, o); fwrite(m.data.data(), 4, m.data.size(), o);
                               fprintf(stderr, "chunk: n_mel=%d n_len=%d tokens=%d\n", nm, nl, gemma4a_tokens(nl)); }
        fclose(o);
        return 0;
    }
    if (cmd == "mm-embed") {
        if (mmproj.empty() || media.empty() || outp.empty()) { fprintf(stderr, "mm-embed --mmproj P.gguf --media file --out f.bin\n"); return 2; }
        {
            GgufFile peek; std::string perr;
            if (peek.open_file(mmproj.c_str(), &perr) && peek.get_str("clip.vision.projector_type") == "gemma4v")
                return run_mm_embed_g4(mmproj, media, outp, threads > 0 ? threads : 2);
        }
        return run_mm_embed(mmproj, media, outp, threads > 0 ? threads : 2);
    }
    if (model.empty()) usage();
    {
        // dispatch on the model kind: embedding (gemma-embedding2) vs decision (lfm2 d1)
        GgufFile peek; std::string perr;
        if (!peek.open_file(model.c_str(), &perr)) { fprintf(stderr, "cannot open %s: %s\n", model.c_str(), perr.c_str()); return 1; }
        const bool decision = is_lfm2_decision(peek);
        peek.close();
        if (cmd == "decide" || decision) {
            if (!decision) { fprintf(stderr, "%s is not a decision model (use embed/bench)\n", model.c_str()); return 2; }
            if (cmd != "decide" && cmd != "bench") { fprintf(stderr, "%s is a decision model: use `ferro-embed decide --request req.json`\n", model.c_str()); return 2; }
            if (reqf.empty()) { fprintf(stderr, "decide needs --request req.json (or - for stdin)\n"); return 2; }
            if (threads <= 0) threads = std::min(2, std::max(1, (int)std::thread::hardware_concurrency()));
            return run_decide(model, reqf, outp, threads, cmd == "bench" ? std::max(reps, 2) : reps == 5 ? 1 : reps, mmproj);
        }
    }
    if (inp.empty() && tokf.empty()) usage();

    Engine eng;
    std::string err;
    auto t0 = clk::now();
    if (threads <= 0) {
        int hw = (int)std::thread::hardware_concurrency();
        threads = hw <= 0 ? 2 : std::min(2, hw);   // N100-class: 2 spin-tuned cores beat all-4 thrashing
    }
    if (!eng.init(model, threads, &err)) { fprintf(stderr, "init failed: %s\n", err.c_str()); return 1; }
    fprintf(stderr, "[ferro] loaded in %.2fs isa=%d threads=%d dim_out=%d\n",
            std::chrono::duration<double>(clk::now() - t0).count(), isa_level(), threads, eng.dim_out());
    eng.set_use_int8(int8);

    // build packed token stream
    std::vector<int> ids, seg, pos;
    int n_seq = 0;
    if (!tokf.empty()) {
        for (auto& line : read_lines(tokf.c_str())) {
            const char* p = line.c_str(); char* e;
            int n0 = (int)ids.size();
            while (*p) {
                long v = strtol(p, &e, 10);
                if (e == p) break;
                ids.push_back((int)v); seg.push_back(n_seq); pos.push_back((int)(ids.size() - 1 - n0));
                p = e; while (*p == ' ' || *p == '\t') ++p;
            }
            ++n_seq;
        }
    } else if (!inp.empty()) {
        ferro::GgufFile tf; std::string terr;
        if (!tf.open_file(model.c_str(), &terr)) { fprintf(stderr, "[ferro] tokenizer open failed: %s\n", terr.c_str()); return 1; }
        ferro::FerroTokenizer tok;
        if (!tok.load(tf, &terr)) { fprintf(stderr, "[ferro] tokenizer load failed: %s\n", terr.c_str()); return 1; }
        for (auto& line : read_lines(inp.c_str())) {
            auto tids = tok.tokenize(line);
            int n0 = (int)ids.size();
            for (int t : tids) { ids.push_back(t); seg.push_back(n_seq); pos.push_back((int)(ids.size() - 1 - n0)); }
            ++n_seq;
        }
    } else {
        fprintf(stderr, "[ferro] need --in text file or --tokens id file\n");
        return 2;
    }
    fprintf(stderr, "[ferro] %d seqs, %zu tokens\n", n_seq, ids.size());

    std::vector<float> emb((size_t)n_seq * eng.dim_out());
    double best = 1e30, total = 0;
    for (int r = 0; r < reps; ++r) {
        auto s = clk::now();
        size_t i = 0; int seq_off = 0;
        while (i < ids.size()) {
            size_t j = i;
            while (j < ids.size()) {
                size_t j2 = j;
                while (j2 < ids.size() && seg[j2] == seg[j]) ++j2;
                if (j2 - i > (size_t)maxtok && j > i) break;
                j = j2;
            }
            int Tn = (int)(j - i), ns = seg[j - 1] - seg[i] + 1;
            eng.encode(ids.data() + i, seg.data() + i, pos.data() + i, Tn, ns,
                       emb.data() + (size_t)seq_off * eng.dim_out());
            seq_off += ns; i = j;
        }
        double dt = std::chrono::duration<double>(clk::now() - s).count();
        if (r > 0) { best = std::min(best, dt); total += dt; }
        fprintf(stderr, "[ferro] rep %d: %.1f ms  (%.0f tok/s)\n", r, dt * 1e3, ids.size() / dt);
    }
    if (reps > 1)
        fprintf(stderr, "[ferro] BEST %.1f ms | avg(excl warmup) %.1f ms | %.0f tok/s\n",
                best * 1e3, total / (reps - 1) * 1e3, ids.size() / (total / (reps - 1)));

    if (!outp.empty()) {
        if (outp.size() > 4 && outp.substr(outp.size() - 5) == ".json") {
            std::FILE* f = fopen(outp.c_str(), "w");
            fprintf(f, "[\n");
            for (int s = 0; s < n_seq; ++s) {
                fprintf(f, "[");
                for (int c = 0; c < eng.dim_out(); ++c)
                    fprintf(f, "%.8g%s", emb[(size_t)s * eng.dim_out() + c], c + 1 == eng.dim_out() ? "" : ",");
                fprintf(f, "]%s\n", s + 1 == n_seq ? "" : ",");
            }
            fprintf(f, "]\n"); fclose(f);
        } else {
            FILE* f = fopen(outp.c_str(), "wb");
            fwrite(emb.data(), 4, emb.size(), f); fclose(f);
        }
        fprintf(stderr, "[ferro] wrote %s (%zu floats)\n", outp.c_str(), emb.size());
    }
    Pool::get().shutdown();
    return 0;
}
