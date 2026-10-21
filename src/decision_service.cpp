// decision_service.cpp - see decision_service.h
#include "decision_service.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ferro {

using clk = std::chrono::steady_clock;
static double secs_since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

bool DecisionService::init(const std::string& model, const std::string& mmproj_path, int threads, std::string* err, bool eager_mm) {
    const auto t0 = clk::now();
    model_ = model;
    if (!eng_.init(model, threads, err)) return false;
    if (!tok_.load(eng_.gguf(), err)) { if (err) *err = "tokenizer: " + *err; return false; }
    if (!dp_.init(eng_.gguf(), &tok_, err)) { if (err) *err = "decision: " + *err; return false; }
    mmproj_ = mmproj_path;
    fprintf(stderr, "[ferro] decision model %s loaded in %.2fs threads=%d device=%s\n", name().c_str(), secs_since(t0), threads, eng_.on_gpu() ? "igpu" : "cpu");
    if (eager_mm) {                                 // server: pay the projector load now, not on the first image request (none found = text-only d1)
        if (mmproj_.empty()) mmproj_ = find_mmproj_near(model_);
        if (!mmproj_.empty()) {
            try { std::string e; ensure_mm(&e); } catch (const std::exception& e) { if (err) *err = e.what(); return false; }
        }
    }
    return true;
}

// loads the multimodal projector once; throws invalid_argument when there is none (request error), runtime_error when it cannot be read
bool DecisionService::ensure_mm(std::string* err) {
    std::lock_guard<std::mutex> g(mm_mu_);
    if (mm_loaded_) return true;
    if (mmproj_.empty()) mmproj_ = find_mmproj_near(model_);
    if (mmproj_.empty()) throw std::invalid_argument("this request has images/audio but no mmproj GGUF was given (--mmproj) or found next to the model");
    std::string merr;
    const auto tl = clk::now();
    if (!mm_.load(mmproj_, &merr)) throw std::runtime_error("mmproj " + mmproj_ + ": " + merr);
    if (eng_.on_gpu()) mm_.upload_gpu();
    mm_loaded_ = true;
    fprintf(stderr, "[ferro] mmproj %s loaded in %.2fs (vision %s, audio %s)\n", mmproj_.c_str(), secs_since(tl), mm_.has_vision() ? "yes" : "no", mm_.has_audio() ? "yes" : "no");
    (void)err;
    return true;
}

DecisionPrepared DecisionService::prepare(const Json& body) {
    const auto tp0 = clk::now();
    DecisionPrepared P;
    P.batch = body.is_array();
    if (P.batch) {
        if (body.arr.empty()) throw std::invalid_argument("the request array is empty");
        for (auto& r : body.arr) P.reqs.push_back(dp_.parse(r));
    } else {
        P.reqs.push_back(dp_.parse(body));
    }
    bool any_media = false;
    for (auto& r : P.reqs) any_media = any_media || !r.files.empty();
    if (any_media) ensure_mm(nullptr);

    for (auto& r : P.reqs) {
        if (r.files.empty()) { P.req_media.push_back(-1); continue; }
        DecisionPrepared::ReqMedia rm;
        std::vector<MediaItem> items;
        for (auto& f : r.files) items.push_back(decode_media(f, 16000));
        for (auto& it : items) rm.is_audio = rm.is_audio || it.is_audio;
        if (rm.is_audio && items.size() > 1) throw std::invalid_argument("a request has images or one audio clip, not both");
        if (rm.is_audio) {
            if (!mm_.has_audio()) throw std::invalid_argument("this mmproj has no audio encoder");
            if (r.file_specs.size() == 1 && !r.file_specs[0].crops.empty()) throw std::invalid_argument("crop / crops apply to images, not audio");
            if (!d1omni_audio_mel(items[0].pcm.data(), items[0].pcm.size(), mm_.n_mel(), rm.mel)) throw std::invalid_argument("the audio clip is empty");
            rm.n_tok = audio_tokens(rm.mel);
            rm.n_files = 1;
        } else {
            if (!mm_.has_vision()) throw std::invalid_argument("this mmproj has no vision encoder");
            for (size_t i = 0; i < items.size(); ++i) {
                const MediaItem& it = items[i];
                const FileSpec& fs = r.file_specs[i];
                auto add = [&](const uint8_t* rgb, int nx, int ny) {
                    auto v = d1omni_preprocess_image(rgb, nx, ny, mm_.vision_params());
                    for (auto& x : v) { rm.n_tok += vision_tokens(x, mm_.vision_params()); rm.vin.push_back(std::move(x)); }
                    ++rm.n_files;
                };
                if (fs.full) add(it.rgb.data(), it.nx, it.ny);
                for (size_t c = 0; c < fs.crops.size(); ++c) {
                    std::vector<uint8_t> cut; int ox = 0, oy = 0; std::string cerr;
                    if (!crop_rgb(it.rgb.data(), it.nx, it.ny, fs.crops[c], cut, ox, oy, &cerr))
                        throw std::invalid_argument((fs.crops.size() > 1 ? "crops[" + std::to_string(c) + "]: " : std::string("crop: ")) + cerr);
                    add(cut.data(), ox, oy);
                }
            }
        }
        P.n_images += rm.is_audio ? 0 : rm.n_files;
        P.req_media.push_back((int)P.rmedia.size());
        P.rmedia.push_back(std::move(rm));
    }
    for (size_t ri = 0; ri < P.reqs.size(); ++ri) {
        DecisionMediaInfo mi;
        if (P.req_media[ri] >= 0) { const auto& rm = P.rmedia[P.req_media[ri]]; mi.n_tokens = rm.n_tok; mi.is_audio = rm.is_audio; mi.n_files = rm.n_files; }
        for (auto& q : P.reqs[ri].questions) {
            DecisionSeq sq = dp_.build(P.reqs[ri], q, mi);
            sq.media = P.req_media[ri];
            P.seqs.push_back(std::move(sq));
        }
    }
    for (auto& s : P.seqs) P.n_tok += s.tokens.size() + s.n_media;
    for (auto& rm : P.rmedia) P.n_media_tok += rm.n_tok;
    P.prep_ms = secs_since(tp0) * 1e3;
    if (getenv("FERRO_DECISION_DUMP")) {
        size_t k = 0;
        for (size_t ri = 0; ri < P.reqs.size(); ++ri) for (auto& q : P.reqs[ri].questions) {
            DecisionMediaInfo mi;
            if (P.req_media[ri] >= 0) { mi.n_tokens = P.rmedia[P.req_media[ri]].n_tok; mi.is_audio = P.rmedia[P.req_media[ri]].is_audio; mi.n_files = P.rmedia[P.req_media[ri]].n_files; }
            fprintf(stderr, "[prompt %s] (media %d) %s\n[tokens]", q.id.c_str(), mi.n_tokens, dp_.render_debug(P.reqs[ri], q, mi).c_str());
            for (int t : P.seqs[k].tokens) fprintf(stderr, " %d", t);
            fprintf(stderr, "\n"); ++k;
        }
    }
    return P;
}

bool DecisionService::infer(const std::vector<DecisionPrepared*>& jobs, std::vector<std::vector<std::vector<float>>>& scores, std::string* err) {
    // encode every media block of every job once, then one packed forward over all question sequences
    std::vector<std::vector<float>> membd;
    size_t n_media = 0; for (auto* j : jobs) n_media += j->rmedia.size();
    membd.reserve(n_media);
    std::vector<DecisionMediaEmbd> media; media.reserve(n_media);
    std::vector<DecisionSeq> seqs;
    std::vector<size_t> seq_off;                                    // first sequence of each job
    for (auto* j : jobs) {
        const int moff = (int)media.size();
        for (size_t i = 0; i < j->rmedia.size(); ++i) {
            auto& rm = j->rmedia[i];
            membd.push_back(rm.is_audio ? mm_.encode_audio(rm.mel) : mm_.encode_vision(rm.vin));
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
                if (all.size() == membd.back().size()) { membd.back() = all; fprintf(stderr, "[debug] media %zu: encoder output replaced by %s\n", i, ov); }
            }
            DecisionMediaEmbd me; me.embd = membd.back().data(); me.n_tok = (int)(membd.back().size() / eng_.model().n_embd);
            if (me.n_tok != rm.n_tok) { if (err) *err = "media encoder produced " + std::to_string(me.n_tok) + " tokens, expected " + std::to_string(rm.n_tok); return false; }
            media.push_back(me);
        }
        seq_off.push_back(seqs.size());
        for (auto s : j->seqs) { if (s.media >= 0) s.media += moff; seqs.push_back(std::move(s)); }
    }
    seq_off.push_back(seqs.size());
    std::vector<std::vector<float>> all;
    static const bool sequential = getenv("FERRO_DECISION_SEQUENTIAL") != nullptr;      // one request per forward (latency mode)
    try {
        if (!sequential) eng_.score(seqs, all, media);
        else {
            size_t k0 = 0;
            for (auto* j : jobs) for (auto& rq : j->reqs) {
                std::vector<DecisionSeq> part(seqs.begin() + k0, seqs.begin() + k0 + rq.questions.size());
                std::vector<std::vector<float>> sc;
                eng_.score(part, sc, media);
                all.insert(all.end(), sc.begin(), sc.end());
                k0 += rq.questions.size();
            }
        }
    } catch (const std::exception& e) { if (err) *err = std::string("engine error: ") + e.what(); return false; }
    scores.assign(jobs.size(), {});
    for (size_t i = 0; i < jobs.size(); ++i)
        for (size_t k = seq_off[i]; k < seq_off[i + 1]; ++k) scores[i].push_back(std::move(all[k]));
    return true;
}

Json DecisionService::format(const DecisionPrepared& P, const std::vector<std::vector<float>>& scores) const {
    Json all = Json::array();
    size_t k = 0;
    for (size_t ri = 0; ri < P.reqs.size(); ++ri) {
        auto& r = P.reqs[ri];
        Json answers = Json::object();
        int64_t ntok = 0;
        for (auto& q : r.questions) { ntok += (int64_t)(P.seqs[k].tokens.size() + P.seqs[k].n_media); answers.set(q.id, dp_.answer(q, scores[k], P.req_media[ri] >= 0)); ++k; }
        Json res = Json::object();
        res.set("model", Json::str(eng_.gguf().get_str("general.name")));
        res.set("answers", answers);
        Json usage = Json::object();
        usage.set("input_tokens", Json::integer(ntok));
        usage.set("output_tokens", Json::integer(0));
        res.set("usage", usage);
        all.push(res);
    }
    return P.batch ? all : all.arr[0];
}

}  // namespace ferro
