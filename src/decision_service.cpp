// decision_service.cpp - see decision_service.h
#include "decision_service.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

namespace ferro {

using clk = std::chrono::steady_clock;
static double secs_since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

bool DecisionService::init(const std::string& model, const std::string& mmproj_path, int threads, std::string* err, bool eager_mm) {
    const auto t0 = clk::now();
    model_ = model;
    {                                                              // d1-2B (LFM3-d1) is its own beast: stateless whole-prompt forward, no LFM2 engine
        GgufFile probe;
        std::string perr;
        if (probe.open_file(model.c_str(), &perr) && is_lfm3_d1(probe)) {
            probe.close();
            if (!d1_.init(model, threads, err)) return false;
            if (!tok_.load(d1_.gguf(), err)) { if (err) *err = "tokenizer: " + *err; return false; }
            lfm3d1_load(d1c_, d1_.gguf(), tok_, err);
            lfm3d1_ = true;
            fprintf(stderr, "[ferro] decision model %s (lfm3-d1) loaded in %.2fs threads=%d device=%s\n", name().c_str(), secs_since(t0), threads, d1_.on_gpu() ? "igpu" : "cpu");
            (void)eager_mm;
            return true;
        }
    }
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
    if (lfm3d1_ ? d1_.on_gpu() : eng_.on_gpu()) mm_.upload_gpu();   // vision tower on the iGPU (CPU fallback kept)
    mm_loaded_ = true;
    fprintf(stderr, "[ferro] mmproj %s loaded in %.2fs (vision %s, audio %s)\n", mmproj_.c_str(), secs_since(tl), mm_.has_vision() ? "yes" : "no", mm_.has_audio() ? "yes" : "no");
    (void)err;
    return true;
}

DecisionPrepared DecisionService::prepare(const Json& body) {
    const auto tp0 = clk::now();
    DecisionPrepared P;
    P.batch = body.is_array();
    if (lfm3d1_) {                                                 // d1-2B: one stateless whole-prompt pass per question
        std::vector<const Json*> reqs;
        if (P.batch) {
            if (body.arr.empty()) throw std::invalid_argument("the request array is empty");
            for (auto& r : body.arr) reqs.push_back(&r);
        } else reqs.push_back(&body);
        for (const Json* r : reqs) {
            // files / images ("images" is an alias of "files", as in llama.cpp decision_load_image); each entry
            // is a data-URL image (or a {url|data} object); one <|vision_start|>...<|vision_end|> run per entry
            std::vector<std::string> urls;
            for (const char* key : { "files", "images" }) {
                if (const Json* fs = r->find(key)) {
                    if (!fs->is_array()) throw std::invalid_argument("\"" + std::string(key) + "\" must be an array of data URLs");
                    for (auto& f : fs->arr) {
                        std::string u;
                        if (f.t == Json::String) u = f.s;
                        else if (const Json* j = f.find("url")) { if (!j->is_string()) throw std::invalid_argument(std::string(key) + ": url must be a string"); u = j->s; }
                        else if (const Json* j = f.find("data")) { if (!j->is_string()) throw std::invalid_argument(std::string(key) + ": data must be a string"); u = j->s; }
                        else throw std::invalid_argument(std::string(key) + ": entries must be data URLs (strings)");
                        urls.push_back(std::move(u));
                    }
                    break;
                }
            }
            const bool has_media = !urls.empty();
            if (has_media) {
                if (urls.size() > DecisionPrompter::MAX_MEDIA_FILES) throw std::invalid_argument("too many images, the maximum is " + std::to_string(DecisionPrompter::MAX_MEDIA_FILES));
                ensure_mm(nullptr);
                if (!mm_.has_vision()) throw std::invalid_argument("this mmproj has no vision encoder");
            }
            DecisionPrepared::ReqMedia rm;
            std::vector<int> per_entry;
            if (has_media) {
                for (size_t i = 0; i < urls.size(); ++i) {
                    auto bytes = media_bytes_from_url(urls[i]);
                    MediaItem it = decode_media(bytes, 16000);
                    if (it.is_audio) throw std::invalid_argument("the d1-2B vision path accepts images only (an audio clip arrived)");
                    auto v = lfm3vl_preprocess_image(it.rgb.data(), it.nx, it.ny, mm_.vision_params());
                    per_entry.push_back(lfm3vl_vision_tokens(v, mm_.vision_params()));
                    rm.n_tok += per_entry.back();
                    ++rm.n_files;
                    rm.vin.push_back(std::move(v));
                }
                P.n_images += (int)urls.size();
            }
            auto built = lfm3d1_build(d1c_, *r, per_entry);
            DecisionRequest dq;
            if (const Json* s = r->find("state")) dq.state = *s;
            for (auto& b : built) {
                DecisionQuestion dqn;
                dqn.id = b.id;
                dqn.type = b.type;
                dq.questions.push_back(std::move(dqn));
                DecisionSeq s;
                s.tokens = b.tokens;
                s.type = b.type;
                s.media = -1;
                P.n_tok += b.tokens.size();
                P.seqs.push_back(std::move(s));
            }
            P.lq.push_back(std::move(built));
            P.req_media.push_back(has_media ? (int)P.rmedia.size() : -1);
            if (has_media) P.rmedia.push_back(std::move(rm));
            P.reqs.push_back(std::move(dq));
        }
        P.prep_ms = secs_since(tp0);
        return P;
    }
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
    scores.assign(jobs.size(), {});
    if (lfm3d1_) {                                                 // each question is its own stateless prompt; score_batch packs them into
                                                                   // as few iGPU forwards as fit (segments stay independent)
        std::vector<std::vector<std::vector<float>>> rows(jobs.size());     // vision rows per (job, media block), encoded once
        std::vector<D1Item> items;
        std::vector<size_t> job_of;
        const int d = d1_.model().n_embd;
        for (size_t ji = 0; ji < jobs.size(); ++ji) {
            DecisionPrepared* P = jobs[ji];
            rows[ji].resize(P->rmedia.size());
            for (size_t i = 0; i < P->rmedia.size(); ++i) {
                const auto tv0 = std::chrono::steady_clock::now();
                rows[ji][i] = mm_.encode_vision(P->rmedia[i].vin);
                if (getenv("FERRO_VERBOSE")) fprintf(stderr, "[d1] vision encode: %d rows in %.0f ms\n", P->rmedia[i].n_tok, secs_since(tv0) * 1e3);
            }
            for (size_t ri = 0; ri < P->lq.size(); ++ri)
                for (size_t qi = 0; qi < P->lq[ri].size(); ++qi) {
                    const auto& b = P->lq[ri][qi];
                    D1Item it; it.toks = &b.tokens; it.labels = &b.groups;
                    const int mi = P->req_media[ri];
                    if (mi >= 0) {
                        const auto& embd = rows[ji][mi];
                        if ((int)embd.size() != P->rmedia[mi].n_tok * d) { if (err) *err = "vision encoder produced " + std::to_string(embd.size() / d) + " rows, expected " + std::to_string(P->rmedia[mi].n_tok); return false; }
                        for (auto& run : b.media_runs)
                            for (int j = 0; j < run.second; ++j) it.media_pos.push_back(run.first + j);
                        if (it.media_pos.size() != P->rmedia[mi].n_tok) { if (err) *err = "prompt media rows (" + std::to_string(it.media_pos.size()) + ") do not match the encoded images (" + std::to_string(P->rmedia[mi].n_tok) + ")"; return false; }
                        it.media_rows = embd.data();
                    }
                    items.push_back(std::move(it)); job_of.push_back(ji);
                }
        }
        std::vector<std::vector<float>> sc;
        if (!d1_.score_batch(items, sc, err)) return false;
        for (size_t k = 0; k < items.size(); ++k) scores[job_of[k]].push_back(std::move(sc[k]));
        return true;
    }
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
    if (lfm3d1_) {                                                 // the /v1/systemone shape for d1-2B (llama.cpp format_answer)
        static std::atomic<uint64_t> ctr{ 0 };
        auto hexrand = [] {
            uint64_t x = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count() ^ (0x9E3779B97F4A7C15ull * (++ctr + 1));
            x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33;
            char buf[17];
            snprintf(buf, sizeof buf, "%016llx", (unsigned long long)x);
            return std::string(buf);
        };
        auto resp = [&](size_t ri) {
            Json answers = Json::object();
            uint64_t in_tok = 0;
            size_t k = 0;
            for (size_t rj = 0; rj < ri; ++rj) k += P.lq[rj].size();
            for (size_t qi = 0; qi < P.lq[ri].size(); ++qi, ++k) {
                answers.set(P.lq[ri][qi].id, lfm3d1_answer(d1c_, P.lq[ri][qi], scores[k]));
                int n_media = 0;
                for (auto& run : P.lq[ri][qi].media_runs) n_media += run.second;
                in_tok += P.lq[ri][qi].tokens.size() + n_media;
            }
            Json usage = Json::object();
            usage.set("prompt_tokens", Json::integer((int64_t)in_tok));
            usage.set("total_tokens", Json::integer((int64_t)in_tok));
            Json out = Json::object();
            out.set("id", Json::str("sys1-" + hexrand()));
            out.set("object", Json::str("systemone.completion"));
            out.set("created", Json::integer((int64_t)std::time(nullptr)));
            out.set("model", Json::str(d1_.gguf().get_str("general.name")));
            out.set("answers", answers);
            out.set("usage", usage);
            return out;
        };
        if (!P.batch) return resp(0);
        Json arr = Json::array();
        for (size_t ri = 0; ri < P.lq.size(); ++ri) arr.push(resp(ri));
        return arr;
    }
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
