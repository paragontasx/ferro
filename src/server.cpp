// server.cpp - see server.h
//
// Threads: the main thread accepts connections; each connection gets a handler thread (HTTP/1.1 keep-alive); ONE inference worker
// owns the engine, the GPU and the thread pool (the device is a single in-order queue, so inference is serialised by design).
// Scheduling: the worker takes the oldest job; if it is text-only it also takes every following text-only job (and briefly waits
// for requests that are already in flight) so they share packed forward passes - that is where the iGPU throughput is
// (16 x 188 tokens: 850 tok/s; one short sequence at a time is several times slower per token). Image / audio / video jobs run
// alone, in arrival order.
#include "server.h"
#include "decision_service.h"
#include "embed_mm.h"
#include "chat_template.h"
#include "json.h"
#include "model_lfm2_lm.h"
#include "sampler.h"
#include "threadpool.h"
#include "tokenizer_bpe.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <climits>
#include <ctime>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace ferro {
namespace {

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// ---------------------------------------------------------------- base64
static bool b64_decode(const char* s, size_t n, std::vector<uint8_t>& out) {
    static int8_t tab[256]; static bool init = false;
    if (!init) {
        memset(tab, -1, sizeof tab);
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) tab[(uint8_t)a[i]] = (int8_t)i;
        tab[(uint8_t)'-'] = 62; tab[(uint8_t)'_'] = 63;                  // url-safe alphabet too
        init = true;
    }
    out.clear(); out.reserve(n / 4 * 3 + 3);
    uint32_t acc = 0; int bits = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t c = (uint8_t)s[i];
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int v = tab[c];
        if (v < 0) return false;
        acc = (acc << 6) | (uint32_t)v; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((uint8_t)((acc >> bits) & 0xFF)); }
    }
    return true;
}
static std::string b64_encode(const uint8_t* p, size_t n) {
    static const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o; o.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        o += a[(v >> 18) & 63]; o += a[(v >> 12) & 63];
        o += i + 1 < n ? a[(v >> 6) & 63] : '='; o += i + 2 < n ? a[v & 63] : '=';
    }
    return o;
}

// ---------------------------------------------------------------- request model
struct BadRequest : std::runtime_error { using std::runtime_error::runtime_error; };

// data:[<mime>][;base64],<payload> -> bytes. Remote URLs and local paths are refused on purpose (no SSRF, no file reads).
static std::vector<uint8_t> decode_data_url(const std::string& u, const char* what) {
    if (u.compare(0, 5, "data:") != 0)
        throw BadRequest(std::string(what) + ": only data: URLs (base64) are accepted - remote URLs and file paths are not fetched");
    const size_t comma = u.find(',');
    if (comma == std::string::npos) throw BadRequest(std::string(what) + ": malformed data URL");
    if (u.substr(5, comma - 5).find(";base64") == std::string::npos) throw BadRequest(std::string(what) + ": data URL must be base64");
    std::vector<uint8_t> b;
    if (!b64_decode(u.data() + comma + 1, u.size() - comma - 1, b) || b.empty()) throw BadRequest(std::string(what) + ": invalid base64");
    return b;
}
static const Json* sub(const Json& j, const char* k) { return j.is_object() ? j.find(k) : nullptr; }
static std::string url_of(const Json& v) {          // "image_url": "data:..." | {"url": "data:..."}
    if (v.is_string()) return v.s;
    if (const Json* u = sub(v, "url")) if (u->is_string()) return u->s;
    return std::string();
}

// one content part -> one EmbedItem
static EmbedItem parse_part(const Json& p, bool has_mm) {
    if (p.is_string()) { EmbedItem it; it.kind = EmbedItem::TEXT; it.value = p.s; return it; }
    const Json* t = sub(p, "type");
    if (!t || !t->is_string()) throw BadRequest("every content part needs a \"type\" (text, image_url, input_audio, audio_url, video_url)");
    EmbedItem it;
    auto need_mm = [&](const char* k) { if (!has_mm) throw BadRequest(std::string("this server was started without --mmproj: ") + k + " inputs are unavailable"); };
    if (t->s == "text") {
        const Json* x = sub(p, "text");
        if (!x || !x->is_string()) throw BadRequest("text part without a \"text\" string");
        it.kind = EmbedItem::TEXT; it.value = x->s;
    } else if (t->s == "image_url") {
        need_mm("image");
        const Json* x = sub(p, "image_url");
        it.kind = EmbedItem::IMAGE; it.bytes = decode_data_url(x ? url_of(*x) : std::string(), "image_url");
    } else if (t->s == "input_audio") {
        need_mm("audio");
        const Json* x = sub(p, "input_audio"); const Json* d = x ? sub(*x, "data") : nullptr;
        if (!d || !d->is_string()) throw BadRequest("input_audio needs {\"data\": <base64>, \"format\": ...}");
        it.kind = EmbedItem::AUDIO;
        if (!b64_decode(d->s.data(), d->s.size(), it.bytes) || it.bytes.empty()) throw BadRequest("input_audio: invalid base64");
    } else if (t->s == "audio_url") {
        need_mm("audio");
        const Json* x = sub(p, "audio_url");
        it.kind = EmbedItem::AUDIO; it.bytes = decode_data_url(x ? url_of(*x) : std::string(), "audio_url");
    } else if (t->s == "video_url") {
        need_mm("video");
        const Json* x = sub(p, "video_url");
        it.kind = EmbedItem::VIDEO; it.bytes = decode_data_url(x ? url_of(*x) : std::string(), "video_url");
    } else throw BadRequest("unknown content part type \"" + t->s + "\"");
    return it;
}

struct Job {
    std::vector<std::vector<EmbedItem>> inputs;   // one embedding per entry
    bool text_only = true;
    float video_fps = 4.0f;
    // result
    std::vector<std::vector<float>> out;
    std::vector<int> n_tokens;
    std::string err; int status = 200;
    double queue_ms = 0, run_ms = 0; int batch_seqs = 0;
    clk::time_point enq = clk::now();
    std::mutex m; std::condition_variable cv; bool done = false;
};

// A text generation request (chat or completion). The HTTP handler thread builds it and waits on `events`; the worker owns everything
// else (sequence state, sampler) and appends text deltas as tokens are produced.
struct Delta { bool reasoning; std::string text; };
struct GenJob {
    // request
    std::vector<int> prompt;
    SamplerParams sp;
    int max_tokens = 0;
    std::vector<std::string> stops;
    bool stream = false, chat = true, include_usage = false;
    // worker-side state
    LmSeq seq;
    std::unique_ptr<Sampler> sampler;
    std::vector<int> hist;
    int next = 0, n_gen = 0, mode = 0;            // mode: 0 = content, 1 = reasoning (inside <think>)
    std::string pend;                             // decoded text not yet emitted (incomplete UTF-8 / possible stop-string prefix)
    double prompt_ms = 0, gen_ms = 0;
    clk::time_point t_enq = clk::now(), t_gen0;
    // results / events
    std::mutex m; std::condition_variable cv;
    std::deque<Delta> events;
    bool done = false;
    std::string finish = "stop", err; int status = 200;
    std::atomic<bool> cancelled{false};
};

// A decision request (POST /v1/systemone). The worker does everything (parse, decode + crop media, encode, score) so the CPU-side work of
// several queued requests is done back to back and their questions share one forward pass.
struct DecJob {
    std::string body;
    std::string out, err; int status = 200;
    double queue_ms = 0, run_ms = 0, prep_ms = 0; size_t n_tok = 0, n_q = 0; int n_img = 0;
    clk::time_point enq = clk::now();
    std::mutex m; std::condition_variable cv; bool done = false;
};

// ---------------------------------------------------------------- server state
struct Server {
    ServerOptions o;
    MmEmbedder me;
    bool has_emb = false;
    // language model
    LmEngine lm; BpeTokenizer ltok;
    bool has_lm = false;
    // decision model (d1-omni)
    DecisionService d1; bool has_d1 = false; std::string d1_name;
    std::deque<std::shared_ptr<DecJob>> dq;       // guarded by qm
    std::atomic<uint64_t> n_d1_req{0}, n_d1_q{0}, n_d1_tok{0}, n_d1_pass{0}, d1_us{0};
    ChatTemplateOptions chat_opt;                 // template variant, detected from the GGUF
    std::string lm_name; int think_open = -1, think_close = -1, eos_id = 7;
    std::deque<std::shared_ptr<GenJob>> gq;       // waiting for a slot (guarded by qm)
    std::vector<std::shared_ptr<GenJob>> active;  // worker only
    std::atomic<int> n_active{0};
    std::vector<float> lg;                        // worker only: logits [slots][vocab]
    std::atomic<uint64_t> n_gen_req{0}, n_gen_tok{0}, n_prompt_tok{0};
    std::atomic<uint64_t> gen_us{0}, pp_us{0};
    std::string model_name;
    bool has_mm = false;
    int dim = 768;
    std::atomic<bool> stop{false};
    int listen_fd = -1;
    // job queue
    std::mutex qm; std::condition_variable qcv; std::deque<std::shared_ptr<Job>> q;
    std::atomic<int> inflight{0};                 // embedding requests between "parsed" and "answered"
    std::atomic<int> conns{0};
    // stats
    std::atomic<uint64_t> n_req{0}, n_err{0}, n_tok{0}, n_emb{0}, n_batches{0};
    std::atomic<uint64_t> busy_us{0};
    clk::time_point t_start = clk::now();
};
static Server* g_srv = nullptr;
static void on_signal(int) { if (g_srv) { g_srv->stop = true; if (g_srv->listen_fd >= 0) shutdown(g_srv->listen_fd, SHUT_RDWR); } }

// ---------------------------------------------------------------- generation (worker side)
static size_t utf8_complete_prefix(const std::string& s) {         // length of the longest prefix that does not end inside a UTF-8 sequence
    size_t n = s.size(), i = n;
    for (int back = 0; back < 4 && i > 0; ++back) {
        --i;
        const unsigned char c = (unsigned char)s[i];
        if ((c & 0xC0) == 0x80) continue;                           // continuation byte: keep looking for the lead byte
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return (n - i) < need ? i : n;
    }
    return n;
}
static void gen_emit(GenJob& j, const std::string& text) {
    if (text.empty()) return;
    { std::lock_guard<std::mutex> g(j.m); j.events.push_back({j.mode == 1, text}); }
    j.cv.notify_all();
}
// Emit what is safe to show. Holds back an incomplete UTF-8 tail and any suffix that could still grow into a stop string.
// Returns true when a stop string was found (text before it has been emitted).
static bool gen_drain(GenJob& j, bool final) {
    size_t valid = final ? j.pend.size() : utf8_complete_prefix(j.pend);
    if (j.mode == 0 && !j.stops.empty()) {
        size_t best = std::string::npos;
        for (const auto& st : j.stops) { const size_t p = j.pend.find(st); if (p != std::string::npos) best = std::min(best, p); }
        if (best != std::string::npos) { gen_emit(j, j.pend.substr(0, best)); j.pend.clear(); return true; }
        if (!final) {
            size_t hold = 0;
            for (const auto& st : j.stops)
                for (size_t k = std::min(st.size() - 1, j.pend.size()); k >= 1; --k)
                    if (j.pend.compare(j.pend.size() - k, k, st, 0, k) == 0) { hold = std::max(hold, k); break; }
            valid = std::min(valid, j.pend.size() - hold);
        }
    }
    gen_emit(j, j.pend.substr(0, valid));
    j.pend.erase(0, valid);
    return false;
}
// Consume one sampled token. Returns true when the sequence is finished (j.finish says why).
static bool gen_token(Server& S, GenJob& j, int id) {
    const bool split = S.o.reasoning_format != "none";
    if (id == S.eos_id) { gen_drain(j, true); j.finish = "stop"; return true; }
    j.hist.push_back(id); j.n_gen++; j.next = id;
    if (split && id == S.think_open) { gen_drain(j, true); j.mode = 1; }
    else if (split && id == S.think_close) { gen_drain(j, true); j.mode = 0; }
    else {
        j.pend += S.ltok.decode_piece(id);
        if (gen_drain(j, false)) { j.finish = "stop"; return true; }
    }
    if (j.n_gen >= j.max_tokens || j.seq.n_past + 1 >= S.lm.max_ctx()) { gen_drain(j, true); j.finish = "length"; return true; }
    return false;
}
static void gen_finish(Server& S, const std::shared_ptr<GenJob>& jp, const char* err = nullptr, int status = 500) {
    GenJob& j = *jp;
    S.lm.release(j.seq);
    j.gen_ms = ms_since(j.t_gen0);
    S.n_gen_req++; S.n_gen_tok += (uint64_t)j.n_gen; S.n_prompt_tok += j.prompt.size(); S.gen_us += (uint64_t)(j.gen_ms * 1000); S.pp_us += (uint64_t)(j.prompt_ms * 1000);
    { std::lock_guard<std::mutex> g(j.m); if (err) { j.err = err; j.status = status; } j.done = true; }
    j.cv.notify_all();
}
static void gen_start(Server& S, const std::shared_ptr<GenJob>& jp) {
    GenJob& j = *jp;
    const int V = S.lm.n_vocab();
    j.t_gen0 = clk::now();
    S.lm.reset(j.seq);
    std::string err;
    const auto t0 = clk::now();
    const int n = (int)j.prompt.size();
    bool ok = false;
    try { ok = S.lm.step({&j.seq}, {j.prompt.data()}, {n}, S.lg.data(), &err); } catch (const std::exception& e) { err = e.what(); }
    j.prompt_ms = ms_since(t0);
    if (!ok) { gen_finish(S, jp, (err.empty() ? "prefill failed" : err).c_str()); return; }
    j.hist = j.prompt;
    j.sampler.reset(new Sampler(j.sp));
    const int tok = j.sampler->sample(S.lg.data(), V, j.hist);
    if (gen_token(S, j, tok)) { gen_finish(S, jp); return; }
    S.active.push_back(jp); S.n_active = (int)S.active.size();
}
// One scheduler iteration for generation: admit waiting requests into free slots (prefill), then one batched decode step.
static void gen_iteration(Server& S) {
    const int V = S.lm.n_vocab();
    while (true) {
        std::shared_ptr<GenJob> j;
        {
            std::lock_guard<std::mutex> lk(S.qm);
            if (S.gq.empty() || (int)S.active.size() >= S.lm.n_slots()) break;
            j = S.gq.front(); S.gq.pop_front();
        }
        if (j->cancelled) { gen_finish(S, j); continue; }
        gen_start(S, j);
        if (S.stop) break;
    }
    // drop cancelled / shutting-down sequences
    for (size_t i = 0; i < S.active.size();) {
        if (S.active[i]->cancelled || S.stop) { gen_finish(S, S.active[i]); S.active.erase(S.active.begin() + (long)i); } else ++i;
    }
    S.n_active = (int)S.active.size();
    if (S.active.empty()) return;
    std::vector<LmSeq*> seqs; std::vector<const int*> toks; std::vector<int> ns;
    for (auto& j : S.active) { seqs.push_back(&j->seq); toks.push_back(&j->next); ns.push_back(1); }
    std::string err; bool ok = false;
    try { ok = S.lm.step(seqs, toks, ns, S.lg.data(), &err); } catch (const std::exception& e) { err = e.what(); }
    if (!ok) {
        for (auto& j : S.active) gen_finish(S, j, (err.empty() ? "decode failed" : err).c_str());
        S.active.clear(); S.n_active = 0; return;
    }
    std::vector<std::shared_ptr<GenJob>> keep;
    for (size_t i = 0; i < S.active.size(); ++i) {
        auto& j = S.active[i];
        const int tok = j->sampler->sample(S.lg.data() + i * (size_t)V, V, j->hist);
        if (gen_token(S, *j, tok)) gen_finish(S, j); else keep.push_back(j);
    }
    S.active.swap(keep); S.n_active = (int)S.active.size();
}

// ---------------------------------------------------------------- inference worker
// One scheduler step for decision requests: take the queued ones (up to the token budget), prepare each (a bad request only fails itself),
// run ONE packed forward over all their questions, answer them.
static void d1_iteration(Server& S) {
    std::vector<std::shared_ptr<DecJob>> jobs; std::vector<DecisionPrepared> preps; std::vector<std::shared_ptr<DecJob>> ok_jobs;
    size_t budget = 0;
    const auto t0 = clk::now();
    while (jobs.size() < 32 && (int)budget < S.o.d1_batch_tokens) {
        std::shared_ptr<DecJob> j;
        { std::lock_guard<std::mutex> g(S.qm); if (S.dq.empty()) break; j = S.dq.front(); S.dq.pop_front(); }
        jobs.push_back(j);
        j->queue_ms = ms_since(j->enq);
        try {
            DecisionPrepared P = S.d1.prepare(json_parse(j->body));
            j->prep_ms = P.prep_ms; j->n_tok = P.n_tok; j->n_q = P.seqs.size(); j->n_img = P.n_images;
            budget += P.n_tok;
            preps.push_back(std::move(P)); ok_jobs.push_back(j);
        } catch (const std::invalid_argument& e) { j->err = e.what(); j->status = 400; }
        catch (const std::exception& e) { j->err = std::string("invalid request: ") + e.what(); j->status = e.what()[0] == 'm' && std::string(e.what()).rfind("mmproj", 0) == 0 ? 500 : 400; }
    }
    if (jobs.empty()) return;
    if (!ok_jobs.empty()) {
        std::vector<DecisionPrepared*> ptrs; for (auto& p : preps) ptrs.push_back(&p);
        std::vector<std::vector<std::vector<float>>> scores; std::string err;
        const bool ok = S.d1.infer(ptrs, scores, &err);
        for (size_t i = 0; i < ok_jobs.size(); ++i) {
            DecJob& j = *ok_jobs[i];
            if (!ok) { j.err = err.empty() ? "decision inference failed" : err; j.status = 500; continue; }
            try { j.out = json_dump(S.d1.format(preps[i], scores[i])); S.n_d1_req++; S.n_d1_q += j.n_q; S.n_d1_tok += j.n_tok; }
            catch (const std::exception& e) { j.err = std::string("decision failed: ") + e.what(); j.status = 500; }
        }
        S.n_d1_pass++;
    }
    const double run = ms_since(t0);
    S.d1_us += (uint64_t)(run * 1000); S.busy_us += (uint64_t)(run * 1000);
    for (auto& j : jobs) { j->run_ms = run; { std::lock_guard<std::mutex> g(j->m); j->done = true; } j->cv.notify_all(); }
}

static void worker_loop(Server& S, std::promise<std::string>& ready) {
    {
        std::string err;
        if (S.has_emb && !S.me.init(S.o.model, S.o.mmproj, S.o.threads, &err)) { ready.set_value(err.empty() ? "embedding model init failed" : err); return; }
        if (S.has_lm) {
            if (!S.lm.init(S.o.lm, S.o.threads, S.o.lm_ctx, &err, S.o.lm_slots)) { ready.set_value(err.empty() ? "language model init failed" : "language model: " + err); return; }
            if (!S.ltok.load(S.lm.model().gguf, &err)) { ready.set_value("language model tokenizer: " + err); return; }
            S.chat_opt.typed_content = lfm_template_is_typed(S.lm.model().gguf.get_str("tokenizer.chat_template"));
            S.think_open = S.ltok.token_id("<think>"); S.think_close = S.ltok.token_id("</think>"); S.eos_id = S.lm.model().eos_id;
            S.lg.assign((size_t)S.lm.n_slots() * (size_t)S.lm.n_vocab(), 0.0f);
        }
        if (S.has_d1 && !S.d1.init(S.o.d1, S.o.d1_mmproj, S.o.threads, &err, true)) { ready.set_value(err.empty() ? "decision model init failed" : "decision model: " + err); return; }
        ready.set_value(std::string());
    }
    MmEmbedOptions mo;
    while (true) {
        std::vector<std::shared_ptr<Job>> batch;
        {
            std::unique_lock<std::mutex> lk(S.qm);
            S.qcv.wait(lk, [&] { return S.stop || !S.q.empty() || !S.gq.empty() || !S.dq.empty() || S.n_active.load() > 0; });
            if (S.stop && S.q.empty() && S.gq.empty() && S.dq.empty() && S.n_active.load() == 0) break;
            if (!S.q.empty()) {
                batch.push_back(S.q.front()); S.q.pop_front();
                if (batch[0]->text_only) {
                    // gather more text-only jobs; wait (briefly) only while other requests are demonstrably on their way
                    size_t seqs = batch[0]->inputs.size();
                    const auto deadline = clk::now() + std::chrono::milliseconds(S.o.batch_wait_ms);
                    while (seqs < 32) {
                        if (!S.q.empty()) {
                            if (!S.q.front()->text_only || seqs + S.q.front()->inputs.size() > 32) break;
                            seqs += S.q.front()->inputs.size(); batch.push_back(S.q.front()); S.q.pop_front(); continue;
                        }
                        if (S.inflight.load() <= (int)batch.size() || clk::now() >= deadline) break;
                        S.qcv.wait_until(lk, deadline);
                    }
                }
            }
        }
        if (!batch.empty()) {
            const auto t0 = clk::now();
            for (auto& j : batch) j->queue_ms = ms_since(j->enq);
            if (batch[0]->text_only) {
                std::vector<std::string> texts; std::vector<std::pair<size_t, size_t>> owner;     // (job, input index)
                for (size_t ji = 0; ji < batch.size(); ++ji)
                    for (size_t k = 0; k < batch[ji]->inputs.size(); ++k) { texts.push_back(batch[ji]->inputs[k][0].value); owner.push_back({ji, k}); }
                std::vector<std::vector<float>> out; std::vector<int> nt; std::vector<std::string> errs;
                try { S.me.embed_text_batch(texts, S.o.max_batch_tokens, out, nt, errs); }
                catch (const std::exception& e) { errs.assign(texts.size(), std::string("engine error: ") + e.what()); out.assign(texts.size(), {}); nt.assign(texts.size(), 0); }
                for (auto& j : batch) { j->out.assign(j->inputs.size(), {}); j->n_tokens.assign(j->inputs.size(), 0); j->batch_seqs = (int)texts.size(); }
                for (size_t i = 0; i < owner.size(); ++i) {
                    Job& j = *batch[owner[i].first];
                    j.out[owner[i].second] = std::move(out[i]); j.n_tokens[owner[i].second] = nt[i];
                    if (!errs[i].empty() && j.err.empty()) { j.err = errs[i]; j.status = errs[i].compare(0, 12, "engine error") == 0 ? 500 : 400; }
                }
                S.n_batches++;
            } else {
                Job& j = *batch[0];
                MmEmbedOptions opt = mo; opt.video_fps = j.video_fps;
                j.out.assign(j.inputs.size(), {}); j.n_tokens.assign(j.inputs.size(), 0); j.batch_seqs = 1;
                for (size_t k = 0; k < j.inputs.size() && j.err.empty(); ++k) {
                    std::string err; int nt = 0;
                    try {
                        if (!S.me.embed(j.inputs[k], opt, j.out[k], &err, &nt)) { j.err = err.empty() ? "embedding failed" : err; j.status = 400; }
                    } catch (const std::exception& e) { j.err = std::string("engine error: ") + e.what(); j.status = 500; }
                    j.n_tokens[k] = nt;
                    j.inputs[k].clear(); j.inputs[k].shrink_to_fit();                // free the decoded media early
                }
                S.n_batches++;
            }
            const double run = ms_since(t0);
            S.busy_us += (uint64_t)(run * 1000);
            for (auto& j : batch) {
                j->run_ms = run;
                { std::lock_guard<std::mutex> g(j->m); j->done = true; }
                j->cv.notify_all();
            }
        }
        if (S.has_d1) d1_iteration(S);
        if (S.has_lm) gen_iteration(S);
    }
    // shutting down: fail whatever is still queued
    std::lock_guard<std::mutex> lk(S.qm);
    for (auto& j : S.q) { j->err = "server is shutting down"; j->status = 503; { std::lock_guard<std::mutex> g(j->m); j->done = true; } j->cv.notify_all(); }
    S.q.clear();
    for (auto& j : S.gq) { std::lock_guard<std::mutex> g(j->m); j->err = "server is shutting down"; j->status = 503; j->done = true; j->cv.notify_all(); }
    S.gq.clear();
    for (auto& j : S.dq) { std::lock_guard<std::mutex> g(j->m); j->err = "server is shutting down"; j->status = 503; j->done = true; j->cv.notify_all(); }
    S.dq.clear();
}

// ---------------------------------------------------------------- HTTP
struct HttpReq { std::string method, target, path; std::map<std::string, std::string> h; std::string body; bool keep = true; };

static bool send_all(int fd, const char* p, size_t n) {
    while (n) { ssize_t w = send(fd, p, n, MSG_NOSIGNAL); if (w <= 0) return false; p += w; n -= (size_t)w; }
    return true;
}
static const char* reason(int s) {
    switch (s) { case 200: return "OK"; case 400: return "Bad Request"; case 401: return "Unauthorized"; case 404: return "Not Found";
                 case 405: return "Method Not Allowed"; case 413: return "Payload Too Large"; case 500: return "Internal Server Error";
                 case 503: return "Service Unavailable"; default: return "OK"; }
}
static bool respond(int fd, int status, const std::string& body, bool keep, const char* ctype = "application/json", const char* extra = "") {
    char head[384];
    int n = snprintf(head, sizeof head, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: %s\r\n%s\r\n",
                     status, reason(status), ctype, body.size(), keep ? "keep-alive" : "close", extra);
    return send_all(fd, head, (size_t)n) && send_all(fd, body.data(), body.size());
}
static std::string jesc(const std::string& s) {
    std::string o; o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) { case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
                     default: if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c; }
    }
    return o;
}
static std::string err_json(const std::string& msg, const char* type) { return "{\"error\":{\"message\":\"" + jesc(msg) + "\",\"type\":\"" + type + "\",\"code\":null}}"; }

// 0 = ok, 1 = closed/timeout/garbage (drop the connection), 2 = too large (already answered)
static int read_request(int fd, HttpReq& r, size_t max_body) {
    std::string buf; char tmp[16384];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) return 1;
        buf.append(tmp, (size_t)n);
        hdr_end = buf.find("\r\n\r\n");
        if (hdr_end == std::string::npos && buf.size() > 65536) return 1;
    }
    const std::string head = buf.substr(0, hdr_end);
    std::string rest = buf.substr(hdr_end + 4);
    size_t pos = head.find("\r\n");
    const std::string line = head.substr(0, pos);
    const size_t s1 = line.find(' '), s2 = line.rfind(' ');
    if (s1 == std::string::npos || s2 == s1) return 1;
    r.method = line.substr(0, s1); r.target = line.substr(s1 + 1, s2 - s1 - 1);
    const std::string ver = line.substr(s2 + 1);
    r.path = r.target.substr(0, r.target.find('?'));
    r.keep = ver != "HTTP/1.0";
    while (pos != std::string::npos && pos < head.size()) {
        const size_t e = head.find("\r\n", pos + 2);
        const std::string l = head.substr(pos + 2, e == std::string::npos ? std::string::npos : e - pos - 2);
        pos = e;
        const size_t c = l.find(':'); if (c == std::string::npos) continue;
        std::string k = l.substr(0, c), v = l.substr(c + 1);
        for (auto& ch : k) ch = (char)tolower((unsigned char)ch);
        while (!v.empty() && (v[0] == ' ' || v[0] == '\t')) v.erase(0, 1);
        r.h[k] = v;
    }
    auto it = r.h.find("connection");
    if (it != r.h.end()) { std::string v = it->second; for (auto& ch : v) ch = (char)tolower((unsigned char)ch); if (v == "close") r.keep = false; else if (v == "keep-alive") r.keep = true; }
    auto ex = r.h.find("expect");
    if (ex != r.h.end()) send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    auto te = r.h.find("transfer-encoding");
    if (te != r.h.end() && te->second.find("chunked") != std::string::npos) {
        std::string body, cur = std::move(rest);
        auto fill = [&]() { ssize_t n = recv(fd, tmp, sizeof tmp, 0); if (n <= 0) return false; cur.append(tmp, (size_t)n); return true; };
        while (true) {
            size_t le; while ((le = cur.find("\r\n")) == std::string::npos) if (!fill()) return 1;
            const size_t sz = strtoull(cur.substr(0, le).c_str(), nullptr, 16);
            cur.erase(0, le + 2);
            if (sz == 0) break;
            if (body.size() + sz > max_body) { respond(fd, 413, err_json("request body too large", "invalid_request_error"), false); return 2; }
            while (cur.size() < sz + 2) if (!fill()) return 1;
            body.append(cur, 0, sz); cur.erase(0, sz + 2);
        }
        r.body = std::move(body);
        return 0;
    }
    size_t want = 0;
    auto cl = r.h.find("content-length");
    if (cl != r.h.end()) want = (size_t)strtoull(cl->second.c_str(), nullptr, 10);
    if (want > max_body) { respond(fd, 413, err_json("request body too large (raise --max-body-mb)", "invalid_request_error"), false); return 2; }
    r.body = std::move(rest);
    r.body.reserve(want);
    while (r.body.size() < want) {
        ssize_t n = recv(fd, tmp, std::min(sizeof tmp, want - r.body.size()), 0);
        if (n <= 0) return 1;
        r.body.append(tmp, (size_t)n);
    }
    if (r.body.size() > want) r.body.resize(want);
    return 0;
}

// ---------------------------------------------------------------- endpoints
static std::string models_json(const Server& S) {
    std::string d;
    if (S.has_emb)
        d += "{\"id\":\"" + jesc(S.model_name) + "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"ferro\",\"capabilities\":[\"embeddings\"],\"modalities\":[\"text\"" +
             (S.has_mm ? ",\"image\",\"audio\",\"video\"" : "") + "],\"dimensions\":" + std::to_string(S.dim) + "}";
    if (S.has_lm)
        d += std::string(d.empty() ? "" : ",") + "{\"id\":\"" + jesc(S.lm_name) + "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"ferro\",\"capabilities\":[\"chat\",\"completions\"],\"modalities\":[\"text\"]," +
             "\"context_length\":" + std::to_string(S.lm.max_ctx()) + ",\"slots\":" + std::to_string(S.lm.n_slots()) + "}";
    if (S.has_d1)
        d += std::string(d.empty() ? "" : ",") + "{\"id\":\"" + jesc(S.d1_name) + "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"ferro\",\"capabilities\":[\"decision\"],\"endpoint\":\"/v1/systemone\",\"modalities\":[\"text\"" +
             (S.d1.has_vision() ? ",\"image\"" : "") + (S.d1.has_audio() ? ",\"audio\"" : "") + "]}";
    return "{\"object\":\"list\",\"data\":[" + d + "]}";
}
static std::string health_json(Server& S) {
    size_t qd; { std::lock_guard<std::mutex> g(S.qm); qd = S.q.size(); }
    const double up = ms_since(S.t_start) / 1000.0;
    char b[640];
    snprintf(b, sizeof b, "{\"status\":\"ok\",\"model\":\"%s\",\"multimodal\":%s,\"queue\":%zu,\"inflight\":%d,\"requests\":%llu,\"errors\":%llu,"
                          "\"embeddings\":%llu,\"tokens\":%llu,\"batches\":%llu,\"uptime_s\":%.1f,\"gpu_busy\":%.3f}",
             jesc(S.model_name).c_str(), S.has_mm ? "true" : "false", qd, S.inflight.load(), (unsigned long long)S.n_req.load(),
             (unsigned long long)S.n_err.load(), (unsigned long long)S.n_emb.load(), (unsigned long long)S.n_tok.load(),
             (unsigned long long)S.n_batches.load(), up, up > 0 ? S.busy_us.load() / 1e6 / up : 0.0);
    std::string out = b;
    if (S.has_lm) {
        size_t gq; { std::lock_guard<std::mutex> g(S.qm); gq = S.gq.size(); }
        char l[512];
        const double tg = S.gen_us.load() > 0 ? (double)S.n_gen_tok.load() / (S.gen_us.load() / 1e6) : 0;
        snprintf(l, sizeof l, ",\"lm\":{\"model\":\"%s\",\"device\":\"%s\",\"slots\":%d,\"active\":%d,\"queued\":%zu,\"ctx\":%d,\"requests\":%llu,\"prompt_tokens\":%llu,\"generated_tokens\":%llu,\"tokens_per_s_busy\":%.2f}}",
                 jesc(S.lm_name).c_str(), S.lm.on_gpu() ? "igpu" : "cpu", S.lm.n_slots(), S.n_active.load(), gq, S.lm.max_ctx(), (unsigned long long)S.n_gen_req.load(),
                 (unsigned long long)S.n_prompt_tok.load(), (unsigned long long)S.n_gen_tok.load(), tg);
        out.pop_back(); out += l;
    }
    if (S.has_d1) {
        size_t dq; { std::lock_guard<std::mutex> g(S.qm); dq = S.dq.size(); }
        char l[400];
        snprintf(l, sizeof l, ",\"d1\":{\"model\":\"%s\",\"device\":\"%s\",\"queued\":%zu,\"requests\":%llu,\"questions\":%llu,\"tokens\":%llu,\"passes\":%llu,\"busy_s\":%.2f,\"image\":%s,\"audio\":%s}}",
                 jesc(S.d1_name).c_str(), S.d1.on_gpu() ? "igpu" : "cpu", dq, (unsigned long long)S.n_d1_req.load(), (unsigned long long)S.n_d1_q.load(), (unsigned long long)S.n_d1_tok.load(),
                 (unsigned long long)S.n_d1_pass.load(), S.d1_us.load() / 1e6, S.d1.has_vision() ? "true" : "false", S.d1.has_audio() ? "true" : "false");
        out.pop_back(); out += l;
    }
    return out;
}

// ---------------------------------------------------------------- chat / completions
static double jnum(const Json* j, double def) { return !j ? def : j->t == Json::Int ? (double)j->i : j->t == Json::Float ? j->d : def; }
static bool client_gone(int fd) { char c; const ssize_t r = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT); return r == 0; }
static bool send_chunk(int fd, const std::string& data) {
    char head[32]; const int n = snprintf(head, sizeof head, "%zx\r\n", data.size());
    return send_all(fd, head, (size_t)n) && send_all(fd, data.data(), data.size()) && send_all(fd, "\r\n", 2);
}
static std::string dquote(const std::string& s) { return "\"" + jesc(s) + "\""; }

static std::shared_ptr<GenJob> parse_gen_request(Server& S, const std::string& text, bool chat) {
    Json b;
    try { b = json_parse(text); } catch (const std::exception& e) { throw BadRequest(std::string("invalid JSON: ") + e.what()); }
    if (!b.is_object()) throw BadRequest("request body must be a JSON object");
    auto j = std::make_shared<GenJob>();
    j->chat = chat;
    std::string rendered;
    if (chat) {
        const Json* m = b.find("messages");
        if (!m || !m->is_array() || m->arr.empty()) throw BadRequest("\"messages\" must be a non-empty array");
        try { rendered = lfm_chat_render(*m, b.find("tools"), S.chat_opt); } catch (const std::exception& e) { throw BadRequest(std::string("messages: ") + e.what()); }
    } else {
        const Json* p = b.find("prompt");
        if (p && p->is_array() && p->arr.size() == 1) p = &p->arr[0];
        if (!p || !p->is_string()) throw BadRequest("\"prompt\" must be a string");
        rendered = "<|startoftext|>" + p->s;
    }
    j->prompt = S.ltok.tokenize(rendered, true);
    if (j->prompt.empty()) throw BadRequest("empty prompt");
    const int ctx = S.lm.max_ctx();
    if ((int)j->prompt.size() >= ctx) throw BadRequest("prompt is " + std::to_string(j->prompt.size()) + " tokens but the context is " + std::to_string(ctx) + " (start with --lm-ctx N)");
    if (const Json* n = b.find("n")) if (jnum(n, 1) != 1) throw BadRequest("only n = 1 is supported");
    SamplerParams sp;                                    // defaults: LiquidAI's recommended generation settings
    sp.temperature = (float)jnum(b.find("temperature"), sp.temperature);
    sp.top_p = (float)jnum(b.find("top_p"), sp.top_p);
    sp.top_k = (int)jnum(b.find("top_k"), sp.top_k);
    sp.min_p = (float)jnum(b.find("min_p"), sp.min_p);
    sp.repetition_penalty = (float)jnum(b.find("repetition_penalty"), jnum(b.find("repeat_penalty"), sp.repetition_penalty));
    sp.seed = (uint64_t)jnum(b.find("seed"), 0);
    if (sp.temperature < 0 || sp.top_p <= 0 || sp.top_p > 1 || sp.top_k < 0 || sp.repetition_penalty <= 0) throw BadRequest("sampling parameter out of range");
    j->sp = sp;
    double mt = jnum(b.find("max_completion_tokens"), jnum(b.find("max_tokens"), jnum(b.find("n_predict"), -1)));
    const int room = ctx - (int)j->prompt.size() - 1;
    j->max_tokens = mt <= 0 ? room : std::min((int)mt, room);
    if (const Json* st = b.find("stop")) {
        if (st->is_string()) j->stops.push_back(st->s);
        else if (st->is_array()) for (const Json& x : st->arr) if (x.is_string()) j->stops.push_back(x.s);
        j->stops.erase(std::remove(j->stops.begin(), j->stops.end(), std::string()), j->stops.end());
        if (j->stops.size() > 8) throw BadRequest("at most 8 stop strings");
    }
    if (const Json* s = b.find("stream")) j->stream = s->t == Json::Bool && s->b;
    if (const Json* so = b.find("stream_options")) if (const Json* iu = sub(*so, "include_usage")) j->include_usage = iu->t == Json::Bool && iu->b;
    return j;
}

static std::string usage_json(const GenJob& j) {
    char b[512];
    const double pp = j.prompt_ms > 0 ? j.prompt.size() / (j.prompt_ms / 1000.0) : 0, tg = j.gen_ms - j.prompt_ms > 0 ? (j.n_gen > 1 ? (j.n_gen - 1) / ((j.gen_ms - j.prompt_ms) / 1000.0) : 0) : 0;
    snprintf(b, sizeof b, "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%d,\"total_tokens\":%zu},\"timings\":{\"prompt_ms\":%.1f,\"prompt_per_second\":%.1f,\"predicted_ms\":%.1f,\"predicted_per_second\":%.2f}",
             j.prompt.size(), j.n_gen, j.prompt.size() + (size_t)j.n_gen, j.prompt_ms, pp, j.gen_ms - j.prompt_ms, tg);
    return b;
}

// POST /v1/chat/completions and /v1/completions. Returns the HTTP status, or -1 when the response has already been written to `fd` (streaming).
static int handle_generate(Server& S, int fd, bool keep, const std::string& text, bool chat, std::string& body, std::string& logline) {
    if (!S.has_lm) { body = err_json("no language model loaded (start the server with --lm MODEL.gguf)", "invalid_request_error"); return 404; }
    std::shared_ptr<GenJob> j;
    try { j = parse_gen_request(S, text, chat); }
    catch (const BadRequest& e) { body = err_json(e.what(), "invalid_request_error"); return 400; }
    {
        std::lock_guard<std::mutex> lk(S.qm);
        if ((int)S.gq.size() + S.n_active.load() >= S.o.max_queue) { body = err_json("server is busy (queue full), retry later", "server_error"); return 503; }
        S.gq.push_back(j);
    }
    S.qcv.notify_all();
    static std::atomic<uint64_t> counter{0};
    const uint64_t id = ++counter; const long created = (long)time(nullptr);
    const std::string idstr = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(id), model = dquote(S.lm_name);
    auto wait = [&](std::deque<Delta>& out) -> bool {          // returns true once the job is done and `out` holds the last events
        std::unique_lock<std::mutex> lk(j->m);
        while (j->events.empty() && !j->done) {
            j->cv.wait_for(lk, std::chrono::milliseconds(250));
            if (j->events.empty() && !j->done && client_gone(fd)) { j->cancelled = true; }
        }
        out.swap(j->events);
        return j->done;
    };
    auto finish_str = [&] { return j->finish; };
    auto summary = [&] { char b[160]; snprintf(b, sizeof b, "prompt=%zu gen=%d finish=%s pp=%.0ftok/s", j->prompt.size(), j->n_gen, j->finish.c_str(), j->prompt_ms > 0 ? j->prompt.size() / (j->prompt_ms / 1000.0) : 0.0); return std::string(b); };
    std::deque<Delta> ev;
    if (!j->stream) {
        std::string content, reasoning;
        while (true) { const bool d = wait(ev); for (auto& e : ev) (e.reasoning ? reasoning : content) += e.text; ev.clear(); if (d) break; }
        if (!j->err.empty()) { body = err_json(j->err, "server_error"); return j->status; }
        if (chat) {
            body = "{\"id\":" + dquote(idstr) + ",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) + ",\"model\":" + model +
                   ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" + dquote(content) +
                   (reasoning.empty() ? "" : ",\"reasoning_content\":" + dquote(reasoning)) + "},\"finish_reason\":" + dquote(finish_str()) + "}]," + usage_json(*j) + "}";
        } else {
            body = "{\"id\":" + dquote(idstr) + ",\"object\":\"text_completion\",\"created\":" + std::to_string(created) + ",\"model\":" + model +
                   ",\"choices\":[{\"index\":0,\"text\":" + dquote(content) + (reasoning.empty() ? "" : ",\"reasoning_content\":" + dquote(reasoning)) +
                   ",\"finish_reason\":" + dquote(finish_str()) + "}]," + usage_json(*j) + "}";
        }
        logline = summary();
        return 200;
    }
    // ---- streaming (server-sent events over chunked transfer encoding)
    const std::string head = std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: ") + (keep ? "keep-alive" : "close") + "\r\n\r\n";
    bool alive = send_all(fd, head.data(), head.size());
    auto chunk_json = [&](const std::string& delta, const char* fin) {
        std::string c = "{\"id\":" + dquote(idstr) + ",\"object\":\"" + (chat ? "chat.completion.chunk" : "text_completion") + "\",\"created\":" + std::to_string(created) + ",\"model\":" + model + ",\"choices\":[{\"index\":0,";
        c += chat ? "\"delta\":" + delta : "\"text\":" + delta;
        c += std::string(",\"finish_reason\":") + (fin ? dquote(fin) : "null") + "}]";
        return c;
    };
    auto sse = [&](const std::string& json) { if (alive) alive = send_chunk(fd, "data: " + json + "\n\n"); };
    if (chat) sse(chunk_json("{\"role\":\"assistant\",\"content\":\"\"}", nullptr) + "}");
    bool done = false;
    while (!done) {
        done = wait(ev);
        for (auto& e : ev) {
            if (chat) sse(chunk_json(e.reasoning ? "{\"reasoning_content\":" + dquote(e.text) + "}" : "{\"content\":" + dquote(e.text) + "}", nullptr) + "}");
            else sse(chunk_json(dquote(e.text), nullptr) + "}");
        }
        ev.clear();
        if (!alive) { j->cancelled = true; break; }
    }
    if (alive && !j->err.empty()) sse(err_json(j->err, "server_error"));
    else if (alive) {
        sse(chunk_json(chat ? "{}" : "\"\"", finish_str().c_str()) + "}");
        if (j->include_usage) sse("{\"id\":" + dquote(idstr) + ",\"object\":\"" + (chat ? "chat.completion.chunk" : "text_completion") + "\",\"created\":" + std::to_string(created) + ",\"model\":" + model + ",\"choices\":[]," + usage_json(*j) + "}");
        if (alive) alive = send_chunk(fd, "data: [DONE]\n\n");
        if (alive) alive = send_all(fd, "0\r\n\r\n", 5);
    }
    logline = summary() + (alive ? "" : " (client disconnected)");
    return alive ? -1 : -2;           // -2: connection is unusable, close it
}

// POST /v1/embeddings. Returns the HTTP status; `body` is the response body.
static int handle_embeddings(Server& S, const std::string& text, std::string& body, std::string& logline) {
    if (!S.has_emb) { body = err_json("no embedding model loaded (start the server with --model EG2.gguf)", "invalid_request_error"); return 404; }
    struct Inflight { std::atomic<int>& c; Inflight(std::atomic<int>& c_) : c(c_) { c++; } ~Inflight() { c--; } } inflight_guard(S.inflight);
    auto job = std::make_shared<Job>();
    int dims = 0; bool b64 = false;
    try {
        Json req = json_parse(text);
        if (!req.is_object()) throw BadRequest("the request body must be a JSON object");
        const Json* in = req.find("input");
        if (!in) throw BadRequest("missing \"input\"");
        if (const Json* d = req.find("dimensions")) { if (d->t == Json::Int) dims = (int)d->i; else if (!d->is_null()) throw BadRequest("\"dimensions\" must be an integer"); }
        if (dims < 0 || dims > S.dim) throw BadRequest("\"dimensions\" must be between 1 and " + std::to_string(S.dim));
        if (const Json* f = req.find("encoding_format")) {
            if (f->is_string() && f->s == "base64") b64 = true;
            else if (!(f->is_null() || (f->is_string() && f->s == "float"))) throw BadRequest("encoding_format must be \"float\" or \"base64\"");
        }
        if (const Json* v = req.find("video_fps")) { job->video_fps = (float)(v->t == Json::Int ? (double)v->i : v->d); if (!(job->video_fps > 0 && job->video_fps <= 30)) throw BadRequest("video_fps must be in (0, 30]"); }
        else job->video_fps = S.o.video_fps;
        auto one = [&](const Json& e) {
            std::vector<EmbedItem> items;
            if (e.is_string()) { EmbedItem it; it.value = e.s; items.push_back(std::move(it)); }
            else if (e.is_array()) {
                if (!e.arr.empty() && e.arr[0].t == Json::Int) throw BadRequest("token-id inputs are not supported; send text");
                for (const Json& p : e.arr) items.push_back(parse_part(p, S.has_mm));
            } else if (e.is_object()) {
                if (const Json* c = e.find("content")) {
                    if (c->is_string()) { EmbedItem it; it.value = c->s; items.push_back(std::move(it)); }
                    else if (c->is_array()) for (const Json& p : c->arr) items.push_back(parse_part(p, S.has_mm));
                    else throw BadRequest("\"content\" must be a string or an array of parts");
                } else items.push_back(parse_part(e, S.has_mm));
            } else throw BadRequest("each input must be a string, an array of content parts or an object with \"content\"");
            if (items.empty()) throw BadRequest("empty input");
            bool txt = items.size() == 1 && items[0].kind == EmbedItem::TEXT;
            job->text_only = job->text_only && txt;
            job->inputs.push_back(std::move(items));
        };
        if (in->is_array()) { if (in->arr.empty()) throw BadRequest("\"input\" is an empty array"); for (const Json& e : in->arr) one(e); }
        else one(*in);
        if (job->inputs.size() > 256) throw BadRequest("at most 256 inputs per request");
    } catch (const BadRequest& e) { body = err_json(e.what(), "invalid_request_error"); logline = e.what(); return 400; }
    catch (const std::exception& e) { body = err_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error"); logline = "bad json"; return 400; }

    { std::lock_guard<std::mutex> g(S.qm);
      if ((int)S.q.size() >= S.o.max_queue) { body = err_json("server is busy (queue full), retry shortly", "server_error"); logline = "queue full"; return 503; }
      S.q.push_back(job); }
    S.qcv.notify_all();
    { std::unique_lock<std::mutex> lk(job->m); job->cv.wait(lk, [&] { return job->done; }); }
    if (!job->err.empty()) { body = err_json(job->err, job->status >= 500 ? "server_error" : "invalid_request_error"); logline = job->err; return job->status; }

    int total = 0; for (int n : job->n_tokens) total += n;
    S.n_tok += (uint64_t)total; S.n_emb += job->out.size();
    body.clear(); body.reserve(job->out.size() * (size_t)(S.dim * 13 + 64) + 256);
    body += "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < job->out.size(); ++i) {
        std::vector<float>& v = job->out[i];
        if (dims > 0 && dims < (int)v.size()) {                       // Matryoshka truncation + renormalise
            v.resize(dims); double ss = 0; for (float x : v) ss += (double)x * x;
            const float inv = ss > 0 ? (float)(1.0 / std::sqrt(ss)) : 0.f; for (float& x : v) x *= inv;
        }
        if (i) body += ',';
        body += "{\"object\":\"embedding\",\"index\":" + std::to_string(i) + ",\"embedding\":";
        if (b64) { body += '"'; body += b64_encode((const uint8_t*)v.data(), v.size() * 4); body += '"'; }
        else {
            body += '[';
            char b[32];
            for (size_t k = 0; k < v.size(); ++k) { snprintf(b, sizeof b, "%s%.9g", k ? "," : "", std::isfinite(v[k]) ? v[k] : 0.0f); body += b; }
            body += ']';
        }
        body += '}';
    }
    char tail[256];
    snprintf(tail, sizeof tail, "],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%d,\"total_tokens\":%d}}", jesc(S.model_name).c_str(), total, total);
    body += tail;
    char lg[160]; snprintf(lg, sizeof lg, "n=%zu tok=%d queue=%.1fms run=%.0fms batch=%d%s", job->out.size(), total, job->queue_ms, job->run_ms, job->batch_seqs, job->text_only ? "" : " mm");
    logline = lg;
    return 200;
}

static int handle_systemone(Server& S, const std::string& text, std::string& body, std::string& logline) {
    auto job = std::make_shared<DecJob>(); job->body = text;
    { std::lock_guard<std::mutex> g(S.qm);
      if ((int)S.dq.size() >= S.o.max_queue) { body = err_json("server is busy (queue full), retry shortly", "server_error"); logline = "queue full"; return 503; }
      S.dq.push_back(job); }
    S.qcv.notify_all();
    { std::unique_lock<std::mutex> lk(job->m); job->cv.wait(lk, [&] { return job->done; }); }
    if (!job->err.empty()) { body = err_json(job->err, job->status >= 500 ? "server_error" : "invalid_request_error"); logline = job->err; return job->status; }
    body = job->out;
    char lg[200]; snprintf(lg, sizeof lg, "q=%zu tok=%zu%s queue=%.1fms prep=%.0fms run=%.0fms", job->n_q, job->n_tok, job->n_img ? (" img=" + std::to_string(job->n_img)).c_str() : "", job->queue_ms, job->prep_ms, job->run_ms);
    logline = lg;
    return 200;
}

static void handle_conn(Server* Sp, int fd) {
    Server& S = *Sp;
    struct timeval tv; tv.tv_sec = 60; tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    while (!S.stop) {
        HttpReq r;
        const int rc = read_request(fd, r, S.o.max_body_mb << 20);
        if (rc != 0) break;
        const auto t0 = clk::now();
        int status = 200; std::string body, logline; const char* ctype = "application/json";
        S.n_req++;
        const bool authed = S.o.api_key.empty() || r.path == "/health" || r.path == "/" ||
                            (r.h.count("authorization") && r.h["authorization"] == "Bearer " + S.o.api_key);
        if (!authed) { status = 401; body = err_json("invalid or missing API key", "authentication_error"); }
        else if (r.path == "/health" || r.path == "/v1/health") body = health_json(S);
        else if (r.path == "/v1/models" || r.path == "/models") body = models_json(S);
        else if (r.path == "/") body = "{\"name\":\"ferro-embed\",\"endpoints\":[\"POST /v1/embeddings\",\"POST /v1/systemone\",\"POST /v1/chat/completions\",\"POST /v1/completions\",\"GET /v1/models\",\"GET /health\"]}";
        else if (r.path == "/apply-template" || r.path == "/v1/apply-template") {              // same contract as llama.cpp: {"messages": [...], "tools": [...]} -> {"prompt": "..."}
            try {
                if (r.method != "POST") throw BadRequest("use POST");
                Json b = json_parse(r.body); const Json* m = b.is_object() ? b.find("messages") : nullptr;
                if (!m || !m->is_array()) throw BadRequest("\"messages\" must be an array");
                body = "{\"prompt\":" + dquote(lfm_chat_render(*m, b.find("tools"), S.chat_opt)) + "}";
            } catch (const std::exception& e) { status = 400; body = err_json(e.what(), "invalid_request_error"); }
        }
        else if (r.path == "/v1/chat/completions" || r.path == "/chat/completions" || r.path == "/v1/completions" || r.path == "/completions") {
            if (r.method != "POST") { status = 405; body = err_json("use POST", "invalid_request_error"); }
            else {
                const bool chat = r.path.find("chat") != std::string::npos;
                status = handle_generate(S, fd, r.keep && !S.stop, r.body, chat, body, logline);
                if (status == -1 || status == -2) {                          // streamed: already answered
                    fprintf(stderr, "[serve] %s %s 200 %.0fms %s\n", r.method.c_str(), r.path.c_str(), ms_since(t0), logline.c_str());
                    if (status == -2 || !r.keep) break;
                    continue;
                }
            }
        }
        else if (r.path == "/v1/systemone" || r.path == "/systemone") {
            if (r.method != "POST") { status = 405; body = err_json("use POST", "invalid_request_error"); }
            else if (!S.has_d1) { status = 404; body = err_json("no decision model loaded (start the server with --d1 d1-omni.gguf)", "invalid_request_error"); }
            else status = handle_systemone(S, r.body, body, logline);
        }
        else if (r.path == "/v1/embeddings" || r.path == "/embeddings") {
            if (r.method != "POST") { status = 405; body = err_json("use POST", "invalid_request_error"); }
            else status = handle_embeddings(S, r.body, body, logline);
        } else { status = 404; body = err_json("unknown endpoint " + r.path, "invalid_request_error"); }
        if (status >= 400) S.n_err++;
        const bool ok = respond(fd, status, body, r.keep && !S.stop, ctype, status == 503 ? "Retry-After: 1\r\n" : "");
        if (r.path != "/health")
            fprintf(stderr, "[serve] %s %s %d %.0fms %s\n", r.method.c_str(), r.path.c_str(), status, ms_since(t0), logline.c_str());
        if (!ok || !r.keep) break;
    }
    close(fd);
    S.conns--;
}

}  // namespace

int run_server(const ServerOptions& opt) {
    Server& S = *new Server;                       // intentionally leaked: detached connection threads may outlive this function
    S.o = opt; S.has_mm = !opt.mmproj.empty(); S.has_emb = !opt.model.empty(); S.has_lm = !opt.lm.empty(); S.has_d1 = !opt.d1.empty(); g_srv = &S;
    std::promise<std::string> ready; auto ready_f = ready.get_future();
    fprintf(stderr, "[serve] loading%s%s%s%s ...\n", S.has_emb ? (" " + opt.model).c_str() : "", S.has_mm ? " + mmproj" : "", S.has_lm ? " and LM " : "", S.has_lm ? opt.lm.c_str() : "");
    if (S.has_d1) fprintf(stderr, "[serve] + decision model %s\n", opt.d1.c_str());
    const auto t0 = clk::now();
    std::thread worker([&] { worker_loop(S, ready); });
    const std::string err = ready_f.get();
    if (!err.empty()) { fprintf(stderr, "[serve] init failed: %s\n", err.c_str()); worker.join(); Pool::get().shutdown(); return 1; }
    if (S.has_emb) {
        S.dim = S.me.dim_out();
        S.model_name = opt.model_name;
        if (S.model_name.empty()) { GgufFile g; std::string e; S.model_name = (g.open_file(opt.model.c_str(), &e) && !g.get_str("general.name").empty()) ? g.get_str("general.name") : "embeddinggemma-2"; }
        fprintf(stderr, "[serve] embeddings \"%s\" ready (dim %d, %s)\n", S.model_name.c_str(), S.dim, S.has_mm ? "text+image+audio+video" : "text only");
    }
    if (S.has_lm) {
        S.lm_name = opt.lm_name;
        if (S.lm_name.empty()) { std::string f = opt.lm; const size_t sl = f.find_last_of('/'); if (sl != std::string::npos) f = f.substr(sl + 1); if (f.size() > 5 && f.compare(f.size() - 5, 5, ".gguf") == 0) f.resize(f.size() - 5); S.lm_name = f; }
        fprintf(stderr, "[serve] language model \"%s\" ready (%d slot(s) x %d ctx, %s, chat template: %s)\n", S.lm_name.c_str(), S.lm.n_slots(), S.lm.max_ctx(), S.lm.on_gpu() ? "iGPU" : "CPU",
                S.chat_opt.typed_content ? "typed-content variant" : "plain-content variant");
        {   // device buffers on an iGPU are system RAM and cannot be swapped out: say how much the KV cache takes and compare with any container limit
            const LmModel& lm = S.lm.model();
            const double kv_mb = (double)lm.n_attn_layers * S.lm.n_slots() * S.lm.max_ctx() * 2 * lm.n_kv * lm.head_dim * 4 / 1048576.0;
            double lim_mb = 0; if (FILE* f = fopen("/sys/fs/cgroup/memory.max", "r")) { unsigned long long v = 0; if (fscanf(f, "%llu", &v) == 1) lim_mb = v / 1048576.0; fclose(f); }
            fprintf(stderr, "[serve] KV cache %.0f MB (%d slot(s) x %d ctx)%s\n", kv_mb, S.lm.n_slots(), S.lm.max_ctx(), lim_mb > 0 ? (" | container memory limit " + std::to_string((long)lim_mb) + " MB").c_str() : "");
        }
    }
    if (S.has_d1) {
        S.d1_name = S.d1.name();
        fprintf(stderr, "[serve] decision model \"%s\" ready (%s, image %s, audio %s) - POST /v1/systemone\n", S.d1_name.c_str(), S.d1.on_gpu() ? "iGPU" : "CPU", S.d1.has_vision() ? "yes" : "no", S.d1.has_audio() ? "yes" : "no");
    }
    {   // device buffers on an iGPU are system RAM and cannot be swapped out: compare everything that is resident with any container limit.
        // Calibrated on measurements (anon+shmem): ~700 MB runtime + 1.3 x all loaded GGUFs (incl. projectors) + the LM's KV cache.
        double lim_mb = 0; if (FILE* f = fopen("/sys/fs/cgroup/memory.max", "r")) { unsigned long long v = 0; if (fscanf(f, "%llu", &v) == 1) lim_mb = v / 1048576.0; fclose(f); }
        auto mb = [](const std::string& path) { struct stat st; return !path.empty() && stat(path.c_str(), &st) == 0 ? st.st_size / 1048576.0 : 0.0; };
        double w_mb = mb(opt.model) + mb(opt.mmproj) + mb(opt.lm) + mb(opt.d1) + (S.has_d1 ? mb(S.d1.mmproj_path()) : 0.0);
        double kv_mb = 0; if (S.has_lm) { const LmModel& lm = S.lm.model(); kv_mb = (double)lm.n_attn_layers * S.lm.n_slots() * S.lm.max_ctx() * 2 * lm.n_kv * lm.head_dim * 4 / 1048576.0; }
        const double need_mb = 700 + 1.3 * w_mb + kv_mb;
        if (lim_mb > 0 && need_mb > lim_mb)
            fprintf(stderr, "[serve] WARNING: ~%.0f MB needed (weights %.0f MB + KV %.0f MB + runtime) vs a %.0f MB container limit -> drop a model, or lower --lm-slots / --lm-ctx\n", need_mb, w_mb, kv_mb, lim_mb);
        else if (lim_mb > 0 && need_mb > 0.8 * lim_mb)
            fprintf(stderr, "[serve] note: ~%.0f MB resident of a %.0f MB container limit - little headroom for request bursts (video / large images); consider fewer models or smaller --lm-ctx\n", need_mb, lim_mb);
    }
    fprintf(stderr, "[serve] ready in %.1fs\n", std::chrono::duration<double>(clk::now() - t0).count());

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int on = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t)opt.port);
    if (inet_pton(AF_INET, opt.host.c_str(), &a.sin_addr) != 1) { fprintf(stderr, "[serve] bad --host %s (IPv4 address expected)\n", opt.host.c_str()); S.stop = true; S.qcv.notify_all(); worker.join(); Pool::get().shutdown(); return 2; }
    if (bind(fd, (sockaddr*)&a, sizeof a) != 0 || listen(fd, 128) != 0) {
        fprintf(stderr, "[serve] cannot listen on %s:%d: %s\n", opt.host.c_str(), opt.port, strerror(errno));
        close(fd); S.stop = true; S.qcv.notify_all(); worker.join(); Pool::get().shutdown(); return 1;
    }
    S.listen_fd = fd;
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal); signal(SIGPIPE, SIG_IGN);
    fprintf(stderr, "[serve] listening on http://%s:%d  (%s%sGET /v1/models, GET /health)%s\n", opt.host.c_str(), opt.port, S.has_emb ? "POST /v1/embeddings, " : "", S.has_lm ? "POST /v1/chat/completions, POST /v1/completions, " : "", opt.api_key.empty() ? "" : "  [API key required]");
    while (!S.stop) {
        pollfd p{fd, POLLIN, 0};
        if (poll(&p, 1, 500) <= 0) continue;
        int c = accept(fd, nullptr, nullptr);
        if (c < 0) continue;
        if (S.conns.load() >= opt.max_connections) { respond(c, 503, err_json("too many connections", "server_error"), false); close(c); continue; }
        S.conns++;
        std::thread(handle_conn, &S, c).detach();
    }
    fprintf(stderr, "[serve] shutting down ...\n");
    close(fd); S.qcv.notify_all(); worker.join();
    for (int i = 0; i < 40 && S.conns.load() > 0; ++i) usleep(50000);      // let handlers drain
    Pool::get().shutdown();
    g_srv = nullptr;
    return 0;
}

}  // namespace ferro
