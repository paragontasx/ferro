// server.cpp - see server.h
//
// Threads: the main thread accepts connections; each connection gets a handler thread (HTTP/1.1 keep-alive); ONE inference worker
// owns the engine, the GPU and the thread pool (the device is a single in-order queue, so inference is serialised by design).
// Scheduling: the worker takes the oldest job; if it is text-only it also takes every following text-only job (and briefly waits
// for requests that are already in flight) so they share packed forward passes - that is where the iGPU throughput is
// (16 x 188 tokens: 850 tok/s; one short sequence at a time is several times slower per token). Image / audio / video jobs run
// alone, in arrival order.
#include "server.h"
#include "memplan.h"
#include "prefix_store.h"
#include "webui.h"
#include <malloc.h>
#include "gpu_cl.h"
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
    bool cache_prompt = true;                     // request field (llama.cpp name): false = neither use nor create prefix-cache entries
    int prefix_hint = 0;                          // tokens of the system message when it is an exact prefix of the prompt (a natural cache boundary)
    int cached = 0;                               // prompt tokens restored from the prefix cache instead of prefilled
    int pf_pos = 0; std::vector<int> pf_cuts; bool pc_on = false; double pf_ms = 0;   // resumable prefill: next prompt position, snapshot boundaries still to leave, compute time so far
    bool spec = true; int spec_fail = 0, spec_skip = 0; uint64_t spec_d = 0, spec_a = 0; double spec_acc = 0.55;     // speculative decoding: allowed, back-off state, drafted / accepted tokens
    double ttft_ms = -1;                          // time from enqueue to the first generated token
    int think_budget = -1, think_n = 0; bool in_think = false, forced_close = false, ignore_eos = false;   // reasoning-token cap and tracker (independent of reasoning_format)
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

// A cached prompt prefix: the device state of the LM after `toks` (K/V rows + conv state). Worker-thread only.
struct PrefixEntry { std::vector<int> toks; std::shared_ptr<LmPrefix> px; uint64_t used = 0; };

// ---------------------------------------------------------------- server state
// API keys: any number, each with a label (for the log); matched in constant time; the file form is re-read on SIGHUP.
struct ApiKeys {
    std::mutex m; std::vector<std::pair<std::string, std::string>> keys;               // label, key
    bool empty() { std::lock_guard<std::mutex> g(m); return keys.empty(); }
    bool match(const std::string& presented, std::string* label) {
        std::lock_guard<std::mutex> g(m); bool found = false;
        for (const auto& k : keys) {
            unsigned diff = (unsigned)(k.second.size() ^ presented.size());
            for (size_t i = 0; i < k.second.size(); ++i) diff |= (unsigned char)k.second[i] ^ (unsigned char)(i < presented.size() ? presented[i] : 0);
            if (diff == 0 && !found) { found = true; *label = k.first; }
        }
        return found;
    }
    void add_line(std::string line, int idx) {                                          // "label: key" or a bare key
        auto trim = [](std::string& t) { while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back(); size_t i = 0; while (i < t.size() && isspace((unsigned char)t[i])) ++i; t.erase(0, i); };
        trim(line); if (line.empty() || line[0] == '#') return;
        const size_t c = line.find(": "); std::string label = "key" + std::to_string(idx), key = line;
        if (c != std::string::npos) { label = line.substr(0, c); key = line.substr(c + 2); trim(label); trim(key); }
        if (!key.empty()) keys.push_back({label, key});
    }
};
// Rolling rate over the last `win` seconds of activity (tokens per second while something was running, not averaged with idle time).
struct RateWin {
    std::mutex m; std::deque<std::tuple<double, int, double>> ev;                     // (end time s, tokens, busy seconds)
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    void add(int n, double busy_ms) { std::lock_guard<std::mutex> g(m); const double t = now(); ev.emplace_back(t, n, busy_ms / 1000.0); while (!ev.empty() && t - std::get<0>(ev.front()) > 30) ev.pop_front(); }
    double rate(double win = 10) {                                                  // tokens per busy second over the last `win` s; 0 once idle for 3 s
        std::lock_guard<std::mutex> g(m); const double t = now(); double tok = 0, busy = 0;
        if (ev.empty() || t - std::get<0>(ev.back()) > 3.0) return 0.0;
        for (auto it = ev.rbegin(); it != ev.rend() && t - std::get<0>(*it) <= win; ++it) { tok += std::get<1>(*it); busy += std::get<2>(*it); }
        return busy > 0 ? tok / busy : 0.0;
    }
};

struct Server {
    ServerOptions o;
    ApiKeys keys; std::atomic<bool> keys_reload{false};
    RateWin rw_decode, rw_prefill;                 // live token rates (/health, /metrics)
    std::atomic<uint64_t> ttft_n{0}, ttft_us{0}, ttft_b[8]{};   // time-to-first-token histogram (buckets 0.1 .25 .5 1 2.5 5 10 s, then +Inf)
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
    std::deque<std::shared_ptr<GenJob>> prefilling;   // worker only: prompts being filled slice by slice (each holds a slot)
    std::atomic<int> n_active{0}, n_prefill{0};
    double pf_rate = 0;                           // measured prefill tokens/s (EMA), sizes the slices
    std::vector<float> lg;                        // worker only: logits [slots][vocab]
    // prompt-prefix cache (worker only, except the atomics)
    std::vector<PrefixEntry> pcache; size_t pc_bytes = 0; uint64_t pc_tick = 0;
    std::deque<std::vector<int>> recent;          // last prompts, to learn shared prefixes
    std::atomic<uint64_t> pc_hits{0}, pc_miss{0}, pc_saved{0}, pc_evict{0}, pc_entries{0}, pc_used{0};
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
    std::mutex mem_m; double mem_inflight = 0;     // MB reserved by admitted media requests that are not finished
    std::atomic<uint64_t> n_mem_rejected{0};
    std::atomic<uint64_t> spec_drafted{0}, spec_accepted{0}, spec_steps{0};
    std::vector<float> lg_spec;                    // worker only: logits of every row of a verification window
    std::unique_ptr<PrefixStore> pstore;           // on-disk prefix snapshots (null = off)
    std::atomic<uint64_t> ps_hits{0};              // requests that restored a snapshot from disk
    std::atomic<bool> trim_pending{false};         // a media request ran: return freed heap to the OS when the worker is idle
    clk::time_point t_start = clk::now();
};
static Server* g_srv = nullptr;
static void on_hup(int) { if (g_srv) g_srv->keys_reload = true; }
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
    if (id == S.eos_id && !j.ignore_eos) { gen_drain(j, true); j.finish = "stop"; return true; }
    if (j.in_think && id != S.think_close && j.think_budget >= 0 && j.think_n >= j.think_budget && S.think_close >= 0) { id = S.think_close; j.forced_close = true; }   // budget spent: close the block, the answer comes next
    if (id == S.think_open) { j.in_think = true; j.think_n = 0; } else if (id == S.think_close) j.in_think = false; else if (j.in_think) j.think_n++;
    j.hist.push_back(id); j.n_gen++; j.next = id;
    if (j.n_gen == 1) j.ttft_ms = ms_since(j.t_enq);
    if (id == S.eos_id) { /* ignore_eos: keep generating, show nothing for the end token */ }
    else if (split && id == S.think_open) { gen_drain(j, true); j.mode = 1; }
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
    if (!err && j.finish == "length" && j.in_think) fprintf(stderr, "[serve] warning: reply reached max_tokens inside the reasoning block (content is empty); raise max_tokens or set thinking_budget\n");
    j.gen_ms = ms_since(j.t_gen0);
    if (j.ttft_ms >= 0) { static const double ub[7] = {100, 250, 500, 1000, 2500, 5000, 10000}; int b = 0; while (b < 7 && j.ttft_ms > ub[b]) ++b; S.ttft_b[b]++; S.ttft_n++; S.ttft_us += (uint64_t)(j.ttft_ms * 1000); }
    S.n_gen_req++; S.n_gen_tok += (uint64_t)j.n_gen; S.n_prompt_tok += j.prompt.size(); S.gen_us += (uint64_t)(j.gen_ms * 1000); S.pp_us += (uint64_t)(j.prompt_ms * 1000);
    { std::lock_guard<std::mutex> g(j.m); if (err) { j.err = err; j.status = status; } j.done = true; }
    j.cv.notify_all();
}
static bool pc_starts(const std::vector<int>& full, const std::vector<int>& pre) { return pre.size() <= full.size() && std::equal(pre.begin(), pre.end(), full.begin()); }
static PrefixEntry* pc_longest(Server& S, const std::vector<int>& prompt) {          // longest cached prefix that leaves >= 1 token to prefill (we need its logits)
    PrefixEntry* best = nullptr;
    for (auto& e : S.pcache) if (e.toks.size() < prompt.size() && (!best || e.toks.size() > best->toks.size()) && pc_starts(prompt, e.toks)) best = &e;
    return best;
}
static bool pc_has(Server& S, const std::vector<int>& prompt, int c) {
    for (auto& e : S.pcache) if ((int)e.toks.size() == c && pc_starts(prompt, e.toks)) return true;
    return false;
}
static void pc_store(Server& S, const std::vector<int>& prompt, std::shared_ptr<LmPrefix> px, bool persist = true) {
    const size_t budget = (size_t)S.o.prefix_cache_mb << 20;
    if (!px || px->bytes > budget) return;
    while (!S.pcache.empty() && S.pc_bytes + px->bytes > budget) {                 // evict least recently used
        size_t k = 0; for (size_t i = 1; i < S.pcache.size(); ++i) if (S.pcache[i].used < S.pcache[k].used) k = i;
        S.pc_bytes -= S.pcache[k].px->bytes; S.pcache.erase(S.pcache.begin() + (long)k); S.pc_evict++;
    }
    PrefixEntry e; e.toks.assign(prompt.begin(), prompt.begin() + px->n); e.used = ++S.pc_tick; S.pc_bytes += px->bytes; e.px = std::move(px);
    S.pcache.push_back(std::move(e));
    S.pc_entries = S.pcache.size(); S.pc_used = S.pc_bytes;
    if (persist && S.pstore && !S.pstore->has(S.pcache.back().toks)) {             // a new snapshot also goes to disk (written in the background)
        std::vector<uint8_t> blob;
        if (S.lm.export_prefix(*S.pcache.back().px, &blob)) S.pstore->save_async(S.pcache.back().toks, std::move(blob));
    }
}

static void gpu_sync_for_debug() { gpu::finish(); }
// Start a request: take a slot, restore the longest cached prefix, plan where to leave snapshots. The prompt is then filled slice by
// slice (gen_prefill_chunk) so decode streams, d1 requests and embeddings are served in between instead of waiting for the whole prompt.
static void gen_begin(Server& S, const std::shared_ptr<GenJob>& jp) {
    GenJob& j = *jp;
    j.t_gen0 = clk::now();
    S.lm.reset(j.seq);
    const int n = (int)j.prompt.size();
    int pos = 0;
    j.pc_on = S.o.prefix_cache_mb > 0 && j.cache_prompt && S.lm.on_gpu();
    std::vector<int> cuts;                                                          // where to leave a new cache entry behind while prefilling
    if (j.pc_on) {
        PrefixEntry* e = pc_longest(S, j.prompt);
        PrefixStore::Entry de;
        if (S.pstore && S.pstore->longest(j.prompt, &de) && (!e || de.toks.size() > e->toks.size() + 32)) {   // the disk has a longer prefix than the device cache: bring it in
            std::vector<uint8_t> blob;
            if (S.pstore->load(de, &blob)) {
                if (auto px = S.lm.import_prefix((int)de.toks.size(), blob.data(), blob.size())) {
                    pc_store(S, j.prompt, std::move(px), false); S.pstore->touch(de); S.ps_hits++;
                    e = pc_longest(S, j.prompt);
                }
            }
        }
        if (e) {
            if (S.lm.restore(j.seq, *e->px)) { pos = e->px->n; e->used = ++S.pc_tick; S.pc_hits++; S.pc_saved += (uint64_t)pos; }
            else S.lm.reset(j.seq);                                                // a failed restore may have left partial conv state: start clean
        } else S.pc_miss++;
        j.cached = pos;
        int learned = 0;                                                            // longest common prefix with a recent prompt = a prefix this workload repeats
        for (const auto& r : S.recent) { int l = 0; const int m = (int)std::min(r.size(), j.prompt.size()); while (l < m && r[l] == j.prompt[l]) ++l; learned = std::max(learned, l); }
        for (int c : {j.prefix_hint, learned}) {
            if (c < S.o.prefix_min_tokens || c <= pos + 8 || c > n - 1) continue;
            if (std::find(cuts.begin(), cuts.end(), c) != cuts.end() || pc_has(S, j.prompt, c)) continue;
            if (S.lm.prefix_bytes(c) > ((size_t)S.o.prefix_cache_mb << 20)) continue;
            cuts.push_back(c);
        }
        std::sort(cuts.begin(), cuts.end());
    }
    j.pf_pos = pos; j.pf_cuts = cuts; j.pf_ms = 0;
    S.prefilling.push_back(jp); S.n_prefill = (int)S.prefilling.size();
}
static bool other_work_waiting(Server& S) {
    std::lock_guard<std::mutex> lk(S.qm);
    return !S.q.empty() || !S.dq.empty() || !S.gq.empty();
}
// Advance the oldest prefilling prompt by one slice; when it is complete, sample its first token and let it join the decode batch.
static void gen_prefill_chunk(Server& S) {
    const int V = S.lm.n_vocab();
    std::shared_ptr<GenJob> jp = S.prefilling.front(); S.prefilling.pop_front();
    GenJob& j = *jp;
    auto upd = [&] { S.n_prefill = (int)S.prefilling.size(); };
    if (j.cancelled || S.stop) { gen_finish(S, jp); upd(); return; }
    const int n = (int)j.prompt.size();
    // slice size: everything at once when nothing else is waiting (best throughput), else what fits the time slice at the measured rate
    int end = n;
    if (!S.active.empty() || other_work_waiting(S)) {
        const double rate = S.pf_rate > 0 ? S.pf_rate : 300.0;
        end = std::min(n, j.pf_pos + std::max(64, std::min(1024, (int)(rate * S.o.prefill_slice_ms / 1000.0))));
    }
    for (int c : j.pf_cuts) if (c > j.pf_pos && c < end) end = c;                    // never step over a snapshot boundary
    std::string err; bool ok = true;
    const auto t0 = clk::now();
    try { ok = S.lm.step({&j.seq}, {j.prompt.data() + j.pf_pos}, {end - j.pf_pos}, S.lg.data(), &err, true); } catch (const std::exception& e) { err = e.what(); ok = false; }
    const double ms = ms_since(t0);
    if (!ok) { gen_finish(S, jp, (err.empty() ? "prefill failed" : err).c_str()); upd(); return; }
    if (end - j.pf_pos >= 48 && ms > 0) { const double r = (end - j.pf_pos) * 1000.0 / ms; S.pf_rate = S.pf_rate > 0 ? 0.6 * S.pf_rate + 0.4 * r : r; }
    S.rw_prefill.add(end - j.pf_pos, ms);
    j.pf_ms += ms; j.pf_pos = end;
    if (end < n) {
        if (std::find(j.pf_cuts.begin(), j.pf_cuts.end(), end) != j.pf_cuts.end()) {   // reached a boundary: leave a snapshot behind
            const auto ts = clk::now();
            pc_store(S, j.prompt, S.lm.snapshot(j.seq));
            if (getenv("FERRO_PC_DEBUG")) { gpu_sync_for_debug(); fprintf(stderr, "[prefix] snapshot of %d tokens took %.1f ms (alloc+copy, synced)\n", end, ms_since(ts)); }
        }
        S.prefilling.push_back(jp); upd(); return;                                  // round-robin with other prefilling prompts
    }
    upd();
    j.prompt_ms = j.pf_ms;
    if (j.pc_on) { S.recent.push_back(j.prompt); if (S.recent.size() > 8) S.recent.pop_front(); }
    j.hist = j.prompt;
    j.sampler.reset(new Sampler(j.sp));
    const int tok = j.sampler->sample(S.lg.data(), V, j.hist);
    if (gen_token(S, j, tok)) { gen_finish(S, jp); return; }
    S.active.push_back(jp); S.n_active = (int)S.active.size();
}
// Prompt lookup: the tokens that followed the most recent earlier occurrence of the last n tokens (n = 4, 3, 2) - replies often copy the prompt
// (ids, names, quoted text) or repeat themselves, and a copied run can be verified k tokens per forward instead of one.
static std::vector<int> lookup_draft(const std::vector<int>& h, int kmax, int eos, int min_n) {
    const int L = (int)h.size();
    for (int n = std::min(4, L - 1); n >= min_n; --n) {
        for (int p = L - n - 1; p >= 0; --p) {
            bool eq = true; for (int k = 0; k < n; ++k) if (h[p + k] != h[L - n + k]) { eq = false; break; }
            if (!eq) continue;
            std::vector<int> d;
            for (int k = p + n; k < L && (int)d.size() < kmax; ++k) { if (h[k] == eos) break; d.push_back(h[k]); }
            if (!d.empty()) return d;
        }
    }
    return {};
}
// One batched decode step for every active sequence. When some stream can draft (prompt lookup), the step is a verification window instead: each
// stream's rows [last token, drafts...] run with exactly the arithmetic of serial decode, tokens are sampled row by row (one sampler draw per produced
// token, as serially) and a draft is kept only while it equals the sampled token - so the output is identical to decoding one token at a time.
static void gen_decode_step(Server& S) {
    const int V = S.lm.n_vocab();
    // drop cancelled / shutting-down sequences
    for (size_t i = 0; i < S.active.size();) {
        if (S.active[i]->cancelled || S.stop) { gen_finish(S, S.active[i]); S.active.erase(S.active.begin() + (long)i); } else ++i;
    }
    S.n_active = (int)S.active.size();
    if (S.active.empty()) return;
    std::vector<std::vector<int>> drafts(S.active.size());
    bool any = false;
    if (S.o.spec_draft > 0 && S.active.size() < 8 && S.lm.spec_ok()) {
        int budget = 8 - (int)S.active.size();
        for (size_t i = 0; i < S.active.size() && budget > 0; ++i) {
            GenJob& j = *S.active[i];
            if (!j.spec) continue;
            if (j.spec_skip > 0) { --j.spec_skip; continue; }                       // recent drafts failed: sit out a few steps
            const int kj = j.spec_acc > 0.85 ? S.o.spec_draft : std::min(3, S.o.spec_draft);      // long windows only pay when nearly every draft is accepted
            const int room = std::min({kj, budget, j.max_tokens - j.n_gen - 1, S.lm.max_ctx() - j.seq.n_past - 2});
            if (room <= 0) continue;
            drafts[i] = lookup_draft(j.hist, room, S.eos_id, j.spec_acc >= 0.6 ? 2 : 3);       // a stream that has not shown drafting works needs a 3-token match
            if (!drafts[i].empty()) { budget -= (int)drafts[i].size(); any = true; }
        }
    }
    std::string err; bool ok = false;
    const auto tstep = clk::now();
    int produced = 0;
    if (!any) {
        std::vector<LmSeq*> seqs; std::vector<const int*> toks; std::vector<int> ns;
        for (auto& j : S.active) { seqs.push_back(&j->seq); toks.push_back(&j->next); ns.push_back(1); }
        try { ok = S.lm.step(seqs, toks, ns, S.lg.data(), &err); } catch (const std::exception& e) { err = e.what(); }
        if (!ok) {
            for (auto& j : S.active) gen_finish(S, j, (err.empty() ? "decode failed" : err).c_str());
            S.active.clear(); S.n_active = 0; return;
        }
        std::vector<std::shared_ptr<GenJob>> keep;
        for (size_t i = 0; i < S.active.size(); ++i) {
            auto& j = S.active[i];
            const int tok = j->sampler->sample(S.lg.data() + i * (size_t)V, V, j->hist);
            ++produced;
            if (gen_token(S, *j, tok)) gen_finish(S, j); else keep.push_back(j);
        }
        S.rw_decode.add(produced, ms_since(tstep));
        S.active.swap(keep); S.n_active = (int)S.active.size();
        return;
    }
    // ---- verification window
    std::vector<LmSeq*> seqs; std::vector<std::vector<int>> rows; std::vector<int> off;
    int total = 0;
    for (size_t i = 0; i < S.active.size(); ++i) {
        seqs.push_back(&S.active[i]->seq); rows.push_back({S.active[i]->next}); rows.back().insert(rows.back().end(), drafts[i].begin(), drafts[i].end());
        off.push_back(total); total += (int)rows.back().size();
    }
    S.lg_spec.resize((size_t)total * V);
    try { ok = S.lm.verify(seqs, rows, S.lg_spec.data(), &err); } catch (const std::exception& e) { err = e.what(); }
    if (!ok) {
        for (auto& j : S.active) gen_finish(S, j, (err.empty() ? "decode failed" : err).c_str());
        S.active.clear(); S.n_active = 0; return;
    }
    std::vector<std::shared_ptr<GenJob>> keep; std::vector<LmSeq*> cseqs; std::vector<int> coff, cacc;
    for (size_t i = 0; i < S.active.size(); ++i) {
        auto& j = S.active[i];
        const int m = (int)rows[i].size(); int acc = 0, confirmed = 0; bool fin = false;
        for (int r = 0; r < m; ++r) {
            const int tok = j->sampler->sample(S.lg_spec.data() + (size_t)(off[i] + r) * V, V, j->hist);
            fin = gen_token(S, *j, tok); acc = r + 1; ++produced;
            if (fin) break;
            if (r + 1 < m && j->next == rows[i][r + 1]) { ++confirmed; continue; }   // the draft was what the model would have said: its row is valid too
            break;
        }
        const int k = m - 1;
        if (k > 0) {
            j->spec_d += (uint64_t)k; j->spec_a += (uint64_t)confirmed; S.spec_drafted += (uint64_t)k; S.spec_accepted += (uint64_t)confirmed;
            j->spec_acc = 0.6 * j->spec_acc + 0.4 * ((double)confirmed / (double)k);
            if (confirmed == 0) { j->spec_fail++; j->spec_skip = std::min(1 << std::min(j->spec_fail, 4), 16) - 1; } else j->spec_fail = 0;
        }
        if (fin) { gen_finish(S, j); continue; }
        keep.push_back(j); cseqs.push_back(&j->seq); coff.push_back(off[i]); cacc.push_back(acc);
    }
    if (!cseqs.empty() && !S.lm.commit(cseqs, coff, cacc)) {
        for (auto& j : keep) gen_finish(S, j, "decode failed (commit)");
        keep.clear();
    }
    S.spec_steps++;
    S.rw_decode.add(produced, ms_since(tstep));
    S.active.swap(keep); S.n_active = (int)S.active.size();
}
// One scheduler iteration for generation: admit waiting requests into free slots, fill ONE slice of one prompt, then run `decode_share`
// batched decode steps. Between iterations the worker loop serves d1 and embedding work, so no job waits for another's whole prefill.
static void gen_iteration(Server& S) {
    while (true) {
        std::shared_ptr<GenJob> j;
        {
            std::lock_guard<std::mutex> lk(S.qm);
            if (S.gq.empty() || (int)(S.active.size() + S.prefilling.size()) >= S.lm.n_slots()) break;
            j = S.gq.front(); S.gq.pop_front();
        }
        if (j->cancelled) { gen_finish(S, j); continue; }
        gen_begin(S, j);
        if (S.stop) break;
    }
    for (size_t i = 0; i < S.prefilling.size();) {                                    // cancelled prompts give their slot back at once
        if (S.prefilling[i]->cancelled) { gen_finish(S, S.prefilling[i]); S.prefilling.erase(S.prefilling.begin() + (long)i); } else ++i;
    }
    S.n_prefill = (int)S.prefilling.size();
    if (!S.prefilling.empty()) gen_prefill_chunk(S);
    if (S.prefilling.empty()) { gen_decode_step(S); return; }
    // a prompt is still filling: give the decode streams their share of the time (at least one step) before the next slice
    const double budget = S.o.prefill_slice_ms * (double)S.o.decode_share / (100.0 - S.o.decode_share);
    const auto td = clk::now();
    do gen_decode_step(S); while (!S.active.empty() && ms_since(td) < budget);
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
            if (!S.lm.init(S.o.lm, S.o.threads, S.o.lm_ctx, &err, S.o.lm_slots, S.o.kv_int8)) { ready.set_value(err.empty() ? "language model init failed" : "language model: " + err); return; }
            if (!S.ltok.load(S.lm.model().gguf, &err)) { ready.set_value("language model tokenizer: " + err); return; }
            S.chat_opt.typed_content = lfm_template_is_typed(S.lm.model().gguf.get_str("tokenizer.chat_template"));
            S.think_open = S.ltok.token_id("<think>"); S.think_close = S.ltok.token_id("</think>"); S.eos_id = S.lm.model().eos_id;
            S.lg.assign((size_t)S.lm.n_slots() * (size_t)S.lm.n_vocab(), 0.0f);
            if (!S.o.snapshot_dir.empty() && S.o.prefix_cache_mb > 0 && S.lm.on_gpu()) {
                S.pstore.reset(new PrefixStore); std::string e;
                if (!S.pstore->open(S.o.snapshot_dir, S.lm.state_id(), (uint64_t)S.o.snapshot_mb << 20, &e)) { fprintf(stderr, "[serve] prefix snapshots disabled: %s\n", e.c_str()); S.pstore.reset(); }
                else fprintf(stderr, "[serve] prefix snapshots: %zu stored for this model (%.1f MB) in %s, budget %d MB\n", S.pstore->entries(), S.pstore->bytes() / 1048576.0, S.o.snapshot_dir.c_str(), S.o.snapshot_mb);
            }
        }
        if (S.has_d1 && !S.d1.init(S.o.d1, S.o.d1_mmproj, S.o.threads, &err, true)) { ready.set_value(err.empty() ? "decision model init failed" : "decision model: " + err); return; }
        ready.set_value(std::string());
    }
    MmEmbedOptions mo;
    while (true) {
        std::vector<std::shared_ptr<Job>> batch;
        {
            std::unique_lock<std::mutex> lk(S.qm);
            S.qcv.wait(lk, [&] { return S.stop || !S.q.empty() || !S.gq.empty() || !S.dq.empty() || S.n_active.load() > 0 || S.n_prefill.load() > 0; });
            if (S.stop && S.q.empty() && S.gq.empty() && S.dq.empty() && S.n_active.load() == 0 && S.n_prefill.load() == 0) break;
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
        if (S.trim_pending.load() && S.n_active.load() == 0 && S.n_prefill.load() == 0) {      // a media request ran: give its freed heap back while idle
            std::lock_guard<std::mutex> lk(S.qm);
            if (S.q.empty() && S.dq.empty() && S.gq.empty()) { S.trim_pending = false; malloc_trim(0); }
        }
    }
    if (S.pstore) S.pstore->flush();               // finish queued snapshot writes before exiting
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
        char l[1100];
        const double tg = S.gen_us.load() > 0 ? (double)S.n_gen_tok.load() / (S.gen_us.load() / 1e6) : 0;
        snprintf(l, sizeof l, ",\"lm\":{\"model\":\"%s\",\"device\":\"%s\",\"slots\":%d,\"active\":%d,\"prefilling\":%d,\"queued\":%zu,\"ctx\":%d,\"requests\":%llu,\"prompt_tokens\":%llu,\"generated_tokens\":%llu,\"tokens_per_s_busy\":%.2f,\"prefix_cache\":{\"entries\":%llu,\"mb\":%.1f,\"hits\":%llu,\"misses\":%llu,\"cached_tokens\":%llu,\"evictions\":%llu}}}",
                 jesc(S.lm_name).c_str(), S.lm.on_gpu() ? "igpu" : "cpu", S.lm.n_slots(), S.n_active.load(), S.n_prefill.load(), gq, S.lm.max_ctx(), (unsigned long long)S.n_gen_req.load(),
                 (unsigned long long)S.n_prompt_tok.load(), (unsigned long long)S.n_gen_tok.load(), tg,
                 (unsigned long long)S.pc_entries.load(), S.pc_used.load() / 1048576.0, (unsigned long long)S.pc_hits.load(), (unsigned long long)S.pc_miss.load(), (unsigned long long)S.pc_saved.load(), (unsigned long long)S.pc_evict.load());
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
    {   // container memory: what is resident now vs the limit, and what admitted media requests have reserved
        size_t gqn = 0; { std::lock_guard<std::mutex> g(S.qm); gqn = S.gq.size() + S.q.size() + S.dq.size(); }
        char lv[420]; snprintf(lv, sizeof lv, ",\"live\":{\"decode_tokens_per_second\":%.2f,\"prefill_tokens_per_second\":%.2f,\"connections\":%d,\"waiting\":%zu,\"ttft_ms_mean\":%.1f},\"speculative\":{\"drafted\":%llu,\"accepted\":%llu,\"steps\":%llu}",
                               S.rw_decode.rate(), S.rw_prefill.rate(), S.conns.load(), gqn, S.ttft_n.load() ? S.ttft_us.load() / 1000.0 / (double)S.ttft_n.load() : 0.0,
                               (unsigned long long)S.spec_drafted.load(), (unsigned long long)S.spec_accepted.load(), (unsigned long long)S.spec_steps.load());
        out.pop_back(); out += lv; out += '}';
        double inflight; { std::lock_guard<std::mutex> g(S.mem_m); inflight = S.mem_inflight; }
        const double lim = cgroup_limit_mb(), used = cgroup_used_mb();
        char l[300];
        snprintf(l, sizeof l, ",\"memory\":{\"limit_mb\":%.0f,\"used_mb\":%.0f,\"headroom_mb\":%.0f,\"reserved_mb\":%.0f,\"rejected\":%llu}", lim, used, lim > 0 ? lim - used : 0.0, inflight, (unsigned long long)S.n_mem_rejected.load());
        out.pop_back(); out += l;
        if (S.pstore) { char sn[300]; snprintf(sn, sizeof sn, ",\"prefix_snapshots\":{\"entries\":%zu,\"mb\":%.1f,\"saved\":%llu,\"disk_hits\":%llu}", S.pstore->entries(), S.pstore->bytes() / 1048576.0, (unsigned long long)S.pstore->saved(), (unsigned long long)S.ps_hits.load()); out += sn; }
        out += '}';
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
    if (const Json* cp = b.find("cache_prompt")) j->cache_prompt = !(cp->t == Json::Bool && !cp->b);
    j->think_budget = (int)jnum(b.find("thinking_budget"), jnum(b.find("reasoning_budget"), S.o.thinking_budget));
    if (const Json* ie = b.find("ignore_eos")) j->ignore_eos = ie->t == Json::Bool && ie->b;
    if (const Json* dr = b.find("draft")) j->spec = !(dr->t == Json::Bool && !dr->b);
    if (chat && j->cache_prompt && S.o.prefix_cache_mb > 0) {      // natural cache boundary: the end of a leading system message (checked, not assumed, to be an exact prefix)
        const Json& m0 = b.find("messages")->arr[0];
        const Json* role = m0.is_object() ? m0.find("role") : nullptr;
        if (role && role->is_string() && role->s == "system") {
            try {
                Json one = Json::array(); one.push(m0);
                ChatTemplateOptions o = S.chat_opt; o.add_generation_prompt = false;
                const std::vector<int> pre = S.ltok.tokenize(lfm_chat_render(one, b.find("tools"), o), true);
                if (!pre.empty() && pre.size() < j->prompt.size() && std::equal(pre.begin(), pre.end(), j->prompt.begin())) j->prefix_hint = (int)pre.size();
            } catch (const std::exception&) {}
        }
    }
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

// A reply that stops at max_tokens while still inside <think> has empty content: say so (and say how many reasoning tokens a forced close cut).
static std::string warn_json(const GenJob& j) {
    std::string w;
    if (j.finish == "length" && j.in_think) w += "\"reply reached max_tokens inside the reasoning block, so content is empty: raise max_tokens or set thinking_budget\"";
    if (j.forced_close) w += std::string(w.empty() ? "" : ",") + "\"reasoning was closed at thinking_budget=" + std::to_string(j.think_budget) + " tokens\"";
    return w.empty() ? std::string() : ",\"warnings\":[" + w + "]";
}
static std::string usage_json(const GenJob& j) {
    char b[640];
    const size_t fresh = j.prompt.size() - (size_t)j.cached;                 // tokens actually prefilled (the rest came from the prefix cache)
    const double pp = j.prompt_ms > 0 ? fresh / (j.prompt_ms / 1000.0) : 0, tg = j.gen_ms - j.prompt_ms > 0 ? (j.n_gen > 1 ? (j.n_gen - 1) / ((j.gen_ms - j.prompt_ms) / 1000.0) : 0) : 0;
    snprintf(b, sizeof b, "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%d,\"total_tokens\":%zu,\"prompt_tokens_details\":{\"cached_tokens\":%d}},\"timings\":{\"cache_n\":%d,\"prompt_ms\":%.1f,\"prompt_per_second\":%.1f,\"predicted_ms\":%.1f,\"predicted_per_second\":%.2f}",
             j.prompt.size(), j.n_gen, j.prompt.size() + (size_t)j.n_gen, j.cached, j.cached, j.prompt_ms, pp, j.gen_ms - j.prompt_ms, tg);
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
        if ((int)S.gq.size() + S.n_active.load() + S.n_prefill.load() >= S.o.max_queue) { body = err_json("server is busy (queue full), retry later", "server_error"); return 503; }
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
    auto summary = [&] {                                  // pp = prefill speed over the tokens actually prefilled; cached = restored from the prefix cache
        char b[200]; const size_t fresh = j->prompt.size() - (size_t)j->cached;
        snprintf(b, sizeof b, "prompt=%zu%s gen=%d%s finish=%s pp=%.0ftok/s", j->prompt.size(), j->cached ? (" cached=" + std::to_string(j->cached)).c_str() : "", j->n_gen, j->spec_d ? (" spec=" + std::to_string(j->spec_a) + "/" + std::to_string(j->spec_d)).c_str() : "", j->finish.c_str(), j->prompt_ms > 0 ? fresh / (j->prompt_ms / 1000.0) : 0.0);
        return std::string(b); };
    std::deque<Delta> ev;
    if (!j->stream) {
        std::string content, reasoning;
        while (true) { const bool d = wait(ev); for (auto& e : ev) (e.reasoning ? reasoning : content) += e.text; ev.clear(); if (d) break; }
        if (!j->err.empty()) { body = err_json(j->err, "server_error"); return j->status; }
        if (chat) {
            body = "{\"id\":" + dquote(idstr) + ",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) + ",\"model\":" + model +
                   ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" + dquote(content) +
                   (reasoning.empty() ? "" : ",\"reasoning_content\":" + dquote(reasoning)) + "},\"finish_reason\":" + dquote(finish_str()) + "}]," + usage_json(*j) + warn_json(*j) + "}";
        } else {
            body = "{\"id\":" + dquote(idstr) + ",\"object\":\"text_completion\",\"created\":" + std::to_string(created) + ",\"model\":" + model +
                   ",\"choices\":[{\"index\":0,\"text\":" + dquote(content) + (reasoning.empty() ? "" : ",\"reasoning_content\":" + dquote(reasoning)) +
                   ",\"finish_reason\":" + dquote(finish_str()) + "}]," + usage_json(*j) + warn_json(*j) + "}";
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
        if (fin) c += warn_json(*j);
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
// Admit a request that carries media only if its estimated transient memory fits the container headroom right now (reserved while it is in flight),
// instead of letting the allocation push the whole container into the OOM killer.
struct MemReserve {
    Server* S = nullptr; double mb = 0;
    ~MemReserve() { if (S && mb > 0) { std::lock_guard<std::mutex> g(S->mem_m); S->mem_inflight -= mb; } }
};
static int mem_admit(Server& S, const std::string& text, MemReserve& res, std::string& body, std::string& logline) {
    const double est = media_cost_mb(text);
    if (est <= 0) return 0;
    const double lim = cgroup_limit_mb();
    if (lim > 0) {
        const double used = cgroup_used_mb(), reserve = 96;
        std::lock_guard<std::mutex> g(S.mem_m);
        if (used + S.mem_inflight + est + reserve > lim) {
            const bool never = S.mem_inflight <= 0;                 // nothing else in flight: waiting will not help
            S.n_mem_rejected++;
            char b[400]; snprintf(b, sizeof b, "not enough memory for this request: it needs ~%.0f MB, %.0f MB of the %.0f MB container limit is free%s. %s",
                                  est, std::max(0.0, lim - used - S.mem_inflight - reserve), lim, never ? "" : " (other requests are in flight)",
                                  never ? "Send a smaller image (downscale), a crop, fewer frames, or a shorter clip." : "Retry shortly.");
            body = err_json(b, "server_error"); logline = "rejected: memory"; return never ? 413 : 503;
        }
        S.mem_inflight += est; res.S = &S; res.mb = est;
    }
    S.trim_pending = true;
    return 0;
}

static int handle_embeddings(Server& S, const std::string& text, std::string& body, std::string& logline) {
    MemReserve mem_res; if (int st = mem_admit(S, text, mem_res, body, logline)) return st;
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
    MemReserve mem_res; if (int st = mem_admit(S, text, mem_res, body, logline)) return st;
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

// Load (or re-load) the API keys: --api-key (repeatable) + --api-key-file. A failed re-read keeps the keys that were loaded.
static void reload_api_keys(Server& S) {
    ApiKeys fresh;
    for (const auto& k : S.o.api_keys) fresh.add_line(k, (int)fresh.keys.size() + 1);
    if (!S.o.api_key_file.empty()) {
        FILE* f = fopen(S.o.api_key_file.c_str(), "r");
        if (!f) { fprintf(stderr, "[serve] cannot read --api-key-file %s: %s (keeping the loaded keys)\n", S.o.api_key_file.c_str(), strerror(errno)); return; }
        char line[1024]; while (fgets(line, sizeof line, f)) fresh.add_line(line, (int)fresh.keys.size() + 1);
        fclose(f);
    }
    { std::lock_guard<std::mutex> g(S.keys.m); S.keys.keys = fresh.keys; }
    fprintf(stderr, "[serve] API keys loaded: %zu\n", fresh.keys.size());
}

// Prometheus text exposition (GET /metrics).
static std::string metrics_text(Server& S) {
    std::string o; char b[512];
    auto metric = [&](const char* name, const char* type, const char* help, double v) { snprintf(b, sizeof b, "# HELP %s %s\n# TYPE %s %s\n%s %.10g\n", name, help, name, type, name, v); o += b; };
    const double up = std::chrono::duration<double>(clk::now() - S.t_start).count();
    metric("ferro_uptime_seconds", "gauge", "Seconds since the server started.", up);
    metric("ferro_http_requests_total", "counter", "HTTP requests received.", (double)S.n_req.load());
    metric("ferro_http_errors_total", "counter", "HTTP responses with status >= 400.", (double)S.n_err.load());
    metric("ferro_connections", "gauge", "Open client connections.", S.conns.load());
    metric("ferro_gpu_busy_ratio", "gauge", "Share of uptime the embedding worker spent computing.", up > 0 ? S.busy_us.load() / 1e6 / up : 0.0);
    if (S.has_emb) {
        metric("ferro_embeddings_total", "counter", "Embedding vectors returned.", (double)S.n_emb.load());
        metric("ferro_embedding_tokens_total", "counter", "Tokens embedded.", (double)S.n_tok.load());
        metric("ferro_embedding_batches_total", "counter", "Embedding forward passes.", (double)S.n_batches.load());
    }
    if (S.has_lm) {
        size_t gq; { std::lock_guard<std::mutex> g(S.qm); gq = S.gq.size(); }
        metric("ferro_lm_requests_total", "counter", "Generation requests finished.", (double)S.n_gen_req.load());
        metric("ferro_lm_prompt_tokens_total", "counter", "Prompt tokens (including cached ones).", (double)S.n_prompt_tok.load());
        metric("ferro_lm_generated_tokens_total", "counter", "Tokens generated.", (double)S.n_gen_tok.load());
        metric("ferro_lm_active_sequences", "gauge", "Sequences in the decode batch.", S.n_active.load());
        metric("ferro_lm_prefilling_sequences", "gauge", "Prompts being filled slice by slice.", S.n_prefill.load());
        metric("ferro_lm_queued_requests", "gauge", "Generation requests waiting for a slot.", (double)gq);
        metric("ferro_lm_decode_tokens_per_second", "gauge", "Decode speed over the last 10 s of activity (tokens per busy second, all streams).", S.rw_decode.rate());
        metric("ferro_lm_prefill_tokens_per_second", "gauge", "Prefill speed over the last 10 s of activity.", S.rw_prefill.rate());
        metric("ferro_lm_spec_drafted_tokens_total", "counter", "Tokens drafted by prompt lookup.", (double)S.spec_drafted.load());
        metric("ferro_lm_spec_accepted_tokens_total", "counter", "Drafted tokens confirmed by verification.", (double)S.spec_accepted.load());
        metric("ferro_lm_prefix_cache_hits_total", "counter", "Requests that restored a cached prefix.", (double)S.pc_hits.load());
        metric("ferro_lm_prefix_cache_misses_total", "counter", "Requests with no usable cached prefix.", (double)S.pc_miss.load());
        metric("ferro_lm_prefix_cache_cached_tokens_total", "counter", "Prompt tokens served from the prefix cache.", (double)S.pc_saved.load());
        metric("ferro_lm_prefix_cache_evictions_total", "counter", "Prefix cache entries evicted.", (double)S.pc_evict.load());
        metric("ferro_lm_prefix_cache_entries", "gauge", "Entries in the device prefix cache.", (double)S.pc_entries.load());
        metric("ferro_lm_prefix_cache_bytes", "gauge", "Device memory held by the prefix cache.", (double)S.pc_used.load());
        if (S.pstore) {
            metric("ferro_lm_prefix_snapshots_entries", "gauge", "Prefix snapshots on disk for this model.", (double)S.pstore->entries());
            metric("ferro_lm_prefix_snapshots_bytes", "gauge", "Disk used by prefix snapshots.", (double)S.pstore->bytes());
            metric("ferro_lm_prefix_snapshot_disk_hits_total", "counter", "Requests that brought a snapshot in from disk.", (double)S.ps_hits.load());
        }
        o += "# HELP ferro_lm_ttft_seconds Time from request arrival to the first generated token.\n# TYPE ferro_lm_ttft_seconds histogram\n";
        static const char* le[8] = {"0.1", "0.25", "0.5", "1", "2.5", "5", "10", "+Inf"}; uint64_t cum = 0;
        for (int i = 0; i < 8; ++i) { cum += S.ttft_b[i].load(); snprintf(b, sizeof b, "ferro_lm_ttft_seconds_bucket{le=\"%s\"} %llu\n", le[i], (unsigned long long)cum); o += b; }
        snprintf(b, sizeof b, "ferro_lm_ttft_seconds_sum %.6f\nferro_lm_ttft_seconds_count %llu\n", S.ttft_us.load() / 1e6, (unsigned long long)S.ttft_n.load()); o += b;
    }
    if (S.has_d1) {
        metric("ferro_d1_requests_total", "counter", "Decision requests answered.", (double)S.n_d1_req.load());
        metric("ferro_d1_questions_total", "counter", "Decision questions scored.", (double)S.n_d1_q.load());
        metric("ferro_d1_tokens_total", "counter", "Tokens run through the decision model.", (double)S.n_d1_tok.load());
        metric("ferro_d1_passes_total", "counter", "Packed decision forward passes.", (double)S.n_d1_pass.load());
    }
    double inflight; { std::lock_guard<std::mutex> g(S.mem_m); inflight = S.mem_inflight; }
    metric("ferro_memory_limit_bytes", "gauge", "Container memory limit (0 = none found).", cgroup_limit_mb() * 1048576.0);
    metric("ferro_memory_used_bytes", "gauge", "Container anon + shmem memory in use.", cgroup_used_mb() * 1048576.0);
    metric("ferro_memory_reserved_bytes", "gauge", "Memory reserved by admitted media requests still in flight.", inflight * 1048576.0);
    metric("ferro_memory_rejected_total", "counter", "Media requests refused because they would not fit.", (double)S.n_mem_rejected.load());
    return o;
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
        if (S.keys_reload.exchange(false)) reload_api_keys(S);
        bool authed = true; std::string who;
        if (!S.keys.empty()) {
            authed = r.path == "/health" || r.path == "/" || r.path == "/chat" || r.path == "/ui" || (r.path == "/metrics" && S.o.metrics_open);
            if (!authed) {
                std::string presented;
                if (r.h.count("authorization") && r.h["authorization"].compare(0, 7, "Bearer ") == 0) presented = r.h["authorization"].substr(7);
                else if (r.h.count("x-api-key")) presented = r.h["x-api-key"];
                authed = !presented.empty() && S.keys.match(presented, &who);
            }
        }
        if (!authed) { status = 401; body = err_json("invalid or missing API key", "authentication_error"); }
        else if (r.path == "/health" || r.path == "/v1/health") body = health_json(S);
        else if (r.path == "/chat" || r.path == "/ui") { if (r.method != "GET") { status = 405; body = err_json("use GET", "invalid_request_error"); } else { body = kChatHtml; ctype = "text/html; charset=utf-8"; } }
        else if (r.path == "/metrics") { body = metrics_text(S); ctype = "text/plain; version=0.0.4; charset=utf-8"; }
        else if (r.path == "/tokenize" || r.path == "/v1/tokenize" || r.path == "/detokenize" || r.path == "/v1/detokenize") {      // vLLM's fields
            const bool detok = r.path.find("detokenize") != std::string::npos;
            try {
                if (r.method != "POST") throw BadRequest("use POST");
                if (!S.has_lm) throw BadRequest("no language model loaded (start the server with --lm)");
                Json b = json_parse(r.body); if (!b.is_object()) throw BadRequest("body must be a JSON object");
                if (detok) {
                    const Json* t = b.find("tokens"); if (!t || !t->is_array()) throw BadRequest("\"tokens\" must be an array of token ids");
                    if (t->arr.size() > 400000) throw BadRequest("at most 400000 tokens");
                    std::string text; for (const Json& x : t->arr) { const long id = x.t == Json::Int ? (long)x.i : -1; if (id < 0 || id >= S.lm.n_vocab()) throw BadRequest("token id out of range"); text += S.ltok.decode_piece((int)id); }
                    body = "{\"prompt\":" + dquote(text) + "}";
                } else {
                    std::string rendered; const Json* m = b.find("messages");
                    if (m && m->is_array()) { try { rendered = lfm_chat_render(*m, b.find("tools"), S.chat_opt); } catch (const std::exception& e) { throw BadRequest(std::string("messages: ") + e.what()); } }
                    else {
                        const Json* p = b.find("prompt"); if (!p || !p->is_string()) throw BadRequest("give \"prompt\" (a string) or \"messages\"");
                        const Json* sp = b.find("add_special_tokens"); const bool bos = !(sp && sp->t == Json::Bool && !sp->b);
                        rendered = (bos ? "<|startoftext|>" : "") + p->s;
                    }
                    const std::vector<int> ids = S.ltok.tokenize(rendered, true);
                    const Json* rs = b.find("return_token_strs"); const bool strs = rs && rs->t == Json::Bool && rs->b;
                    body = "{\"count\":" + std::to_string(ids.size()) + ",\"max_model_len\":" + std::to_string(S.lm.max_ctx()) + ",\"tokens\":[";
                    for (size_t i = 0; i < ids.size(); ++i) body += (i ? "," : "") + std::to_string(ids[i]);
                    body += "]";
                    if (strs) { body += ",\"token_strs\":["; for (size_t i = 0; i < ids.size(); ++i) body += (i ? "," : "") + dquote(S.ltok.decode_piece(ids[i])); body += "]"; }
                    body += "}";
                }
            } catch (const std::exception& e) { status = 400; body = err_json(e.what(), "invalid_request_error"); }
        }
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
            fprintf(stderr, "[serve] %s %s %d %.0fms %s%s\n", r.method.c_str(), r.path.c_str(), status, ms_since(t0), logline.c_str(), who.empty() ? "" : (" key=" + who).c_str());
        if (!ok || !r.keep) break;
    }
    close(fd);
    S.conns--;
}

}  // namespace

int run_server(const ServerOptions& opt) {
    {   // memory plan BEFORE anything is loaded: an iGPU's buffers are non-swappable system RAM, and an over-committed container is OOM-killed
        const MemPlan plan = plan_memory(opt);
        if (opt.plan_only) { fprintf(stderr, "[plan] memory for this configuration:\n"); print_plan(plan, stderr); return plan.fits ? 0 : 3; }
        if (!plan.fits) {
            fprintf(stderr, "[serve] memory plan: ~%.0f MB needed vs a %.0f MB container limit\n", plan.need_mb, plan.limit_mb); print_plan(plan, stderr);
            if (!opt.no_memory_check) { fprintf(stderr, "[serve] refusing to start (an over-committed container is OOM-killed). Pass --no-memory-check to start anyway.\n"); return 1; }
            fprintf(stderr, "[serve] --no-memory-check: starting anyway\n");
        } else if (plan.limit_mb > 0 && plan.need_mb > 0.85 * plan.limit_mb)
            fprintf(stderr, "[serve] note: ~%.0f MB resident of a %.0f MB container limit; large image/video requests are admitted only while they fit\n", plan.need_mb, plan.limit_mb);
    }
    Server& S = *new Server;                       // intentionally leaked: detached connection threads may outlive this function
    S.o = opt; S.has_mm = !opt.mmproj.empty(); S.has_emb = !opt.model.empty(); S.has_lm = !opt.lm.empty(); S.has_d1 = !opt.d1.empty(); g_srv = &S;
    if (!opt.api_keys.empty() || !opt.api_key_file.empty()) { reload_api_keys(S); if (S.keys.empty()) { fprintf(stderr, "[serve] no usable API key found\n"); return 1; } }
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
            const double kv_mb = (double)lm.n_attn_layers * S.lm.n_slots() * S.lm.max_ctx() * 2 * lm.n_kv * lm.head_dim * (S.lm.kv_int8() ? 1.125 : 4.0) / 1048576.0;
            double lim_mb = 0; if (FILE* f = fopen("/sys/fs/cgroup/memory.max", "r")) { unsigned long long v = 0; if (fscanf(f, "%llu", &v) == 1) lim_mb = v / 1048576.0; fclose(f); }
            fprintf(stderr, "[serve] KV cache %.0f MB (%d slot(s) x %d ctx)%s\n", kv_mb, S.lm.n_slots(), S.lm.max_ctx(), lim_mb > 0 ? (" | container memory limit " + std::to_string((long)lim_mb) + " MB").c_str() : "");
        }
    }
    if (S.has_d1) {
        S.d1_name = S.d1.name();
        fprintf(stderr, "[serve] decision model \"%s\" ready (%s, image %s, audio %s) - POST /v1/systemone\n", S.d1_name.c_str(), S.d1.on_gpu() ? "iGPU" : "CPU", S.d1.has_vision() ? "yes" : "no", S.d1.has_audio() ? "yes" : "no");
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
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal); signal(SIGPIPE, SIG_IGN); signal(SIGHUP, on_hup);
    fprintf(stderr, "[serve] listening on http://%s:%d  (%s%s%sGET /v1/models, GET /health)%s\n", opt.host.c_str(), opt.port, S.has_emb ? "POST /v1/embeddings, " : "", S.has_lm ? "POST /v1/chat/completions, POST /v1/completions, " : "",
            S.has_d1 ? "POST /v1/systemone, " : "", S.keys.empty() ? "" : "  [API key required]");
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
