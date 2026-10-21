// server.cpp - see server.h
//
// Threads: the main thread accepts connections; each connection gets a handler thread (HTTP/1.1 keep-alive); ONE inference worker
// owns the engine, the GPU and the thread pool (the device is a single in-order queue, so inference is serialised by design).
// Scheduling: the worker takes the oldest job; if it is text-only it also takes every following text-only job (and briefly waits
// for requests that are already in flight) so they share packed forward passes - that is where the iGPU throughput is
// (16 x 188 tokens: 850 tok/s; one short sequence at a time is several times slower per token). Image / audio / video jobs run
// alone, in arrival order.
#include "server.h"
#include "embed_mm.h"
#include "json.h"
#include "threadpool.h"

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
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
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

// ---------------------------------------------------------------- server state
struct Server {
    ServerOptions o;
    MmEmbedder me;
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

// ---------------------------------------------------------------- inference worker
static void worker_loop(Server& S, std::promise<std::string>& ready) {
    {
        std::string err;
        if (!S.me.init(S.o.model, S.o.mmproj, S.o.threads, &err)) { ready.set_value(err.empty() ? "init failed" : err); return; }
        ready.set_value(std::string());
    }
    MmEmbedOptions mo;
    while (true) {
        std::vector<std::shared_ptr<Job>> batch;
        {
            std::unique_lock<std::mutex> lk(S.qm);
            S.qcv.wait(lk, [&] { return S.stop || !S.q.empty(); });
            if (S.stop && S.q.empty()) break;
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
    // shutting down: fail whatever is still queued
    std::lock_guard<std::mutex> lk(S.qm);
    for (auto& j : S.q) { j->err = "server is shutting down"; j->status = 503; { std::lock_guard<std::mutex> g(j->m); j->done = true; } j->cv.notify_all(); }
    S.q.clear();
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
    return "{\"object\":\"list\",\"data\":[{\"id\":\"" + jesc(S.model_name) + "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"ferro\"," +
           "\"capabilities\":[\"embeddings\"],\"modalities\":[\"text\"" + (S.has_mm ? ",\"image\",\"audio\",\"video\"" : "") + "],\"dimensions\":" + std::to_string(S.dim) + "}]}";
}
static std::string health_json(Server& S) {
    size_t qd; { std::lock_guard<std::mutex> g(S.qm); qd = S.q.size(); }
    const double up = ms_since(S.t_start) / 1000.0;
    char b[512];
    snprintf(b, sizeof b, "{\"status\":\"ok\",\"model\":\"%s\",\"multimodal\":%s,\"queue\":%zu,\"inflight\":%d,\"requests\":%llu,\"errors\":%llu,"
                          "\"embeddings\":%llu,\"tokens\":%llu,\"batches\":%llu,\"uptime_s\":%.1f,\"gpu_busy\":%.3f}",
             jesc(S.model_name).c_str(), S.has_mm ? "true" : "false", qd, S.inflight.load(), (unsigned long long)S.n_req.load(),
             (unsigned long long)S.n_err.load(), (unsigned long long)S.n_emb.load(), (unsigned long long)S.n_tok.load(),
             (unsigned long long)S.n_batches.load(), up, up > 0 ? S.busy_us.load() / 1e6 / up : 0.0);
    return b;
}

// POST /v1/embeddings. Returns the HTTP status; `body` is the response body.
static int handle_embeddings(Server& S, const std::string& text, std::string& body, std::string& logline) {
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
        else if (r.path == "/") body = "{\"name\":\"ferro-embed\",\"endpoints\":[\"POST /v1/embeddings\",\"GET /v1/models\",\"GET /health\"]}";
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
    S.o = opt; S.has_mm = !opt.mmproj.empty(); g_srv = &S;
    std::promise<std::string> ready; auto ready_f = ready.get_future();
    fprintf(stderr, "[serve] loading %s%s ...\n", opt.model.c_str(), S.has_mm ? " + mmproj" : "");
    const auto t0 = clk::now();
    std::thread worker([&] { worker_loop(S, ready); });
    const std::string err = ready_f.get();
    if (!err.empty()) { fprintf(stderr, "[serve] init failed: %s\n", err.c_str()); worker.join(); Pool::get().shutdown(); return 1; }
    S.dim = S.me.dim_out();
    S.model_name = opt.model_name;
    if (S.model_name.empty()) { GgufFile g; std::string e; S.model_name = (g.open_file(opt.model.c_str(), &e) && !g.get_str("general.name").empty()) ? g.get_str("general.name") : "embeddinggemma-2"; }
    fprintf(stderr, "[serve] model \"%s\" ready in %.1fs (dim %d, %s)\n", S.model_name.c_str(), std::chrono::duration<double>(clk::now() - t0).count(), S.dim,
            S.has_mm ? "text+image+audio+video" : "text only");

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
    fprintf(stderr, "[serve] listening on http://%s:%d  (POST /v1/embeddings, GET /v1/models, GET /health)%s\n", opt.host.c_str(), opt.port, opt.api_key.empty() ? "" : "  [API key required]");
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
