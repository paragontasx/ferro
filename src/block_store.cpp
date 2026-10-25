// block_store.cpp — block-granular prompt cache on local disk (see block_store.h for the model).
#include "block_store.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ferro {

static const char kMagic[8] = {'F', 'E', 'R', 'R', 'O', 'B', 'L', 'K'};
static const uint32_t kVersion = 2;   // 2: header carries is_tip (tips may be any length)

#pragma pack(push, 1)
struct Hdr {
    char magic[8];
    uint32_t version;
    uint32_t hdr_bytes;
    uint64_t state_id, chain, parent;
    int32_t blk, n_tokens;
    uint32_t n_layers, n_conv, row_bytes, scale_row_bytes, conv_layer_bytes;
    uint64_t kv_bytes, cst_bytes, tok_bytes;
    uint32_t crc;
    uint32_t is_tip;
    int64_t created;
};
#pragma pack(pop)

static uint32_t crc32_(const void* p, size_t n) {
    static uint32_t t[256]; static bool init = false;
    if (!init) { for (uint32_t i = 0; i < 256; ++i) { uint32_t c = i; for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; t[i] = c; } init = true; }
    const uint8_t* b = (const uint8_t*)p; uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = t[(c ^ b[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
static uint64_t fnv1a(const void* p, size_t n, uint64_t h) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
static bool mkdir_p(const std::string& d) {
    if (d.empty()) return false;
    std::string cur;
    for (size_t i = 0; i <= d.size(); ++i) {
        if (i == d.size() || d[i] == '/') {
            if (!cur.empty() && mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        if (i < d.size()) cur += d[i];
    }
    return true;
}

BlockStore::~BlockStore() { close(); }

size_t BlockStore::blob_bytes(int n_tokens) const {
    const uint64_t per_tok = (uint64_t)geom_.row_bytes + geom_.scale_row_bytes;
    return (size_t)((uint64_t)n_tokens * per_tok * geom_.n_layers + (uint64_t)geom_.n_conv * geom_.conv_layer_bytes);
}

uint64_t BlockStore::chain_next(uint64_t parent, const int* toks, int n) const {
    // The model identity is mixed into every step, so two models that see identical tokens still get different chains.
    return mix64(parent ^ fnv1a(toks, (size_t)n * sizeof(int), 1469598103934665603ull) ^ mix64(id_));
}

uint64_t BlockStore::chain_at(const std::vector<int>& prompt, int end) const {
    uint64_t h = root_;
    for (int i = 0; i < end / blk_; ++i) h = chain_next(h, prompt.data() + (size_t)i * blk_, blk_);
    return h;
}

std::string BlockStore::path_for(uint64_t chain) const {
    char hex[17]; snprintf(hex, sizeof hex, "%016llx", (unsigned long long)chain);
    std::string sub(hex, 2);
    return dir_ + "/" + sub + "/" + hex + ".blk";
}

bool BlockStore::open(const std::string& dir, uint64_t id, int blk, uint64_t budget_bytes, const BlockGeom& geom, std::string* err) {
    if (blk <= 0 || geom.n_layers == 0 || geom.row_bytes == 0) { if (err) *err = "bad block geometry"; return false; }
    dir_ = dir; id_ = id; blk_ = blk; budget_ = budget_bytes; geom_ = geom;
    root_ = mix64(id_ ^ 0x9E3779B97F4A7C15ull);
    if (!mkdir_p(dir_)) { if (err) *err = "cannot create " + dir_; return false; }
    // index what is already on disk: headers only, no payload is read
    size_t bad = 0;
    if (DIR* d = opendir(dir_.c_str())) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            const std::string sub = dir_ + "/" + e->d_name;
            DIR* sd = opendir(sub.c_str()); if (!sd) continue;
            while (dirent* f = readdir(sd)) {
                const size_t ln = std::strlen(f->d_name);
                if (ln < 5 || std::strcmp(f->d_name + ln - 4, ".blk") != 0) continue;
                const std::string p = sub + "/" + f->d_name;
                FILE* fp = fopen(p.c_str(), "rb"); if (!fp) continue;
                Hdr h; bool ok = fread(&h, sizeof h, 1, fp) == 1;
                fclose(fp);
                if (!ok || std::memcmp(h.magic, kMagic, 8) || h.version != kVersion || h.state_id != id_ || h.blk != blk ||
                    h.n_layers != geom_.n_layers || h.row_bytes != geom_.row_bytes || h.scale_row_bytes != geom_.scale_row_bytes ||
                    h.n_conv != geom_.n_conv || h.conv_layer_bytes != geom_.conv_layer_bytes ||
                    h.n_tokens <= 0 || h.n_tokens > 4 * blk) { ++bad; continue; }
                struct stat st; if (stat(p.c_str(), &st) != 0) continue;
                if (!h.is_tip) {
                    Rec r; r.path = p; r.bytes = (uint64_t)st.st_size; r.mtime = (int64_t)st.st_mtime; r.used = ++tick_; r.n_tokens = h.n_tokens;
                    idx_[h.chain] = std::move(r);
                } else {
                    auto it = tips_.find(h.parent);
                    if (it != tips_.end() && it->second.n_tokens >= h.n_tokens) continue;      // keep the longer tip of a conversation
                    Tip t; t.path = p; t.bytes = (uint64_t)st.st_size; t.mtime = (int64_t)st.st_mtime; t.used = ++tick_;
                    t.n_tokens = h.n_tokens; t.hash = h.chain;
                    tips_[h.parent] = std::move(t);
                }
            }
            closedir(sd);
        }
        closedir(d);
    }
    std::thread t([this] { writer_loop(); }); th_ = std::move(t);
    if (bad) fprintf(stderr, "[blocks] %zu foreign/corrupt block files ignored in %s\n", bad, dir_.c_str());
    return true;
}

bool BlockStore::read_block(const std::string& path, uint64_t chain, std::vector<uint8_t>* kv, std::vector<uint8_t>* cst, std::vector<int>* toks) const {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return false;
    Hdr h; bool ok = fread(&h, sizeof h, 1, f) == 1;
    ok = ok && !std::memcmp(h.magic, kMagic, 8) && h.version == kVersion && h.state_id == id_ && h.chain == chain;
    ok = ok && h.kv_bytes == (uint64_t)h.n_tokens * ((uint64_t)geom_.row_bytes + geom_.scale_row_bytes) * geom_.n_layers;
    ok = ok && h.cst_bytes == (uint64_t)geom_.n_conv * geom_.conv_layer_bytes && h.tok_bytes == (uint64_t)h.n_tokens * sizeof(int);
    if (ok) {
        toks->resize((size_t)h.n_tokens); kv->resize((size_t)h.kv_bytes); cst->resize((size_t)h.cst_bytes);
        ok = fread(toks->data(), 1, (size_t)h.tok_bytes, f) == (size_t)h.tok_bytes
          && fread(kv->data(), 1, (size_t)h.kv_bytes, f) == (size_t)h.kv_bytes
          && fread(cst->data(), 1, (size_t)h.cst_bytes, f) == (size_t)h.cst_bytes;
        if (ok) {
            const uint32_t c = crc32_(toks->data(), (size_t)h.tok_bytes);
            const uint32_t c2 = crc32_(kv->data(), (size_t)h.kv_bytes);
            const uint32_t c3 = crc32_(cst->data(), (size_t)h.cst_bytes);
            ok = (c ^ c2 ^ c3) == h.crc;
        }
    }
    fclose(f);
    return ok;
}

bool BlockStore::longest(const std::vector<int>& prompt, Entry* out) const {
    std::lock_guard<std::mutex> g(m_);
    const int n = (int)prompt.size();
    std::vector<uint64_t> chain;
    uint64_t h = root_;
    const int maxb = n / blk_;                             // whole blocks that fit
    for (int i = 0; i < maxb; ++i) {
        const uint64_t nh = chain_next(h, prompt.data() + (size_t)i * blk_, blk_);
        if (!idx_.count(nh)) break;
        chain.push_back(nh); h = nh;
    }
    const int hit = (int)chain.size() * blk_;
    const uint64_t parent = chain.empty() ? root_ : chain.back();
    int tip = 0;
    auto t = tips_.find(parent);
    if (t != tips_.end()) {
        const int L = t->second.n_tokens;
        // usable only if it leaves a token to prefill and those tokens really are that conversation's end
        if (L > 0 && hit + L < n && chain_next(parent, prompt.data() + hit, L) == t->second.hash) tip = L;
    }
    if (hit == 0 && tip == 0) return false;
    out->toks.assign(prompt.begin(), prompt.begin() + hit + tip);
    out->chain = std::move(chain);
    out->tip_len = tip;
    const Rec& r = !out->chain.empty() ? idx_.at(out->chain.back()) : tips_.at(root_);
    out->path = r.path; out->mtime = r.mtime; out->bytes = r.bytes;
    return true;
}

bool BlockStore::load(const Entry& e, std::vector<uint8_t>* blob) const {
    if (blob == nullptr || (e.chain.empty() && e.tip_len == 0)) return false;
    const int N = (int)e.toks.size();
    if (N <= 0) return false;
    const uint64_t per_tok = (uint64_t)geom_.row_bytes + geom_.scale_row_bytes;
    const size_t dst_layer = (size_t)((uint64_t)N * per_tok);
    blob->assign(blob_bytes(N), 0);
    struct Seg { uint64_t hash; int start, len; };
    std::vector<Seg> segs; segs.reserve(e.chain.size() + 1);
    for (size_t j = 0; j < e.chain.size(); ++j) segs.push_back({e.chain[j], (int)(j * (size_t)blk_), blk_});
    const uint64_t parent = e.chain.empty() ? root_ : e.chain.back();
    if (e.tip_len > 0) {
        uint64_t ph = 0;
        { std::lock_guard<std::mutex> g(m_);
          auto it = tips_.find(parent);
          if (it == tips_.end() || it->second.n_tokens != e.tip_len) return false;
          ph = it->second.hash; }
        segs.push_back({ph, (int)e.chain.size() * blk_, e.tip_len});
    }
    std::vector<uint8_t> kv, cst, last_cst; std::vector<int> toks;
    for (const Seg& sg : segs) {
        std::string path;
        { std::lock_guard<std::mutex> g(m_);
          auto it = idx_.find(sg.hash);
          if (it != idx_.end()) path = it->second.path;
          else { auto tt = tips_.find(parent);
                 if (tt == tips_.end() || tt->second.hash != sg.hash) return false;
                 path = tt->second.path; } }
        if (!read_block(path, sg.hash, &kv, &cst, &toks)) return false;
        if ((int)toks.size() != sg.len || !std::equal(toks.begin(), toks.end(), e.toks.begin() + sg.start)) return false;   // no collisions, ever
        const size_t src_layer = (size_t)((uint64_t)sg.len * per_tok);
        for (uint32_t l = 0; l < geom_.n_layers; ++l) {
            const uint8_t* s0 = kv.data() + (size_t)l * src_layer;
            uint8_t* d = blob->data() + (size_t)l * dst_layer;
            std::memcpy(d + (size_t)sg.start * geom_.row_bytes, s0, (size_t)sg.len * geom_.row_bytes);
            if (geom_.scale_row_bytes)
                std::memcpy(d + (size_t)N * geom_.row_bytes + (size_t)sg.start * geom_.scale_row_bytes,
                            s0 + (size_t)sg.len * geom_.row_bytes, (size_t)sg.len * geom_.scale_row_bytes);
        }
        last_cst = cst;                                     // only the deepest segment's conv state is valid
    }
    if (!last_cst.empty()) std::memcpy(blob->data() + (size_t)((uint64_t)N * per_tok * geom_.n_layers), last_cst.data(), last_cst.size());
    { std::lock_guard<std::mutex> g(m_); ++n_loads_; }
    return true;
}

void BlockStore::save_async(const std::vector<int>& prompt, int end, std::vector<uint8_t>&& kv, std::vector<uint8_t>&& cst) {
    if (end <= 0 || end % blk_ != 0 || end > (int)prompt.size()) return;
    const size_t want_kv = (size_t)blk_ * ((size_t)geom_.row_bytes + geom_.scale_row_bytes) * geom_.n_layers;
    if (kv.size() != want_kv || cst.size() != (size_t)geom_.n_conv * geom_.conv_layer_bytes) return;
    Job j;
    j.toks.assign(prompt.begin() + (end - blk_), prompt.begin() + end);
    j.start = end - blk_; j.end = end; j.kv = std::move(kv); j.cst = std::move(cst);
    j.chain = chain_at(prompt, end);
    j.parent = chain_at(prompt, end - blk_);                    // the block before this one (root_ = first block)
    {
        std::lock_guard<std::mutex> g(m_);
        if (stop_ || idx_.count(j.chain)) return;               // already stored: content-addressed, so this is dedupe
        jobs_.push_back(std::move(j));
    }
    cv_.notify_one();
}

void BlockStore::save_tip_async(const std::vector<int>& prompt, int start, int end, std::vector<uint8_t>&& kv, std::vector<uint8_t>&& cst) {
    if (end <= 0 || end > (int)prompt.size() || start < 0 || start >= end) return;
    const int L = end - start;
    if (L % blk_ == 0) return;                                  // lands on a boundary: a whole block covers it
    const size_t want_kv = (size_t)L * ((size_t)geom_.row_bytes + geom_.scale_row_bytes) * geom_.n_layers;
    if (kv.size() != want_kv || cst.size() != (size_t)geom_.n_conv * geom_.conv_layer_bytes) return;
    Job j;
    const int base = start;
    j.toks.assign(prompt.begin() + base, prompt.begin() + end);
    j.start = base; j.end = end; j.tip = true; j.kv = std::move(kv); j.cst = std::move(cst);
    // walk to the parent, remembering every block on the way: the tips hanging off them are this conversation's
    // older ends, superseded by the one we are about to write
    std::vector<uint64_t> ancestors;
    uint64_t h = root_;
    ancestors.push_back(h);
    for (int i = 0; i < base / blk_; ++i) { h = chain_next(h, prompt.data() + (size_t)i * blk_, blk_); ancestors.push_back(h); }
    j.parent = h;                                               // the last whole block this tip hangs off
    j.chain = chain_next(j.parent, prompt.data() + base, L);
    {
        std::lock_guard<std::mutex> g(m_);
        if (stop_) return;
        auto it = tips_.find(j.parent);
        if (it != tips_.end() && it->second.hash == j.chain) return;   // identical tip already there
        for (size_t a = 0; a + 1 < ancestors.size(); ++a) {           // one tip per conversation, not one per boundary
            auto old = tips_.find(ancestors[a]);
            if (old == tips_.end()) continue;
            remove(old->second.path.c_str());
            tips_.erase(old);
        }
        jobs_.push_back(std::move(j));
    }
    cv_.notify_one();
}

bool BlockStore::write_block(const Job& j, std::string* err) {
    const std::string path = path_for(j.chain);
    mkdir_p(path.substr(0, path.rfind('/')));
    const std::string tmp = path + ".tmp";
    Hdr h; std::memset(&h, 0, sizeof h);
    std::memcpy(h.magic, kMagic, 8);
    h.version = kVersion; h.hdr_bytes = sizeof(Hdr);
    h.state_id = id_; h.chain = j.chain; h.parent = j.parent;
    h.blk = blk_; h.n_tokens = j.end - j.start;
    h.n_layers = geom_.n_layers; h.n_conv = geom_.n_conv; h.row_bytes = geom_.row_bytes;
    h.scale_row_bytes = geom_.scale_row_bytes; h.conv_layer_bytes = geom_.conv_layer_bytes;
    h.kv_bytes = j.kv.size(); h.cst_bytes = j.cst.size(); h.tok_bytes = j.toks.size() * sizeof(int);
    h.is_tip = j.tip ? 1 : 0;
    h.created = (int64_t)time(nullptr);
    h.crc = crc32_(j.toks.data(), h.tok_bytes) ^ crc32_(j.kv.data(), h.kv_bytes) ^ crc32_(j.cst.data(), h.cst_bytes);
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { if (err) *err = std::string("cannot write ") + tmp; return false; }
    bool ok = fwrite(&h, sizeof h, 1, f) == 1
           && fwrite(j.toks.data(), 1, h.tok_bytes, f) == h.tok_bytes
           && fwrite(j.kv.data(), 1, h.kv_bytes, f) == h.kv_bytes
           && fwrite(j.cst.data(), 1, h.cst_bytes, f) == h.cst_bytes;
    ok = fflush(f) == 0 && ok;
    fclose(f);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) { remove(tmp.c_str()); if (err) *err = "short write"; return false; }
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        std::lock_guard<std::mutex> g(m_);
        if (j.tip) {
            auto it = tips_.find(j.parent);
            if (it != tips_.end() && it->second.path != path) remove(it->second.path.c_str());   // one tip per conversation
            Tip t; t.path = path; t.bytes = (uint64_t)st.st_size; t.mtime = (int64_t)st.st_mtime; t.used = ++tick_;
            t.n_tokens = h.n_tokens; t.hash = j.chain;
            tips_[j.parent] = std::move(t);
        } else {
            Rec r; r.path = path; r.bytes = (uint64_t)st.st_size; r.mtime = (int64_t)st.st_mtime; r.used = ++tick_; r.n_tokens = h.n_tokens;
            idx_[j.chain] = std::move(r);
        }
        enforce_budget_locked(path);
        ++n_saved_;
    }
    return true;
}

void BlockStore::writer_loop() {
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_ || !jobs_.empty(); });
            if (stop_ && jobs_.empty()) return;
            j = std::move(jobs_.front()); jobs_.pop_front();
            busy_ = true;
        }
        std::string err;
        if (!write_block(j, &err)) fprintf(stderr, "[blocks] %s\n", err.c_str());
        {
            std::lock_guard<std::mutex> g(m_);
            busy_ = false;
        }
        idle_cv_.notify_all();
    }
}

void BlockStore::enforce_budget_locked(const std::string& keep) {
    if (!budget_) return;
    uint64_t total = 0;
    for (const auto& kv : idx_) total += kv.second.bytes;
    for (const auto& kv : tips_) total += kv.second.bytes;
    while (total > budget_ && idx_.size() > 1) {                 // whole blocks are the big ones: evict them first
        auto worst = idx_.begin();
        for (auto it = idx_.begin(); it != idx_.end(); ++it) {
            if (it->second.path == keep) continue;
            if (worst->second.path == keep || it->second.used < worst->second.used) worst = it;
        }
        if (worst->second.path == keep) break;
        remove(worst->second.path.c_str());
        total -= worst->second.bytes;
        idx_.erase(worst);
    }
    while (total > budget_ && !tips_.empty()) {
        auto worst = tips_.begin();
        for (auto it = tips_.begin(); it != tips_.end(); ++it) if (it->second.used < worst->second.used) worst = it;
        remove(worst->second.path.c_str());
        total -= worst->second.bytes;
        tips_.erase(worst);
    }
}

int BlockStore::block_depth(const std::vector<int>& prompt) const {
    std::lock_guard<std::mutex> g(m_);
    uint64_t h = root_;
    int k = 0;
    const int maxb = (int)prompt.size() / blk_;
    for (int i = 0; i < maxb; ++i) {
        h = chain_next(h, prompt.data() + (size_t)i * blk_, blk_);
        if (!idx_.count(h)) break;
        ++k;
    }
    return k * blk_;
}

void BlockStore::touch(const Entry& e) {
    std::lock_guard<std::mutex> g(m_);
    for (uint64_t h : e.chain) { auto it = idx_.find(h); if (it != idx_.end()) it->second.used = ++tick_; }
    if (e.tip_len > 0) { auto it = tips_.find(e.chain.empty() ? root_ : e.chain.back()); if (it != tips_.end()) it->second.used = ++tick_; }
}
bool BlockStore::has_chain(uint64_t chain) const { std::lock_guard<std::mutex> g(m_); return idx_.count(chain) != 0; }
size_t BlockStore::entries() const { std::lock_guard<std::mutex> g(m_); return idx_.size(); }
size_t BlockStore::tips() const { std::lock_guard<std::mutex> g(m_); return tips_.size(); }
uint64_t BlockStore::bytes() const {
    std::lock_guard<std::mutex> g(m_);
    uint64_t b = 0;
    for (const auto& kv : idx_) b += kv.second.bytes;
    for (const auto& kv : tips_) b += kv.second.bytes;
    return b;
}
uint64_t BlockStore::blocks_bytes() const {
    std::lock_guard<std::mutex> g(m_);
    uint64_t b = 0;
    for (const auto& kv : idx_) b += kv.second.bytes;
    return b;
}
void BlockStore::flush() { std::unique_lock<std::mutex> lk(m_); idle_cv_.wait(lk, [&] { return jobs_.empty() && !busy_; }); }
void BlockStore::close() {
    { std::lock_guard<std::mutex> g(m_); if (stop_) return; stop_ = true; }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

}  // namespace ferro
