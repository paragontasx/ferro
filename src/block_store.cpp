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
static const uint32_t kVersion = 1;

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
    size_t n = 0, bad = 0;
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
                    h.n_conv != geom_.n_conv || h.conv_layer_bytes != geom_.conv_layer_bytes || h.n_tokens != blk) { ++bad; continue; }
                struct stat st; if (stat(p.c_str(), &st) != 0) continue;
                Rec r; r.path = p; r.bytes = (uint64_t)st.st_size; r.mtime = (int64_t)st.st_mtime; r.used = ++tick_; r.n_tokens = h.n_tokens;
                idx_[h.chain] = std::move(r); ++n;
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
    const int maxb = (n - 1) / blk_;                       // a usable hit leaves at least one token to prefill
    if (maxb <= 0) return false;
    std::vector<uint64_t> chain; chain.reserve((size_t)maxb);
    uint64_t h = root_;
    for (int i = 0; i < maxb; ++i) {
        h = chain_next(h, prompt.data() + (size_t)i * blk_, blk_);
        auto it = idx_.find(h);
        if (it == idx_.end()) break;
        chain.push_back(h);
    }
    if (chain.empty()) return false;
    const int hit = (int)chain.size() * blk_;
    out->toks.assign(prompt.begin(), prompt.begin() + hit);
    out->chain = std::move(chain);
    const auto& r = idx_.at(out->chain.back());
    out->path = r.path; out->mtime = r.mtime; out->bytes = r.bytes;
    return true;
}

bool BlockStore::load(const Entry& e, std::vector<uint8_t>* blob) const {
    if (e.chain.empty() || blob == nullptr) return false;
    const int N = (int)e.chain.size() * blk_;
    if ((int)e.toks.size() != N) return false;
    const uint64_t per_tok = (uint64_t)geom_.row_bytes + geom_.scale_row_bytes;
    const size_t dst_layer = (size_t)((uint64_t)N * per_tok);
    blob->assign(blob_bytes(N), 0);
    std::vector<uint8_t> kv, cst; std::vector<int> toks;
    size_t nread = 0;
    for (size_t j = 0; j < e.chain.size(); ++j) {
        std::string path;
        { std::lock_guard<std::mutex> g(m_); auto it = idx_.find(e.chain[j]); if (it == idx_.end()) return false; path = it->second.path; }
        if (!read_block(path, e.chain[j], &kv, &cst, &toks)) return false;
        if ((int)toks.size() != blk_ || !std::equal(toks.begin(), toks.end(), e.toks.begin() + (long)(j * blk_))) return false;   // no collisions, ever
        const size_t src_layer = (size_t)((uint64_t)blk_ * per_tok);
        for (uint32_t l = 0; l < geom_.n_layers; ++l) {
            const uint8_t* s = kv.data() + (size_t)l * src_layer;
            uint8_t* d = blob->data() + (size_t)l * dst_layer;
            std::memcpy(d + (size_t)j * blk_ * geom_.row_bytes, s, (size_t)blk_ * geom_.row_bytes);
            if (geom_.scale_row_bytes)
                std::memcpy(d + (size_t)N * geom_.row_bytes + (size_t)j * blk_ * geom_.scale_row_bytes,
                            s + (size_t)blk_ * geom_.row_bytes, (size_t)blk_ * geom_.scale_row_bytes);
        }
        ++nread;
    }
    // conv state: only the deepest block's is valid (it is the state as of the end of the chain)
    if (!cst.empty()) std::memcpy(blob->data() + (size_t)((uint64_t)N * per_tok * geom_.n_layers), cst.data(), cst.size());
    { std::lock_guard<std::mutex> g(m_); ++n_loads_; }
    return true;
}

void BlockStore::save_async(const std::vector<int>& prompt, int end, std::vector<uint8_t>&& kv, std::vector<uint8_t>&& cst) {
    if (end <= 0 || end % blk_ != 0 || end > (int)prompt.size()) return;
    const size_t want_kv = (size_t)blk_ * ((size_t)geom_.row_bytes + geom_.scale_row_bytes) * geom_.n_layers;
    if (kv.size() != want_kv || cst.size() != (size_t)geom_.n_conv * geom_.conv_layer_bytes) return;
    Job j;
    j.toks.assign(prompt.begin() + (end - blk_), prompt.begin() + end);
    j.end = end; j.kv = std::move(kv); j.cst = std::move(cst);
    uint64_t h = root_, parent = 0;
    for (int i = 0; i < end / blk_; ++i) { parent = h; h = chain_next(h, prompt.data() + (size_t)i * blk_, blk_); }
    j.chain = h;
    j.parent = parent;                                          // the chain hash of the block before this one (0 = root block)
    {
        std::lock_guard<std::mutex> g(m_);
        if (stop_ || idx_.count(h)) return;                     // already stored: blocks are content-addressed, so this is dedupe
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
    h.blk = blk_; h.n_tokens = blk_;
    h.n_layers = geom_.n_layers; h.n_conv = geom_.n_conv; h.row_bytes = geom_.row_bytes;
    h.scale_row_bytes = geom_.scale_row_bytes; h.conv_layer_bytes = geom_.conv_layer_bytes;
    h.kv_bytes = j.kv.size(); h.cst_bytes = j.cst.size(); h.tok_bytes = j.toks.size() * sizeof(int);
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
        Rec r; r.path = path; r.bytes = (uint64_t)st.st_size; r.mtime = (int64_t)st.st_mtime; r.used = ++tick_; r.n_tokens = blk_;
        idx_[j.chain] = std::move(r);
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
    while (total > budget_ && idx_.size() > 1) {
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
}

void BlockStore::touch(const Entry& e) {
    std::lock_guard<std::mutex> g(m_);
    for (uint64_t h : e.chain) { auto it = idx_.find(h); if (it != idx_.end()) it->second.used = ++tick_; }
}
bool BlockStore::has_chain(uint64_t chain) const { std::lock_guard<std::mutex> g(m_); return idx_.count(chain) != 0; }
size_t BlockStore::entries() const { std::lock_guard<std::mutex> g(m_); return idx_.size(); }
uint64_t BlockStore::bytes() const { std::lock_guard<std::mutex> g(m_); uint64_t b = 0; for (const auto& kv : idx_) b += kv.second.bytes; return b; }
void BlockStore::flush() { std::unique_lock<std::mutex> lk(m_); idle_cv_.wait(lk, [&] { return jobs_.empty() && !busy_; }); }
void BlockStore::close() {
    { std::lock_guard<std::mutex> g(m_); if (stop_) return; stop_ = true; }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

}  // namespace ferro
