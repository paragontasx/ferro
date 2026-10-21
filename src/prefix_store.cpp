#include "prefix_store.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/time.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <ctime>
#include <cstring>

namespace ferro {

static const char kMagic[8] = {'F', 'E', 'R', 'R', 'O', 'S', 'N', 'P'};
static const uint32_t kVersion = 1;
#pragma pack(push, 1)
struct FileHeader { char magic[8]; uint32_t version; uint64_t id; uint32_t n_tokens; uint64_t payload_bytes; uint32_t crc; };
#pragma pack(pop)

static uint32_t crc32(const uint8_t* p, size_t n) {
    static uint32_t table[256]; static bool init = false;
    if (!init) { for (uint32_t i = 0; i < 256; ++i) { uint32_t c = i; for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; table[i] = c; } init = true; }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
static uint64_t fnv(const void* d, size_t n, uint64_t h = 1469598103934665603ull) { const uint8_t* p = (const uint8_t*)d; for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; }

std::string PrefixStore::name_for(uint64_t id, const std::vector<int>& toks) {
    char b[64]; snprintf(b, sizeof b, "%016llx-%016llx.fsnap", (unsigned long long)id, (unsigned long long)fnv(toks.data(), toks.size() * sizeof(int)));
    return b;
}

bool PrefixStore::open(const std::string& dir, uint64_t id, uint64_t budget_bytes, std::string* err) {
    dir_ = dir; id_ = id; budget_ = budget_bytes;
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) { if (err) *err = "cannot create snapshot directory " + dir + ": " + strerror(errno); return false; }
    DIR* d = opendir(dir.c_str());
    if (!d) { if (err) *err = "cannot read snapshot directory " + dir; return false; }
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() < 6 || n.compare(n.size() - 6, 6, ".fsnap") != 0) { if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) unlink((dir + "/" + n).c_str()); continue; }   // leftovers of an interrupted write
        const std::string path = dir + "/" + n;
        FILE* f = fopen(path.c_str(), "rb"); if (!f) continue;
        FileHeader h; Entry en; bool ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, kMagic, 8) && h.version == kVersion && h.id == id_ && h.n_tokens > 0 && h.n_tokens < (1u << 24);
        if (ok) { en.toks.resize(h.n_tokens); ok = fread(en.toks.data(), sizeof(int), h.n_tokens, f) == h.n_tokens; }
        fclose(f);
        struct stat st; if (!ok || stat(path.c_str(), &st) != 0) continue;       // another model's / version's snapshot: leave it alone, never load it
        en.path = path; en.mtime = (int64_t)st.st_mtime; en.bytes = (uint64_t)st.st_size; entries_.push_back(std::move(en));
    }
    closedir(d);
    { std::lock_guard<std::mutex> g(m_); enforce_budget_locked(""); }
    th_ = std::thread([this] { writer_loop(); });
    return true;
}

PrefixStore::~PrefixStore() {
    { std::lock_guard<std::mutex> g(m_); stop_ = true; }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

bool PrefixStore::longest(const std::vector<int>& prompt, Entry* out) const {
    std::lock_guard<std::mutex> g(m_);
    const Entry* best = nullptr;
    for (const auto& e : entries_)
        if (e.toks.size() < prompt.size() && (!best || e.toks.size() > best->toks.size()) && std::equal(e.toks.begin(), e.toks.end(), prompt.begin())) best = &e;
    if (best) *out = *best;
    return best != nullptr;
}
bool PrefixStore::has(const std::vector<int>& toks) const {
    std::lock_guard<std::mutex> g(m_);
    for (const auto& e : entries_) if (e.toks == toks) return true;
    for (const auto& j : jobs_) if (j.toks == toks) return true;
    return false;
}
size_t PrefixStore::entries() const { std::lock_guard<std::mutex> g(m_); return entries_.size(); }
uint64_t PrefixStore::bytes() const { std::lock_guard<std::mutex> g(m_); uint64_t b = 0; for (const auto& e : entries_) b += e.bytes; return b; }

void PrefixStore::save_async(const std::vector<int>& toks, std::vector<uint8_t>&& blob) {
    { std::lock_guard<std::mutex> g(m_); if (stop_) return; jobs_.push_back({toks, std::move(blob)}); }
    cv_.notify_one();
}
void PrefixStore::flush() { std::unique_lock<std::mutex> lk(m_); idle_cv_.wait(lk, [&] { return jobs_.empty() && !busy_; }); }

void PrefixStore::writer_loop() {
    while (true) {
        Job j;
        { std::unique_lock<std::mutex> lk(m_); cv_.wait(lk, [&] { return stop_ || !jobs_.empty(); });
          if (jobs_.empty()) { if (stop_) return; continue; }
          j = std::move(jobs_.front()); jobs_.pop_front(); busy_ = true; }
        const std::string path = dir_ + "/" + name_for(id_, j.toks), tmp = path + ".tmp";
        bool ok = false;
        if (budget_ == 0 || j.blob.size() + j.toks.size() * 4 + sizeof(FileHeader) <= budget_) {
            if (FILE* f = fopen(tmp.c_str(), "wb")) {
                FileHeader h; memcpy(h.magic, kMagic, 8); h.version = kVersion; h.id = id_; h.n_tokens = (uint32_t)j.toks.size(); h.payload_bytes = j.blob.size(); h.crc = crc32(j.blob.data(), j.blob.size());
                ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(j.toks.data(), sizeof(int), j.toks.size(), f) == j.toks.size() && fwrite(j.blob.data(), 1, j.blob.size(), f) == j.blob.size();
                ok = (fclose(f) == 0) && ok;
                if (ok) ok = rename(tmp.c_str(), path.c_str()) == 0;
                if (!ok) unlink(tmp.c_str());
            }
        }
        { std::lock_guard<std::mutex> g(m_);
          if (ok) {
              struct stat st; Entry en; en.path = path; en.toks = std::move(j.toks); en.mtime = (int64_t)time(nullptr); en.bytes = stat(path.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
              entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.path == path; }), entries_.end());
              entries_.push_back(std::move(en)); n_saved_++; enforce_budget_locked(path);
          }
          busy_ = false; }
        idle_cv_.notify_all();
    }
}

void PrefixStore::enforce_budget_locked(const std::string& keep) {
    if (budget_ == 0) return;
    uint64_t total = 0; for (const auto& e : entries_) total += e.bytes;
    while (total > budget_) {
        size_t k = entries_.size();
        for (size_t i = 0; i < entries_.size(); ++i) if (entries_[i].path != keep && (k == entries_.size() || entries_[i].mtime < entries_[k].mtime)) k = i;
        if (k == entries_.size()) break;
        total -= entries_[k].bytes; unlink(entries_[k].path.c_str()); entries_.erase(entries_.begin() + (long)k);
    }
}

bool PrefixStore::load(const Entry& e, std::vector<uint8_t>* blob) {
    const std::string path = e.path; bool ok = false;
    if (FILE* f = fopen(path.c_str(), "rb")) {
        FileHeader h;
        if (fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, kMagic, 8) && h.version == kVersion && h.id == id_ && fseek(f, (long)(h.n_tokens * sizeof(int)), SEEK_CUR) == 0) {
            blob->resize((size_t)h.payload_bytes);
            ok = fread(blob->data(), 1, blob->size(), f) == blob->size() && crc32(blob->data(), blob->size()) == h.crc;
        }
        fclose(f);
    }
    if (!ok) {                                                                        // unreadable / corrupt / foreign: forget it
        std::lock_guard<std::mutex> g(m_);
        unlink(path.c_str());
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const Entry& x) { return x.path == path; }), entries_.end());
    }
    return ok;
}
void PrefixStore::touch(const Entry& e) {
    std::lock_guard<std::mutex> g(m_);
    for (auto& x : entries_) if (x.path == e.path) { x.mtime = (int64_t)time(nullptr); utimes(x.path.c_str(), nullptr); }
}

}  // namespace ferro
