// gguf.cpp
#include "gguf.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace ferro {

namespace {
struct Reader {
    const uint8_t* p; const uint8_t* end; bool ok = true;
    uint64_t u64() { need(8); uint64_t v; memcpy(&v, p, 8); p += 8; return v; }
    uint32_t u32() { need(4); uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
    std::string str() {
        uint64_t n = u64(); need(n);
        std::string s((const char*)p, n); p += n; return s;
    }
    void need(size_t n) { if (p + n > end) ok = false; }
};
} // namespace

bool GgufFile::open_file(const char* path, std::string* err) {
    open_fd_ = ::open(path, O_RDONLY);
    if (open_fd_ < 0) { *err = std::string("open failed: ") + path; return false; }
    struct stat st; fstat(open_fd_, &st); size_ = st.st_size;
    base_ = (uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, open_fd_, 0);
    if (base_ == MAP_FAILED) { *err = "mmap failed"; return false; }
    // advise read-ahead for the whole file (weights stream at load time)
    madvise(base_, size_, MADV_WILLNEED);

    Reader r{base_, base_ + size_};
    if (memcmp(r.p, "GGUF", 4) != 0) { *err = "bad magic"; return false; }
    r.p += 4;
    r.u32();                       // version
    uint64_t n_tensors = r.u64();
    uint64_t n_kv = r.u64();

    for (uint64_t i = 0; i < n_kv && r.ok; ++i) {
        std::string k = r.str();
        uint32_t t = r.u32();
        Val v; v.type = t;
        if (t == 8) {
            v.s = r.str();
        } else if (t == 9) {
            uint32_t et = r.u32();
            uint64_t n = r.u64();
            if (et == 8) {
                // string array: serialize as \n-separated (tokens have no \n? may!) -> use length-prefixed
                std::string joined;
                for (uint64_t j = 0; j < n && r.ok; ++j) {
                    std::string s = r.str();
                    joined += std::to_string(s.size()); joined += '|'; joined += s;
                }
                v.type = 100; v.raw.assign(joined.begin(), joined.end());
            } else {
                static const size_t esz[] = {1,1,2,2,4,4,4,1,0,0,8};
                size_t es = esz[et];
                r.need(n * es);
                v.type = 100 + et;
                v.raw.assign(r.p, r.p + n * es);
                r.p += n * es;
            }
        } else {
            static const size_t esz[] = {1,1,2,2,4,4,4,1,0,0,8};
            size_t es = esz[t];
            r.need(es);
            v.raw.assign(r.p, r.p + es);
            r.p += es;
        }
        kv.emplace(std::move(k), std::move(v));
    }

    for (uint64_t i = 0; i < n_tensors && r.ok; ++i) {
        GGUFTensor t;
        t.name = r.str();
        uint32_t nd = r.u32();
        t.dims.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) t.dims[d] = r.u64();
        t.type = r.u32();
        t.off = r.u64();
        tensor_list_.push_back(t);
        tensors_.emplace(t.name, t);
    }
    if (!r.ok) { *err = "truncated"; return false; }

    uint32_t align = (uint32_t)get_int("general.alignment", 32);
    uint64_t cur = (uint64_t)(r.p - base_);
    cur = (cur + align - 1) / align * align;
    data_start_ = base_ + cur;
    arch = get_str("general.architecture");
    return true;
}

void GgufFile::close() {
    if (base_) { munmap(base_, size_); base_ = nullptr; }
    if (open_fd_ >= 0) { ::close(open_fd_); open_fd_ = -1; }
}

// --- accessors ---
std::string GgufFile::get_str(const std::string& k, const std::string& def) const {
    auto it = kv.find(k);
    if (it == kv.end()) return def;
    if (it->second.type == 8) return it->second.s;
    return def;
}
static int64_t rd_int(const std::vector<uint8_t>& raw) {
    int64_t v = 0; size_t n = raw.size() < 8 ? raw.size() : 8;
    memcpy(&v, raw.data(), n);
    if (raw.size() == 1) { int8_t b = raw[0]; return b; }   // bool/i8
    if (raw.size() == 4) { int32_t b; memcpy(&b, raw.data(), 4); return b; }
    return v;
}
int64_t GgufFile::get_int(const std::string& k, int64_t def) const {
    auto it = kv.find(k); if (it == kv.end()) return def;
    return rd_int(it->second.raw);
}
double GgufFile::get_f64(const std::string& k, double def) const {
    auto it = kv.find(k); if (it == kv.end()) return def;
    const auto& raw = it->second.raw;
    if (raw.size() == 4) { float f; memcpy(&f, raw.data(), 4); return f; }
    if (raw.size() == 8) { double d; memcpy(&d, raw.data(), 8); return d; }
    return def;
}
bool GgufFile::get_bool(const std::string& k, bool def) const {
    auto it = kv.find(k); if (it == kv.end()) return def;
    return !it->second.raw.empty() && it->second.raw[0] != 0;
}
std::vector<std::string> GgufFile::get_strarr(const std::string& k) const {
    std::vector<std::string> out;
    auto it = kv.find(k); if (it == kv.end() || it->second.type != 100) return out;
    const char* p = (const char*)it->second.raw.data();
    const char* e = p + it->second.raw.size();
    while (p < e) {
        const char* bar = (const char*)memchr(p, '|', e - p);
        if (!bar) break;
        size_t n = strtoull(std::string(p, bar).c_str(), nullptr, 10);
        p = bar + 1;
        out.emplace_back(p, n); p += n;
    }
    return out;
}
std::vector<double> GgufFile::get_f64arr(const std::string& k) const {
    std::vector<double> out;
    auto it = kv.find(k); if (it == kv.end() || it->second.type < 100) return out;
    uint32_t et = it->second.type - 100;
    const auto& raw = it->second.raw;
    if (et == 6) { size_t n = raw.size()/4; out.resize(n);
        for (size_t i = 0; i < n; ++i) { float f; memcpy(&f, raw.data()+i*4, 4); out[i]=f; } }
    else if (et == 10) { size_t n = raw.size()/8; out.resize(n);
        for (size_t i = 0; i < n; ++i) memcpy(&out[i], raw.data()+i*8, 8); }
    return out;
}
std::vector<int64_t> GgufFile::get_iarr(const std::string& k) const {
    std::vector<int64_t> out;
    auto it = kv.find(k); if (it == kv.end() || it->second.type < 100) return out;
    uint32_t et = it->second.type - 100;
    const auto& raw = it->second.raw;
    if (et == 5 || et == 4) { size_t n = raw.size()/4; out.resize(n);
        for (size_t i = 0; i < n; ++i) { int32_t v; memcpy(&v, raw.data()+i*4, 4); out[i]=v; } }
    else if (et == 2 || et == 3) { size_t n = raw.size()/2; out.resize(n);
        for (size_t i = 0; i < n; ++i) { int16_t v; memcpy(&v, raw.data()+i*2, 2); out[i]=v; } }
    else if (et == 0 || et == 1) { size_t n = raw.size(); out.resize(n);
        for (size_t i = 0; i < n; ++i) out[i] = (int8_t)raw[i]; }
    return out;
}

} // namespace ferro
