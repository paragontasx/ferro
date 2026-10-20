// gguf.h — minimal zero-copy GGUF reader (metadata + tensor table)
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace ferro {

struct GGUFTensor {
    std::string name;
    std::vector<uint64_t> dims;   // ggml order: ne[0] fastest
    uint32_t type;                // GgType
    uint64_t off;                 // from data_start
};

class GgufFile {
public:
    bool open_file(const char* path, std::string* err);
    void close();

    // metadata accessors
    bool has(const std::string& k) const { return kv.count(k) != 0; }
    const GGUFTensor* tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const uint8_t* tensor_data(const GGUFTensor& t) const { return data_start_ + t.off; }

    std::string get_str(const std::string& k, const std::string& def = "") const;
    int64_t get_int(const std::string& k, int64_t def = 0) const;
    double get_f64(const std::string& k, double def = 0) const;
    bool get_bool(const std::string& k, bool def = false) const;
    std::vector<std::string> get_strarr(const std::string& k) const;
    std::vector<double> get_f64arr(const std::string& k) const;   // f32 arrays widened
    std::vector<int64_t> get_iarr(const std::string& k) const;

    const std::vector<GGUFTensor>& all_tensors() const { return tensor_list_; }
    std::vector<std::string> keys() const { std::vector<std::string> k; for (auto& p : kv) k.push_back(p.first); return k; }
    size_t file_size() const { return size_; }
    std::string arch;               // general.architecture

private:
    struct Val { uint32_t type = 0; std::string s; std::vector<uint8_t> raw; };
    int open_fd_ = -1;
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
    const uint8_t* data_start_ = nullptr;
    std::map<std::string, Val> kv;
    std::map<std::string, GGUFTensor> tensors_;
    std::vector<GGUFTensor> tensor_list_;
};

} // namespace ferro
