// Minimal native tokenizer for Gemma-family GGUF (BPE with merges, SPM-style spaces).
#pragma once
#include "gguf.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace ferro {

class FerroTokenizer {
public:
    bool load(const GgufFile& g, std::string* err);

    // Plain text -> token ids. Adds BOS/EOS according to GGUF add_bos/add_eos flags.
    std::vector<int> tokenize(const std::string& text) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

private:
    struct Pair { std::string a, b; };
    struct PairHash { size_t operator()(const std::pair<std::string,std::string>& p) const {
        return std::hash<std::string>()(p.first) * 1315423911u ^ std::hash<std::string>()(p.second); } };

    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> byte_to_id_;  // "<0xNN>" pieces
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks_;
    int bos_id_ = -1, eos_id_ = -1, unk_id_ = -1;
    bool add_bos_ = false, add_eos_ = false;
    static constexpr const char* SPACE_PIECE = "\u2581"; // ▁
};

} // namespace ferro
