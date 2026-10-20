// tokenizer_bpe.h — byte-level (GPT-2 style) BPE tokenizer for GGUF "gpt2" vocabularies
// (LFM2 / llama3 pre-tokenizer). Matches llama.cpp's llm_tokenizer_bpe_session, including
// special-token partitioning (CONTROL tokens when parse_special) and ignore_merges.
#pragma once
#include "gguf.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace ferro {

class BpeTokenizer {
public:
    bool load(const GgufFile& g, std::string* err);

    // text -> ids. parse_special: CONTROL tokens written in the text become their ids
    // (USER_DEFINED tokens are always matched, as in llama.cpp). No BOS/EOS added.
    std::vector<int> tokenize(const std::string& text, bool parse_special) const;

    int token_id(const std::string& piece) const;   // -1 if absent
    const std::string& piece(int id) const { return tokens_[id]; }
    int n_vocab() const { return (int)tokens_.size(); }
    int mask_id() const { return mask_id_; }
    int bos_id() const { return bos_id_; }

private:
    void bpe_word(const std::string& word, std::vector<int>& out) const;
    int rank(const std::string& a, const std::string& b) const;

    std::vector<std::string> tokens_;
    std::vector<int> types_;
    std::unordered_map<std::string, int> tok2id_;
    std::unordered_map<std::string, int> merge_rank_;   // key: a + '\x01' + b
    std::vector<int> specials_;                         // CONTROL / USER_DEFINED, longest first
    std::vector<std::string> regex_;
    bool ignore_merges_ = true;
    int mask_id_ = -1, bos_id_ = -1;
};

}  // namespace ferro
