#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace ferro {

static std::vector<std::string> utf8_codepoints(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        size_t len = 1;
        if (c >= 0xF0) len = 4; else if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
        if (i + len > s.size()) len = 1;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

bool FerroTokenizer::load(const GgufFile& g, std::string* err) {
    auto toks = g.get_strarr("tokenizer.ggml.tokens");
    if (toks.empty()) { *err = "no tokenizer.ggml.tokens"; return false; }
    token_to_id_.reserve(toks.size() * 2);
    for (size_t i = 0; i < toks.size(); ++i)
        token_to_id_.emplace(toks[i], (int)i);   // first occurrence wins (llama.cpp behavior)
    for (int b = 0; b < 256; ++b) {
        char buf[8]; snprintf(buf, sizeof buf, "<0x%02X>", b);
        auto it = token_to_id_.find(buf);
        if (it != token_to_id_.end()) byte_to_id_[buf] = it->second;
    }
    auto merges = g.get_strarr("tokenizer.ggml.merges");
    bpe_ranks_.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); ++i) {
        const std::string& w = merges[i];
        size_t pos = w.find(' ', 1);
        if (pos == std::string::npos) continue;
        bpe_ranks_.emplace(std::make_pair(w.substr(0, pos), w.substr(pos + 1)), (int)i);
    }
    bos_id_ = (int)g.get_int("tokenizer.ggml.bos_token_id", -1);
    eos_id_ = (int)g.get_int("tokenizer.ggml.eos_token_id", -1);
    unk_id_ = (int)g.get_int("tokenizer.ggml.unknown_token_id", 0);
    add_bos_ = g.get_int("tokenizer.ggml.add_bos_token", 0) != 0;
    add_eos_ = g.get_int("tokenizer.ggml.add_eos_token", 0) != 0;
    return true;
}

static void emit_symbol(const std::string& sym,
                        const std::unordered_map<std::string,int>& tok2id,
                        const std::unordered_map<std::string,int>& byte2id,
                        int unk_id, std::vector<int>& out) {
    auto it = tok2id.find(sym);
    if (it != tok2id.end()) { out.push_back(it->second); return; }
    // byte fallback: <0xNN> per UTF-8 byte
    for (unsigned char c : sym) {
        char buf[8]; snprintf(buf, sizeof buf, "<0x%02X>", c);
        auto b = byte2id.find(buf);
        out.push_back(b != byte2id.end() ? b->second : unk_id);
    }
}

std::vector<int> FerroTokenizer::tokenize(const std::string& text) const {
    std::vector<int> ids;
    if (add_bos_ && bos_id_ >= 0) ids.push_back(bos_id_);

    // SPM-style: spaces -> U+2581 ; split on newline runs (gemma4 pre-tokenizer); BPE per chunk.
    std::string norm;
    norm.reserve(text.size() + 8);
    for (char c : text) {
        if (c == ' ') { norm += "\u2581"; }
        else norm.push_back(c);
    }

    size_t i = 0;
    while (i < norm.size()) {
        size_t j = i;
        while (j < norm.size() && norm[j] != '\n') ++j;
        std::string chunk = norm.substr(i, j - i);
        if (!chunk.empty()) {
            std::vector<std::string> sym = utf8_codepoints(chunk);
            while (true) {
                int best_rank = INT32_MAX; size_t best = sym.size();
                for (size_t k = 0; k + 1 < sym.size(); ++k) {
                    auto it = bpe_ranks_.find({sym[k], sym[k + 1]});
                    if (it != bpe_ranks_.end() && it->second < best_rank) { best_rank = it->second; best = k; }
                }
                if (best == sym.size()) break;
                std::vector<std::string> merged;
                merged.reserve(sym.size());
                for (size_t k = 0; k < sym.size();) {
                    if (k + 1 < sym.size() &&
                        bpe_ranks_.count({sym[k], sym[k + 1]}) &&
                        bpe_ranks_.at({sym[k], sym[k + 1]}) == best_rank) {
                        merged.push_back(sym[k] + sym[k + 1]);
                        k += 2;
                    } else merged.push_back(sym[k++]);
                }
                sym.swap(merged);
            }
            for (const auto& s : sym) emit_symbol(s, token_to_id_, byte_to_id_, unk_id_, ids);
        }
        if (j < norm.size()) {   // newline run: tokenize as its own chunk
            size_t k = j;
            while (k < norm.size() && norm[k] == '\n') ++k;
            std::string nl = norm.substr(j, k - j);
            emit_symbol(nl, token_to_id_, byte_to_id_, unk_id_, ids);
            j = k;
        }
        i = j;
    }

    if (add_eos_ && eos_id_ >= 0) ids.push_back(eos_id_);
    return ids;
}

} // namespace ferro
