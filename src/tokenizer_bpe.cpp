// tokenizer_bpe.cpp — see tokenizer_bpe.h
#include "tokenizer_bpe.h"
#include "../third_party/llama_unicode/unicode.h"

#include <algorithm>
#include <queue>
#include <string_view>

namespace ferro {

// GGUF token types (llama.cpp LLAMA_TOKEN_TYPE_*)
enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEFINED = 4, TT_UNUSED = 5, TT_BYTE = 6 };

bool BpeTokenizer::load(const GgufFile& g, std::string* err) {
    if (g.get_str("tokenizer.ggml.model") != "gpt2") { *err = "tokenizer is not byte-level BPE (gpt2)"; return false; }
    tokens_ = g.get_strarr("tokenizer.ggml.tokens");
    auto ty = g.get_iarr("tokenizer.ggml.token_type");
    types_.assign(ty.begin(), ty.end());
    if (types_.size() != tokens_.size()) types_.assign(tokens_.size(), TT_NORMAL);
    tok2id_.reserve(tokens_.size() * 2);
    for (int i = 0; i < (int)tokens_.size(); ++i) tok2id_.emplace(tokens_[i], i);

    auto merges = g.get_strarr("tokenizer.ggml.merges");
    merge_rank_.reserve(merges.size() * 2);
    for (int r = 0; r < (int)merges.size(); ++r) {
        const std::string& m = merges[r];
        // "a b" — split at the first space after position 0 (as llama.cpp does)
        size_t sp = m.find(' ', 1);
        if (sp == std::string::npos) continue;
        merge_rank_.emplace(m.substr(0, sp) + '\x01' + m.substr(sp + 1), r);
    }

    const std::string pre = g.get_str("tokenizer.ggml.pre");
    if (pre == "lfm2" || pre == "llama3" || pre == "llama-bpe" || pre == "llama-v3") {
        regex_ = {"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"};
        ignore_merges_ = true;
    } else {
        *err = "unsupported BPE pre-tokenizer: " + pre;
        return false;
    }

    for (int i = 0; i < (int)tokens_.size(); ++i)
        if (types_[i] == TT_CONTROL || types_[i] == TT_USER_DEFINED || types_[i] == TT_UNKNOWN) specials_.push_back(i);
    std::stable_sort(specials_.begin(), specials_.end(),
                     [&](int a, int b) { return tokens_[a].size() > tokens_[b].size(); });

    mask_id_ = (int)g.get_int("tokenizer.ggml.mask_token_id", -1);
    bos_id_  = (int)g.get_int("tokenizer.ggml.bos_token_id", -1);
    return true;
}

int BpeTokenizer::token_id(const std::string& p) const {
    auto it = tok2id_.find(p);
    return it == tok2id_.end() ? -1 : it->second;
}

int BpeTokenizer::rank(const std::string& a, const std::string& b) const {
    auto it = merge_rank_.find(a + '\x01' + b);
    return it == merge_rank_.end() ? -1 : it->second;
}

// llm_tokenizer_bpe_session::tokenize for one pre-tokenized (byte-encoded) word
void BpeTokenizer::bpe_word(const std::string& word, std::vector<int>& out) const {
    if (ignore_merges_) {
        auto it = tok2id_.find(word);
        if (it != tok2id_.end()) { out.push_back(it->second); return; }
    }
    struct Sym { int prev, next; size_t off, n; };
    std::vector<Sym> sym;
    for (size_t off = 0; off < word.size();) {
        size_t n = std::min(word.size() - off, (size_t)unicode_len_utf8(word[off]));
        int idx = (int)sym.size();
        sym.push_back({idx - 1, off + n == word.size() ? -1 : idx + 1, off, n});
        off += n;
    }
    struct Bi { int left, right; std::string text; int rank; };
    auto cmp = [](const Bi& l, const Bi& r) { return l.rank > r.rank || (l.rank == r.rank && l.left > r.left); };
    std::priority_queue<Bi, std::vector<Bi>, decltype(cmp)> q(cmp);
    auto add = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        std::string a = word.substr(sym[l].off, sym[l].n), b = word.substr(sym[r].off, sym[r].n);
        int rk = rank(a, b);
        if (rk < 0) return;
        q.push({l, r, a + b, rk});
    };
    for (int i = 1; i < (int)sym.size(); ++i) add(i - 1, i);
    while (!q.empty()) {
        Bi bg = q.top(); q.pop();
        Sym& L = sym[bg.left]; Sym& R = sym[bg.right];
        if (L.n == 0 || R.n == 0) continue;
        if (word.compare(L.off, L.n, bg.text, 0, L.n) != 0 || L.n + R.n != bg.text.size() ||
            word.compare(R.off, R.n, bg.text, L.n, R.n) != 0) continue;   // outdated
        L.n += R.n; R.n = 0;
        L.next = R.next;
        if (R.next >= 0) sym[R.next].prev = bg.left;
        add(L.prev, bg.left);
        add(bg.left, L.next);
    }
    for (auto& s : sym) {
        if (!s.n) continue;
        std::string str = word.substr(s.off, s.n);
        auto it = tok2id_.find(str);
        if (it != tok2id_.end()) { out.push_back(it->second); continue; }
        for (char c : str) {           // byte fallback (byte_encode: single raw char lookup)
            auto jt = tok2id_.find(std::string(1, c));
            if (jt != tok2id_.end()) out.push_back(jt->second);
        }
    }
}

std::vector<int> BpeTokenizer::tokenize(const std::string& text, bool parse_special) const {
    // fragment list: (special id) or raw [off, len)
    struct Frag { int id; size_t off, len; };
    std::vector<Frag> frags{{-1, 0, text.size()}};
    if (text.empty()) frags.clear();
    for (int sid : specials_) {
        if (!parse_special && (types_[sid] == TT_CONTROL || types_[sid] == TT_UNKNOWN)) continue;
        const std::string& st = tokens_[sid];
        if (st.empty()) continue;
        std::vector<Frag> nf;
        nf.reserve(frags.size());
        for (const Frag& f : frags) {
            if (f.id >= 0) { nf.push_back(f); continue; }
            size_t base = f.off, end = f.off + f.len;
            while (true) {
                size_t m = std::string_view(text.data(), end).find(st, base);
                if (m == std::string::npos) break;
                if (m > base) nf.push_back({-1, base, m - base});
                nf.push_back({sid, 0, 0});
                base = m + st.size();
            }
            if (base < end) nf.push_back({-1, base, end - base});
        }
        frags.swap(nf);
    }
    std::vector<int> out;
    for (const Frag& f : frags) {
        if (f.id >= 0) { out.push_back(f.id); continue; }
        auto words = unicode_regex_split(text.substr(f.off, f.len), regex_, true);
        for (const auto& w : words) bpe_word(w, out);
    }
    return out;
}

}  // namespace ferro
