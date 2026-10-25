// tokenizer_bpe.cpp — see tokenizer_bpe.h
#include "tokenizer_bpe.h"
#include "../third_party/llama_unicode/unicode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string_view>

namespace ferro {

// GGUF token types (llama.cpp LLAMA_TOKEN_TYPE_*)
enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEFINED = 4, TT_UNUSED = 5, TT_BYTE = 6 };

bool BpeTokenizer::load(const GgufFile& g, std::string* err) {
    const std::string model = g.get_str("tokenizer.ggml.model");
    if (model != "gpt2" && model != "gemma4") { *err = "tokenizer is not BPE (gpt2/gemma4)"; return false; }
    spm_ = (model == "gemma4");
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
    if (spm_) {
        // gemma4: SPM-style BPE — spaces are replaced by U+2581 and BPE runs over the whole text;
        // only newline runs are split out (BPE merge lookup asserts no newlines inside a token).
        regex_ = {"[^\\n]+|[\\n]+"};
        ignore_merges_ = false;
    } else if (pre == "lfm2" || pre == "llama3" || pre == "llama-bpe" || pre == "llama-v3") {
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
    static const bool dbg = getenv("G4_TOKDBG") != nullptr;
    if (dbg) { fprintf(stderr, "[tok] word(%zu bytes) =", word.size()); for (unsigned char c : word) fprintf(stderr, " %02x", c); fprintf(stderr, "\n"); }
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
    if (dbg) { fprintf(stderr, "[tok] after merges: %zu syms:", sym.size()); for (auto& z : sym) if (z.n) fprintf(stderr, " [%.*s]", (int)z.n, word.data()+z.off); fprintf(stderr, "\n"); }
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
        std::string seg = text.substr(f.off, f.len);
        if (spm_) {                                  // escape_whitespaces: ' ' -> U+2581 before splitting
            std::string esc;
            esc.reserve(seg.size() + 8);
            for (char c : seg) {
                if (c == ' ') esc += "\xE2\x96\x81"; else esc += c;
            }
            seg.swap(esc);
            // gemma4 pre-tokenizer is exactly "[^\n]+|[\n]+"; split by hand because the vendored
            // unicode_regex_split re-encodes non-ASCII UTF-8 (it was only ever exercised on the
            // byte-level LFM2 vocabulary). Byte-preserving here.
            size_t i = 0;
            while (i < seg.size()) {
                const bool nl = seg[i] == '\n';
                size_t j = i;
                while (j < seg.size() && (seg[j] == '\n') == nl) ++j;
                bpe_word(seg.substr(i, j - i), out);
                i = j;
            }
            continue;
        }
        auto words = unicode_regex_split(seg, regex_, true);
        for (const auto& w : words) bpe_word(w, out);
    }
    return out;
}

std::string BpeTokenizer::decode_piece(int id) const {
    static int cp2byte[512]; static bool init = false;
    if (!init) {
        for (int& v : cp2byte) v = -1;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            if (direct) cp2byte[b] = b; else cp2byte[256 + n++] = b;
        }
        init = true;
    }
    if (id < 0 || id >= (int)tokens_.size()) return std::string();
    const std::string& p = tokens_[id];
    if (spm_) {                                   // gemma4: U+2581 -> space, raw UTF-8 (no byte-level mapping)
        std::string o;
        o.reserve(p.size());
        for (size_t i = 0; i < p.size();) {
            if (i + 2 < p.size() && (unsigned char)p[i] == 0xE2 && (unsigned char)p[i + 1] == 0x96 &&
                (unsigned char)p[i + 2] == 0x81) { o += ' '; i += 3; }
            else { o += p[i]; ++i; }
        }
        return o;
    }
    std::string out;
    for (size_t i = 0; i < p.size();) {
        const unsigned char c = (unsigned char)p[i];
        int cp, len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6 && i + 1 < p.size()) { cp = ((c & 0x1F) << 6) | (p[i + 1] & 0x3F); len = 2; }
        else if ((c >> 4) == 14 && i + 2 < p.size()) { cp = ((c & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F); len = 3; }
        else { out.append(p, i, 1); ++i; continue; }
        if (cp < 512 && cp2byte[cp] >= 0) out += (char)cp2byte[cp];
        else out.append(p, i, (size_t)len);                       // not a byte-level character: keep as is
        i += (size_t)len;
    }
    return out;
}

}  // namespace ferro
