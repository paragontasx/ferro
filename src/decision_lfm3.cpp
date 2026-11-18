// decision_lfm3.cpp — port of the LFM3_D1 path of llama.cpp tools/server/server-decision.cpp:
// request parsing (:288-360), d1_break / decision_escape (:606-660), d1_labels (:1011-1070), render (:1157)
// and format_answer (:1640-1745). The d1-2B "systemone" template is hand-rendered: it is a fixed scaffold
// around the three question types with no control flow beyond type/none checks.
#include "decision_lfm3.h"
#include "json.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace ferro {

static const char* type_name(DecisionType t) { return t == DQ_NOUL ? "noul" : t == DQ_SCORE ? "score" : "choice"; }

// ---- model constants -------------------------------------------------------------

bool lfm3d1_load(Lfm3D1Ctx& c, const GgufFile& g, const BpeTokenizer& tok, std::string* err) {
    (void)err;
    c.tok = &tok;
    for (const char* t : { "noul", "choice", "score" })
        for (const char* b : { "", ".2", ".3_5", ".6_10", ".11" }) {
            double v = g.get_f64(std::string("lfm3.decision.temperature.") + t + b, -1.0);
            if (v > 0.0) c.temps[std::string(t) + b] = (float)v;
        }
    for (int n = 2; n <= 10; ++n) {
        std::vector<double> bias;
        for (int lv = 0; lv < n; ++lv) {
            double v = g.get_f64("lfm3.decision.score_bias." + std::to_string(n) + "." + std::to_string(lv), 0.0);
            if (v != 0.0) { bias.resize(n, 0.0); bias[lv] = v; }
        }
        if (!bias.empty()) c.score_bias[n] = bias;
    }
    auto toks = g.get_strarr("tokenizer.ggml.tokens");
    auto typs = g.get_iarr("tokenizer.ggml.token_type");
    for (size_t i = 0; i < toks.size() && i < typs.size(); ++i)
        if (typs[i] == 3 && toks[i].size() >= 2) {                                    // CONTROL, min 2 bytes (init_quoted_spellings)
            c.quoted.push_back(toks[i]);
            if (c.quoted_first.find(toks[i][0]) == std::string::npos) c.quoted_first += toks[i][0];
        }
    std::sort(c.quoted.begin(), c.quoted.end(), [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    return true;
}

// ---- d1_break: special-token spellings in user text must not survive tokenization ----

static std::string d1_break(const Lfm3D1Ctx& c, const std::string& text) {
    static const std::string zwsp = "\xE2\x80\x8B";
    std::string cur = text;
    for (bool changed = true; changed;) {
        changed = false;
        std::string out;
        for (size_t i = 0; i < cur.size();) {
            const std::string* hit = nullptr;
            if (c.quoted_first.find(cur[i]) != std::string::npos)
                for (const auto& s : c.quoted)
                    if (cur.compare(i, s.size(), s) == 0) { hit = &s; break; }
            if (!hit) { out += cur[i++]; continue; }
            const unsigned char ch = (*hit)[0];                                        // insert the ZWSP after the first UTF-8 character
            const size_t nb = ch < 0x80 ? 1 : ch < 0xE0 ? 2 : ch < 0xF0 ? 3 : 4;
            out += hit->substr(0, nb) + zwsp + hit->substr(nb);
            i += hit->size();
            changed = true;
        }
        cur = std::move(out);
    }
    return cur;
}

static Json escape_json(const Lfm3D1Ctx& c, const Json& v, int depth) {
    if (depth >= 64) throw std::invalid_argument("JSON is nested more than 64 levels deep");
    if (v.t == Json::String) return Json::str(d1_break(c, v.s));
    if (v.t == Json::Array) { Json o = Json::array(); for (const auto& x : v.arr) o.push(escape_json(c, x, depth + 1)); return o; }
    if (v.t == Json::Object) { Json o = Json::object(); for (const auto& kv : v.obj) o.set(d1_break(c, kv.first), escape_json(c, kv.second, depth + 1)); return o; }
    return v;
}

// ---- jinja rendering helpers ------------------------------------------------------

// minja `tojson(indent=2)` = nlohmann dump(2)
static void pretty2(const Json& v, int ind, std::string& out) {
    auto pad = [&](int n) { out += std::string(2 * n, ' '); };
    switch (v.t) {
        case Json::Null: out += "null"; break;
        case Json::Bool: out += v.b ? "true" : "false"; break;
        case Json::Int: out += std::to_string(v.i); break;
        case Json::Float: case Json::String: out += json_to_jinja(v); break;
        case Json::Array: {
            if (v.arr.empty()) { out += "[]"; break; }
            out += "[\n";
            for (size_t i = 0; i < v.arr.size(); ++i) { pad(ind + 1); pretty2(v.arr[i], ind + 1, out); out += i + 1 < v.arr.size() ? ",\n" : "\n"; }
            pad(ind); out += "]";
            break;
        }
        case Json::Object: {
            if (v.obj.empty()) { out += "{}"; break; }
            out += "{\n";
            for (size_t i = 0; i < v.obj.size(); ++i) {
                pad(ind + 1); out += json_to_jinja(Json::str(v.obj[i].first)) + ": ";
                pretty2(v.obj[i].second, ind + 1, out);
                out += i + 1 < v.obj.size() ? ",\n" : "\n";
            }
            pad(ind); out += "}";
            break;
        }
    }
}

static std::string render_state(const Json& s) {
    if (s.t == Json::String) return s.s;
    std::string out; pretty2(s, 0, out); return out;
}

// jinja truthiness of an option description
static bool desc_truthy(const Json& d) {
    switch (d.t) {
        case Json::Null: return false;
        case Json::Bool: return d.b;
        case Json::Int: return d.i != 0;
        case Json::Float: return d.d != 0;
        case Json::String: return !d.s.empty();
        case Json::Array: return !d.arr.empty();
        case Json::Object: return !d.obj.empty();
    }
    return false;
}

static std::string render_desc_or_json(const Json& d) { return d.t == Json::String ? d.s : json_to_jinja(d); }

// ---- labels (d1_labels) -----------------------------------------------------------

static std::vector<int> single_tokens(const Lfm3D1Ctx& c, const std::vector<std::string>& forms) {
    std::vector<int> out;
    for (const auto& f : forms) {
        auto t = c.tok->tokenize(f, false);
        if (t.size() == 1 && std::find(out.begin(), out.end(), t[0]) == out.end()) out.push_back(t[0]);
    }
    return out;
}

// ---- request parsing + rendering --------------------------------------------------

std::vector<Lfm3D1Built> lfm3d1_build(const Lfm3D1Ctx& c, const Json& body, const std::vector<int>& per_entry_tokens) {
    if (!body.is_object()) throw std::invalid_argument("the request body must be a JSON object");
    static const Json null_json;
    // llama.cpp renders {{ image }} as a fixed random media marker and replaces user occurrences with " "
    // (decision_replace_text) so the prompt split at the markers is unambiguous
    static std::atomic<uint64_t> mctr{ 0 };
    auto hexrand = [] {
        uint64_t x = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count() ^ (0x9E3779B97F4A7C15ull * (++mctr + 1));
        x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33;
        char buf[17]; snprintf(buf, sizeof buf, "%016llx", (unsigned long long)x);
        return std::string(buf);
    };
    static const std::string marker = "<__media_" + hexrand() + "__>";
    auto strip = [&](std::string s) {
        for (size_t m = s.find(marker); m != std::string::npos; m = s.find(marker, m + 1)) s.replace(m, marker.size(), " ");
        return s;
    };
    const Json* state = body.find("state");
    const Json& state_e = state ? escape_json(c, *state, 0) : null_json;
    const Json* questions = body.find("questions");
    if (!questions || !questions->is_object()) throw std::invalid_argument("\"questions\" must be an object");

    // nlohmann::json iterates objects with keys sorted: so do we (option and answer order depend on it)
    std::vector<std::pair<std::string, const Json*>> qs;
    for (const auto& kv : questions->obj) qs.push_back({ kv.first, &kv.second });
    std::sort(qs.begin(), qs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<Lfm3D1Built> out;
    for (const auto& qv : qs) {
        const Json& q = *qv.second;
        if (!q.is_object()) throw std::invalid_argument("question \"" + qv.first + "\" must be an object");
        const Json* jt = q.find("type");
        const Json* jins = q.find("instructions");
        const Json* jc = q.find("criteria");
        Lfm3D1Built b;
        b.id = qv.first;
        if (!jt || !jt->is_string()) throw std::invalid_argument("\"type\" must be one of: choice, score, noul");
        if (jt->s == "choice") b.type = DQ_CHOICE;
        else if (jt->s == "score") b.type = DQ_SCORE;
        else if (jt->s == "noul") b.type = DQ_NOUL;
        else throw std::invalid_argument("\"type\" must be one of: choice, score, noul");

        const Json ins = jins ? *jins : null_json;
        const Json crit = jc ? *jc : null_json;

        // options in model order, like parse_question
        std::vector<std::pair<std::string, Json>> opts;
        if (b.type == DQ_CHOICE) {
            if (!crit.is_object() || crit.obj.empty()) throw std::invalid_argument("\"criteria\" must be a non-empty object");
            for (const auto& kv : crit.obj) opts.push_back({ kv.first, kv.second });
            std::sort(opts.begin(), opts.end(), [](const auto& a, const auto& z) { return a.first < z.first; });
            if (opts.size() > 255) throw std::invalid_argument("too many options, the maximum is 255");
        } else if (b.type == DQ_SCORE) {
            if (!crit.is_array() || crit.arr.size() < 2 || crit.arr.size() > 10) throw std::invalid_argument("\"criteria\" must be an array of 2 to 10 levels");
            for (size_t i = 0; i < crit.arr.size(); ++i) opts.push_back({ std::to_string(i), crit.arr[i] });
        } else {
            if (!crit.is_null() && !crit.is_object()) throw std::invalid_argument("\"criteria\" must be an object");
            Json dtrue, dfalse;
            if (crit.is_object()) {
                if (const Json* p = crit.find("true")) dtrue = *p;
                if (const Json* p = crit.find("false")) dfalse = *p;
            }
            opts.push_back({ "true", dtrue });                                                          // true first (noul_true_first)
            opts.push_back({ "false", dfalse });
        }
        const size_t n = opts.size();

        // the escaped texts the template sees
        std::vector<std::string> keys_esc(n);
        std::vector<Json> descs_esc(n);
        for (size_t i = 0; i < n; ++i) { keys_esc[i] = d1_break(c, opts[i].first); descs_esc[i] = escape_json(c, opts[i].second, 0); }

        // ---- labels before rendering: choice shows its assigned code in the prompt ----
        std::vector<std::string> codes(n);
        std::vector<std::vector<int>> groups(n);
        if (b.type != DQ_CHOICE) {
            for (size_t i = 0; i < n; ++i) {
                groups[i] = b.type == DQ_SCORE ? single_tokens(c, { opts[i].first })
                                               : single_tokens(c, opts[i].first == "true" ? std::vector<std::string>{ "yes", "Yes", "YES" }
                                                                                          : std::vector<std::string>{ "no", "No", "NO" });
                if (groups[i].empty()) throw std::invalid_argument("decision label is not a single token: " + opts[i].first);
                codes[i] = opts[i].first;
            }
        } else {
            bool is_letters = true;
            for (const auto& o : opts) is_letters = is_letters && o.first.size() == 1 && std::isalpha((unsigned char)o.first[0]);
            std::vector<std::string> cand(n);
            for (size_t i = 0; i < n; ++i)
                cand[i] = is_letters ? opts[i].first
                        : n <= 26 ? std::string(1, 'A' + i)
                        : i < 26 ? std::string(1, 'A' + i) : std::string{ (char)('A' + (i - 26) / 26), (char)('A' + (i - 26) % 26) };
            std::vector<std::string> pool;
            for (char ch = 'A'; ch <= 'Z'; ++ch) pool.push_back(std::string(1, ch));
            for (int i = 0; i < 100; ++i) { char t[8]; snprintf(t, sizeof t, "%02d", i); pool.push_back(t); }
            for (char ch = 'a'; ch <= 'z'; ++ch) pool.push_back(std::string(1, ch));
            for (int i = 0; i < 200; ++i) { char t[8]; snprintf(t, sizeof t, "#%d", i); pool.push_back(t); }
            for (char a = 'A'; a <= 'Z'; ++a) for (char z = 'A'; z <= 'Z'; ++z) pool.push_back(std::string{ a, z });
            std::vector<int> used;
            auto take = [&](const std::string& code, size_t i) {
                auto toks = c.tok->tokenize(code, false);
                if (toks.size() != 1 || std::find(used.begin(), used.end(), toks[0]) != used.end()) return false;
                used.push_back(toks[0]);
                groups[i] = { toks[0] };
                for (int t : single_tokens(c, { " " + code })) if (t != toks[0]) groups[i].push_back(t);
                codes[i] = code;
                return true;
            };
            for (size_t i = 0; i < n; ++i) {
                if (take(cand[i], i)) continue;
                bool got = false;
                for (size_t k = 0; !got && k < pool.size(); ++k) got = take(pool[k], i);
                if (!got) throw std::invalid_argument("no single-token label left for " + std::to_string(n) + " options");
            }
        }

        // ---- render the systemone template for this one question ----
        std::string p = "<|startoftext|><|im_start|>user\n";
        for (size_t i = 0; i < per_entry_tokens.size(); ++i) p += marker;              // {{ image }} per entry, back to back
        if (!state_e.is_null()) { p += strip(render_state(state_e)); p += "\n\n\nQUESTION:\n"; }
        const Json ins_e = escape_json(c, ins, 0);
        p += strip(ins_e.t == Json::String ? ins_e.s : json_to_jinja(ins_e));

        if (b.type == DQ_CHOICE) {
            p += "\n\nOptions:\n";
            for (size_t i = 0; i < n; ++i) {
                p += codes[i];
                p += " ";
                if (desc_truthy(descs_esc[i])) p += strip(render_desc_or_json(descs_esc[i]));
                else { std::string k = strip(keys_esc[i]); std::replace(k.begin(), k.end(), '_', ' '); p += k; }
                if (i + 1 < n) p += "\n";
            }
            p += "\n\nReply with the option code only.";
        } else if (b.type == DQ_NOUL) {
            bool any = false;
            for (const auto& d : descs_esc) if (!d.is_null()) { any = true; break; }
            if (any)
                for (size_t i = 0; i < n; ++i) {
                    p += opts[i].first == "true" ? "\nYes: " : "\nNo: ";
                    if (descs_esc[i].is_null()) p += "None";
                    else p += strip(render_desc_or_json(descs_esc[i]));
                }
            p += "\n\nReply with yes or no only.";
        } else {
            p += "\n\n";
            for (size_t i = 0; i < n; ++i) p += opts[i].first + " " + strip(render_desc_or_json(descs_esc[i])) + "\n";
            p += "\nReply with a single digit 0-" + std::to_string(n - 1) + " only.";
        }
        p += "<|im_end|>\n<|im_start|>assistant\n<think></think>";

        for (size_t i = 0; i < n; ++i) { b.keys.push_back(opts[i].first); b.descriptions.push_back(descs_esc[i]); }
        b.codes = std::move(codes);
        b.groups = std::move(groups);
        if (per_entry_tokens.empty()) {
            b.tokens = c.tok->tokenize(p, true);
        } else {
            // mtmd piece tokenization: the text pieces around the media markers are tokenized independently
            // (add_special=false per piece; this vocab has add_bos/add_eos=false so nothing is prepended)
            std::vector<std::string> pieces;
            size_t base = 0;
            for (size_t k = 0; k < per_entry_tokens.size(); ++k) {
                const size_t m = p.find(marker, base);
                if (m == std::string::npos) throw std::runtime_error("decision prompt: media marker lost");
                pieces.push_back(p.substr(base, m - base));
                base = m + marker.size();
            }
            pieces.push_back(p.substr(base));
            auto head = c.tok->tokenize(pieces[0], true);
            auto vstart = c.tok->tokenize("<|vision_start|>", true);
            auto vend = c.tok->tokenize("<|vision_end|>", true);
            if (vstart.empty() || vend.empty()) throw std::runtime_error("decision model lacks <|vision_start|>/<|vision_end|> tokens");
            b.tokens = head;
            b.media_runs.reserve(per_entry_tokens.size());
            for (size_t k = 0; k < per_entry_tokens.size(); ++k) {
                if (per_entry_tokens[k] <= 0) throw std::invalid_argument("an image entry has no vision tokens");
                b.media_runs.push_back({ (int)b.tokens.size() + (int)vstart.size(), per_entry_tokens[k] });
                b.tokens.insert(b.tokens.end(), vstart.begin(), vstart.end());
                b.tokens.insert(b.tokens.end(), per_entry_tokens[k], 0);            // placeholder ids, replaced by the image rows
                b.tokens.insert(b.tokens.end(), vend.begin(), vend.end());
            }
            auto tail = c.tok->tokenize(pieces.back(), true);
            b.tokens.insert(b.tokens.end(), tail.begin(), tail.end());
        }
        if (b.tokens.empty()) throw std::invalid_argument("the instructions and the options of a question must not be empty");
        out.push_back(std::move(b));
    }
    return out;
}

// ---- answer -------------------------------------------------------------------------------------

Json lfm3d1_answer(const Lfm3D1Ctx& c, const Lfm3D1Built& q, const std::vector<float>& scores) {
    const size_t n = q.keys.size();
    if (scores.size() != n) throw std::runtime_error("decision result does not match the number of options");

    std::string name = type_name(q.type);
    std::string bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    float temp = 1.0f;
    auto it1 = c.temps.find(name + "." + bucket);
    auto it2 = c.temps.find(name);
    if (it1 != c.temps.end()) temp = it1->second;
    else if (it2 != c.temps.end()) temp = it2->second;

    std::vector<double> logits(scores.begin(), scores.end());
    if (q.type == DQ_SCORE) {
        auto it = c.score_bias.find((int)n);
        if (it != c.score_bias.end()) for (size_t i = 0; i < n; ++i) logits[i] += it->second[i];
    }
    double mx = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(n);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) { probs[i] = std::exp((logits[i] - mx) / temp); sum += probs[i]; }
    for (size_t i = 0; i < n; ++i) probs[i] /= sum;

    Json answer = Json::object();
    answer.set("type", Json::str(name));
    if (q.type == DQ_NOUL) { answer.set("noul", Json::num(probs[0])); return answer; }        // options are [true, false]

    Json probabilities = Json::object();
    for (size_t i = 0; i < n; ++i) probabilities.set(q.keys[i], Json::num(probs[i]));
    if (q.type == DQ_CHOICE) {
        size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
        answer.set("choice", Json::str(q.keys[best]));
        answer.set("probabilities", probabilities);
        answer.set("confidence", Json::num(*std::max_element(probs.begin(), probs.end())));
    } else {
        double mass = 0; size_t level = n - 1;                                                 // the median level (d1 api.py)
        for (size_t i = 0; i < n && mass < 0.5; ++i) { mass += probs[i]; level = i; }
        Json legend = Json::object();
        for (size_t i = 0; i < n; ++i) legend.set(q.keys[i], q.descriptions[i]);
        answer.set("score", Json::num((double)level));
        answer.set("legend", legend);
        answer.set("probabilities", probabilities);
        answer.set("confidence", Json::num(*std::max_element(probs.begin(), probs.end())));
    }
    return answer;
}

}  // namespace ferro
