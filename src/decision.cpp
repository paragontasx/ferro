// decision.cpp — see decision.h. Mirrors llama.cpp server-decision.cpp (LFM2_D1_OMNI, text).
#include "decision.h"
#include "media.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace ferro {

static const std::string MARKER = "<<d1omni:";
static const std::string MARK_STATE = "<<d1omni:state>>";
static const std::string MARK_QUESTION = "<<d1omni:question>>";
static const std::string MARK_OPTION = "<<d1omni:option>>";
static const size_t MAX_TOKENS = 16384;         // D1OMNI_MAX_TOKENS
static const size_t MAX_TOKENS_IMAGE = 896;     // D1OMNI_MAX_TOKENS_IMAGE: text budget after images
static const size_t MAX_TOKENS_AUDIO = 15360;   // D1OMNI_MAX_TOKENS_AUDIO: text budget after an audio clip
static const std::string MEDIA_MARKER = "<__media__>";   // mtmd default marker, blanked out of request text

static const char* type_name(DecisionType t) {
    return t == DQ_CHOICE ? "choice" : t == DQ_SCORE ? "score" : "noul";
}

bool DecisionPrompter::init(const GgufFile& g, const BpeTokenizer* tok, std::string* err) {
    tok_ = tok;
    const std::string pre = g.arch + ".decision.";
    if (g.get_str(pre + "type") != "lfm2-d1-omni") { *err = "not an lfm2-d1-omni decision model"; return false; }
    if (tok->mask_id() < 0) { *err = "decision model has no mask token"; return false; }
    for (const auto& k : g.keys()) {
        if (k.rfind(pre + "temperature.", 0) == 0) {
            float t = (float)g.get_f64(k, 1.0);
            if (t <= 0) { *err = "invalid decision temperature " + k; return false; }
            temps_[k.substr(pre.size() + 12)] = t;
        } else if (k.rfind(pre + "score_bias.", 0) == 0) {
            std::string rest = k.substr(pre.size() + 11);
            size_t dot = rest.find('.');
            if (dot == std::string::npos) continue;
            size_t n = std::stoul(rest.substr(0, dot)), lvl = std::stoul(rest.substr(dot + 1));
            if (n < 2 || n > 255 || lvl >= n) { *err = "invalid score bias " + k; return false; }
            auto& b = score_bias_[n]; b.resize(n, 0.f); b[lvl] = (float)g.get_f64(k, 0.0);
        }
    }
    return true;
}

// ---------------------------------------------------------------- request parsing
DecisionRequest DecisionPrompter::parse(const Json& body) const {
    if (!body.is_object()) throw std::invalid_argument("request must be a JSON object");
    if (!body.contains("state")) throw std::invalid_argument("\"state\" must be provided");
    DecisionRequest req;
    // ---- media (server-decision.cpp parse_state): "files" / "images" arrays of data URLs, and image_url /
    // input_audio parts of chat-message states
    {
        const Json* v = body.find("videos");
        if (v && !v->is_null() && !(v->is_array() && v->arr.empty())) throw std::invalid_argument("\"videos\" is not supported");
    }
    auto add_file = [&](const Json& url, bool audio) {
        if (audio) {
            if (!url.is_string() || url.s.rfind("http", 0) == 0 || url.s.rfind("file://", 0) == 0)
                throw std::invalid_argument("audio must be base64 data");
        } else if (!url.is_string() || url.s.rfind("data:", 0) != 0) {
            throw std::invalid_argument("images must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)");
        }
        if (req.files.size() >= MAX_MEDIA_FILES)
            throw std::invalid_argument(std::string("too many ") + (audio ? "media files" : "images") + ", the maximum is " + std::to_string(MAX_MEDIA_FILES));
        req.files.push_back(media_bytes_from_url(url.s));
    };
    for (const char* key : {"files", "images"}) {
        const Json* m = body.find(key);
        if (!m || m->is_null()) continue;
        if (!m->is_array()) throw std::invalid_argument(std::string("\"") + key + "\" must be an array");
        for (const auto& u : m->arr) add_file(u, false);
    }
    Json state = body.at("state");
    {
        const bool wrapped = state.is_object() && state.contains("messages");
        const Json& messages = wrapped ? state.at("messages") : state;
        if (messages.is_array()) {
            Json out = Json::array();
            for (const auto& msg : messages.arr) {
                const Json* content = msg.is_object() ? msg.find("content") : nullptr;
                if (!content || !content->is_array()) { out.push(msg); continue; }
                Json kept = Json::array();
                for (const auto& part : content->arr) {
                    const Json* ty = part.is_object() ? part.find("type") : nullptr;
                    const std::string tn = ty && ty->is_string() ? ty->s : "";
                    if (tn == "image_url" && part.contains("image_url")) {
                        const Json& iu = part.at("image_url");
                        add_file(iu.is_object() && iu.contains("url") ? iu.at("url") : iu, false);
                    } else if (tn == "input_audio" && part.contains("input_audio")) {
                        const Json& ia = part.at("input_audio");
                        Json d;
                        if (ia.is_object() && ia.contains("data")) d = ia.at("data");
                        else if (ia.is_object() && ia.contains("url")) d = ia.at("url");
                        add_file(d, true);
                    } else {
                        kept.push(part);
                    }
                }
                Json m2 = msg;
                m2.set("content", kept);
                out.push(m2);
            }
            if (wrapped) state.set("messages", out); else state = out;
        }
    }
    const Json* qs = body.find("questions");
    if (!qs || !qs->is_object() || qs->obj.empty()) throw std::invalid_argument("\"questions\" must be a non-empty object");

    req.state = state;
    for (const auto& kv : qs->obj) {
        const std::string& id = kv.first;
        const Json& q = kv.second;
        auto err = [&](const std::string& m) { return std::invalid_argument("questions." + id + ": " + m); };
        if (!q.is_object()) throw err("must be an object");
        const Json* ins = q.find("instructions");
        if (!ins || ins->is_null()) throw err("\"instructions\" must be provided");
        DecisionQuestion dq;
        dq.id = id;
        dq.instructions = *ins;
        const Json* tj = q.find("type");
        std::string tn = tj && tj->is_string() ? tj->s : "";
        const Json* cr = q.find("criteria");
        Json crit = cr ? *cr : Json();
        if (tn == "choice") {
            dq.type = DQ_CHOICE;
            if (!crit.is_object() || crit.obj.empty()) throw err("\"criteria\" must be a non-empty object");
            for (const auto& o : crit.obj) dq.options.push_back({o.first, o.second});
        } else if (tn == "score") {
            dq.type = DQ_SCORE;
            if (!crit.is_array() || crit.arr.size() < 2 || crit.arr.size() > 10) throw err("\"criteria\" must be an array of 2 to 10 levels");
            for (size_t i = 0; i < crit.arr.size(); ++i) dq.options.push_back({std::to_string(i), crit.arr[i]});
        } else if (tn == "noul") {
            dq.type = DQ_NOUL;
            if (!crit.is_null() && !crit.is_object()) throw err("\"criteria\" must be an object");
            for (const char* key : {"false", "true"}) {
                Json d;
                if (crit.is_object() && crit.contains(key)) d = crit.at(key);
                else if (crit.is_object()) {                       // d1-omni also reads "no" / "yes"
                    const char* alias = std::string(key) == "true" ? "yes" : "no";
                    if (crit.contains(alias)) d = crit.at(alias);
                }
                dq.options.push_back({key, d});
            }
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
        }
        if (dq.options.size() > 255) throw err("too many options (" + std::to_string(dq.options.size()) + "), this model supports at most 255");
        req.questions.push_back(std::move(dq));
    }
    return req;
}

// ---------------------------------------------------------------- prompt
// "<|name|>" -> "<¦name¦>" so request text cannot inject special tokens
static std::string escape_special(const std::string& text) {
    auto is_name = [](char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'; };
    std::string out; size_t done = 0;
    for (size_t i = text.find("<|"); i != std::string::npos; i = text.find("<|", std::max(i + 1, done))) {
        size_t j = i + 2;
        while (j < text.size() && is_name(text[j])) j++;
        if (j > i + 2 && text.compare(j, 2, "|>") == 0) {
            out.append(text, done, i - done);
            out += "<\xC2\xA6"; out.append(text, i + 2, j - i - 2); out += "\xC2\xA6>";
            done = j + 2;
        }
    }
    if (done == 0) return text;
    out.append(text, done, std::string::npos);
    return out;
}
static void replace_all(std::string& s, const std::string& a, const std::string& b) {
    if (a.empty()) return;
    for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + b.size())) s.replace(p, a.size(), b);
}
// decision_d1omni_escape (strings and keys) then decision_replace_text(MARKER) (strings only); with media,
// decision_replace_text(media marker -> " ") as well
static Json sanitize(const Json& v, bool media) {
    if (v.is_string()) {
        std::string s = escape_special(v.s);
        replace_all(s, MARKER, "<<d1omni ");
        if (media) replace_all(s, MEDIA_MARKER, " ");
        return Json::str(s);
    }
    if (v.is_array()) { Json o = Json::array(); for (auto& x : v.arr) o.push(sanitize(x, media)); return o; }
    if (v.is_object()) { Json o = Json::object(); for (auto& kv : v.obj) o.set(escape_special(kv.first), sanitize(kv.second, media)); return o; }
    return v;
}
static std::string str_or_json(const Json& v) { return v.is_string() ? v.s : json_to_jinja(v); }
static bool has_desc(const Json& d) { return !d.is_null() && !(d.is_string() && d.s.empty()); }

std::vector<std::string> DecisionPrompter::render_pieces(const DecisionRequest& req, const DecisionQuestion& q,
                                                        const DecisionMediaInfo& media) const {
    const bool has_media = media.n_files > 0, audio = media.is_audio;
    const Json state = sanitize(req.state, has_media), ins = sanitize(q.instructions, has_media);
    bool criteria = false;   // template ns.criteria: some option has a description that is not none
    for (const auto& o : q.options) if (!o.description.is_null()) criteria = true;
    std::vector<std::string> p;
    p.push_back("<|startoftext|><|reserved_7|>");
    std::string st = state.is_string() ? state.s : !state.is_null() ? json_to_jinja(state) : audio ? "{}" : "";
    p.push_back(MARK_STATE + st);
    p.push_back(MARK_QUESTION + "<|reserved_8|>" + str_or_json(ins));
    int idx = 0;
    for (const auto& o0 : q.options) {
        const std::string key = sanitize(Json::str(o0.key), has_media).s;
        const Json d = sanitize(o0.description, has_media);
        std::string opt;
        if (q.type == DQ_CHOICE && audio) {
            char nb[16];
            snprintf(nb, sizeof(nb), "option_%03d: ", idx);
            opt = nb + (has_desc(d) ? str_or_json(d) : key);
        }
        else if (q.type == DQ_CHOICE) opt = key + (has_desc(d) ? ": " + str_or_json(d) : "");
        else if (q.type == DQ_SCORE) opt = "level " + key + ": " + str_or_json(d);
        else if (audio) opt = key + ": " + (key == "true" ? "yes" : "no");
        else opt = key + ": " + (has_desc(d) ? str_or_json(d)
                                 : has_media && !criteria ? (key == "true" ? "yes" : "no")
                                 : key == "true" ? "yes, the statement holds" : "no, the statement does not hold");
        ++idx;
        p.push_back("<|reserved_9|><|mask|>");
        p.push_back(MARK_OPTION + " " + opt);
        p.push_back("<|reserved_10|>");
    }
    p.push_back("<|reserved_11|>");
    return p;
}

std::string DecisionPrompter::render_debug(const DecisionRequest& req, const DecisionQuestion& q, const DecisionMediaInfo& media) const {
    std::string s;
    for (auto& x : render_pieces(req, q, media)) s += x + "|";
    return s;
}

DecisionSeq DecisionPrompter::build(const DecisionRequest& req, const DecisionQuestion& q, const DecisionMediaInfo& media) const {
    size_t n_max = MAX_TOKENS;
    if (media.n_files > 0) {
        if (media.is_audio && media.n_files > 1) throw std::invalid_argument("a request has images or one audio clip, not both");
        const size_t nm = (size_t)media.n_tokens;
        n_max = std::min(media.is_audio ? MAX_TOKENS_AUDIO : MAX_TOKENS_IMAGE, MAX_TOKENS - std::min(MAX_TOKENS, nm));
        if (n_max < 64)
            throw std::invalid_argument("the media take " + std::to_string(nm) + " of the " + std::to_string(MAX_TOKENS) + " positions, send fewer images");
    }
    const int64_t n_opt = (int64_t)q.options.size();
    const int64_t n_budget = std::max<int64_t>(96, std::min<int64_t>(24 * n_opt + 32, (int64_t)n_max / 2));
    const int64_t n_option_max = std::max<int64_t>(2, (n_budget - 3 * n_opt) / n_opt);
    const int64_t n_question_max = std::max<int64_t>(16, n_budget);

    std::vector<int> head, body, tail;
    bool has_state = false;
    for (std::string piece : render_pieces(req, q, media)) {
        int64_t lim = -1; bool is_state = false;
        if (piece.rfind(MARK_STATE, 0) == 0) { piece = piece.substr(MARK_STATE.size()); is_state = true; }
        else if (piece.rfind(MARK_QUESTION, 0) == 0) { piece = piece.substr(MARK_QUESTION.size()); lim = n_question_max; }
        else if (piece.rfind(MARK_OPTION, 0) == 0) { piece = piece.substr(MARK_OPTION.size()); lim = n_option_max; }
        std::vector<int> t = tok_->tokenize(piece, true);
        if (lim >= 0 && (int64_t)t.size() > lim) t.resize(lim);
        if (is_state) { body = std::move(t); has_state = true; }
        else (has_state ? tail : head).insert((has_state ? tail : head).end(), t.begin(), t.end());
    }
    const size_t n_room = n_max - std::min(n_max, head.size() + tail.size());
    body.resize(std::min(body.size(), n_room));
    DecisionSeq s;
    s.type = q.type;
    s.n_media = media.n_files > 0 ? media.n_tokens : 0;
    s.tokens = std::move(head);
    s.tokens.insert(s.tokens.end(), body.begin(), body.end());
    s.tokens.insert(s.tokens.end(), tail.begin(), tail.end());
    if (s.tokens.size() > n_max) s.tokens.resize(n_max);
    for (size_t i = 0; i < s.tokens.size(); ++i)
        if (s.tokens[i] == tok_->mask_id()) s.markers.push_back((int)i);
    if ((int64_t)s.markers.size() != n_opt) throw std::invalid_argument("the options do not fit in the context");
    return s;
}

// ---------------------------------------------------------------- answers
float DecisionPrompter::temperature(const DecisionQuestion& q) const {
    const size_t n = q.options.size();
    const std::string tn = type_name(q.type);
    const std::string bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    for (const std::string& k : {tn + "." + bucket, tn}) {
        auto it = temps_.find(k);
        if (it != temps_.end()) return it->second;
    }
    return 1.0f;
}

Json DecisionPrompter::answer(const DecisionQuestion& q, const std::vector<float>& scores, bool has_media) const {
    const size_t n = q.options.size();
    if (scores.size() != n) throw std::runtime_error("decision result does not match the number of options");
    for (float v : scores) if (std::isnan(v)) throw std::runtime_error("the model could not evaluate the decision");
    const double temp = has_media ? 1.0 : temperature(q);
    std::vector<double> lg(scores.begin(), scores.end());
    if (q.type == DQ_SCORE && score_bias_.count(n)) for (size_t i = 0; i < n; ++i) lg[i] += score_bias_.at(n)[i];
    const double mx = *std::max_element(lg.begin(), lg.end());
    std::vector<double> p(n); double sum = 0;
    for (size_t i = 0; i < n; ++i) { p[i] = std::exp((lg[i] - mx) / temp); sum += p[i]; }
    for (auto& v : p) v /= sum;

    Json a = Json::object();
    a.set("type", Json::str(type_name(q.type)));
    if (q.type == DQ_NOUL) {
        for (size_t i = 0; i < n; ++i) if (q.options[i].key == "true") a.set("noul", Json::num(p[i]));
        return a;
    }
    Json probs = Json::object();
    for (size_t i = 0; i < n; ++i) probs.set(q.options[i].key, Json::num(p[i]));
    if (q.type == DQ_CHOICE) {
        size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        double conf = 1.0;
        if (n >= 2) { double u = 1.0 / n; conf = std::max(0.0, (p[best] - u) / (1.0 - u)); }
        a.set("choice", Json::str(q.options[best].key));
        a.set("probabilities", probs);
        a.set("confidence", Json::num(conf));
    } else {
        double expected = 0; Json legend = Json::object();
        for (size_t i = 0; i < n; ++i) { expected += i * p[i]; legend.set(q.options[i].key, q.options[i].description); }
        double conf = 1.0;
        if (n >= 2) {
            size_t mode = std::max_element(p.begin(), p.end()) - p.begin();
            double dist = 0, du = 0;
            for (size_t i = 0; i < n; ++i) { dist += p[i] * std::fabs((double)i - (double)mode); du += std::fabs((double)i - (n - 1) / 2.0) / n; }
            conf = std::max(0.0, 1.0 - dist / du);
        }
        a.set("score", Json::num(expected));
        a.set("legend", legend);
        a.set("probabilities", probs);
        a.set("confidence", Json::num(conf));
    }
    return a;
}

}  // namespace ferro
