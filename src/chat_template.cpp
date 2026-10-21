// chat_template.cpp - see chat_template.h
#include "chat_template.h"

#include <stdexcept>
#include <vector>

namespace ferro {
namespace {

static std::string content_of(const Json& m, const char* who) {
    const Json* c = m.find("content");
    if (!c || c->is_null()) return std::string();
    if (c->is_string()) return c->s;
    if (c->is_array()) {                                   // OpenAI content parts: text parts are joined with "\n" (llama.cpp's server does the same)
        std::string o;
        bool first = true;
        for (const Json& p : c->arr) {
            if (!first && (p.is_string() || p.is_object())) o += "\n";
            first = false;
            if (p.is_string()) o += p.s;
            else if (p.is_object()) {
                const Json* t = p.find("type"); const Json* x = p.find("text");
                if (t && t->is_string() && t->s == "text" && x && x->is_string()) o += x->s;
                else throw std::invalid_argument(std::string(who) + ": only text content parts are supported by this model");
            }
        }
        return o;
    }
    return json_to_jinja(*c);                              // the template applies `| tojson` to non-string content
}
static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char)s[a])) ++a;
    while (b > a && isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

}  // namespace

std::string lfm_chat_render(const Json& messages, const Json* tools, const ChatTemplateOptions& opt) {
    if (!messages.is_array() || messages.arr.empty()) throw std::invalid_argument("\"messages\" must be a non-empty array");
    size_t first = 0;
    std::string sys;
    auto role_of = [](const Json& m) {
        const Json* r = m.is_object() ? m.find("role") : nullptr;
        if (!r || !r->is_string()) throw std::invalid_argument("every message needs a string \"role\"");
        return r->s;
    };
    if (role_of(messages.arr[0]) == "system") { sys = content_of(messages.arr[0], "system message"); first = 1; }
    if (tools && tools->is_array() && !tools->arr.empty()) {
        sys += sys.empty() ? "" : "\n";
        sys += "List of tools: [";
        for (size_t i = 0; i < tools->arr.size(); ++i) {
            if (i) sys += ", ";
            sys += tools->arr[i].is_string() ? tools->arr[i].s : json_to_jinja(tools->arr[i]);
        }
        sys += "]";
    }
    std::string out = "<|startoftext|>";
    if (!sys.empty()) out += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    long last_assistant = -1;
    for (size_t i = first; i < messages.arr.size(); ++i) if (role_of(messages.arr[i]) == "assistant") last_assistant = (long)i;
    for (size_t i = first; i < messages.arr.size(); ++i) {
        const Json& m = messages.arr[i];
        const std::string role = role_of(m);
        std::string c = content_of(m, role.c_str());
        if (role == "assistant" && !opt.keep_past_thinking && (long)i != last_assistant) {
            const size_t p = c.rfind("</think>");
            if (p != std::string::npos) c = trim(c.substr(p + 8));
        }
        out += "<|im_start|>" + role + "\n" + c + "<|im_end|>\n";
    }
    if (opt.add_generation_prompt) out += "<|im_start|>assistant\n";
    return out;
}

}  // namespace ferro
