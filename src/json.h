// json.h — minimal order-preserving JSON value for the decision API (no dependencies).
// parse(): RFC 8259 (+ duplicate keys overwrite in place, like nlohmann::ordered_json).
// to_jinja(): byte-exact llama.cpp jinja `tojson` (", " / ": " separators, raw UTF-8,
//             Python-repr floats) — the systemone prompt depends on it.
// dump(): compact standard JSON for output.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ferro {

struct Json {
    enum Type { Null, Bool, Int, Float, String, Array, Object } t = Null;
    bool b = false;
    int64_t i = 0;
    double d = 0;
    std::string s;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    Json() = default;
    static Json str(std::string v) { Json j; j.t = String; j.s = std::move(v); return j; }
    static Json num(double v) { Json j; j.t = Float; j.d = v; return j; }
    static Json integer(int64_t v) { Json j; j.t = Int; j.i = v; return j; }
    static Json boolean(bool v) { Json j; j.t = Bool; j.b = v; return j; }
    static Json object() { Json j; j.t = Object; return j; }
    static Json array() { Json j; j.t = Array; return j; }

    bool is_null() const { return t == Null; }
    bool is_string() const { return t == String; }
    bool is_object() const { return t == Object; }
    bool is_array() const { return t == Array; }
    bool contains(const std::string& k) const;
    const Json& at(const std::string& k) const;          // throws if absent
    const Json* find(const std::string& k) const;        // nullptr if absent
    Json& set(const std::string& k, Json v);             // insert or overwrite (keeps position)
    void push(Json v) { arr.push_back(std::move(v)); }
    bool operator==(const Json& o) const;
    bool operator!=(const Json& o) const { return !(*this == o); }
};

Json json_parse(const std::string& text);                // throws std::runtime_error
std::string json_dump(const Json& v);                    // compact output JSON
std::string json_to_jinja(const Json& v);                // llama.cpp jinja `tojson`

}  // namespace ferro
