// json.cpp — see json.h
#include "json.h"

#include <algorithm>
#include <cfloat>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ferro {

bool Json::contains(const std::string& k) const { return find(k) != nullptr; }
const Json* Json::find(const std::string& k) const {
    if (t != Object) return nullptr;
    for (auto& kv : obj) if (kv.first == k) return &kv.second;
    return nullptr;
}
const Json& Json::at(const std::string& k) const {
    const Json* p = find(k);
    if (!p) throw std::runtime_error("missing key: " + k);
    return *p;
}
Json& Json::set(const std::string& k, Json v) {
    if (t == Null) t = Object;
    for (auto& kv : obj) if (kv.first == k) { kv.second = std::move(v); return kv.second; }
    obj.emplace_back(k, std::move(v));
    return obj.back().second;
}
bool Json::operator==(const Json& o) const {
    if (t != o.t) {
        if ((t == Int || t == Float) && (o.t == Int || o.t == Float))
            return (t == Int ? (double)i : d) == (o.t == Int ? (double)o.i : o.d);
        return false;
    }
    switch (t) {
        case Null: return true;
        case Bool: return b == o.b;
        case Int: return i == o.i;
        case Float: return d == o.d;
        case String: return s == o.s;
        case Array: return arr == o.arr;
        case Object: return obj == o.obj;
    }
    return false;
}

// ---------------- parser ----------------
namespace {
struct Parser {
    const std::string& s; size_t p = 0;
    [[noreturn]] void fail(const char* what) {
        throw std::runtime_error(std::string("JSON parse error at byte ") + std::to_string(p) + ": " + what);
    }
    void ws() { while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) ++p; }
    bool lit(const char* w) { size_t n = strlen(w); if (s.compare(p, n, w) == 0) { p += n; return true; } return false; }
    static void put_utf8(std::string& o, uint32_t c) {
        if (c < 0x80) o += (char)c;
        else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 63)); }
        else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 63)); o += (char)(0x80 | (c & 63)); }
        else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 63)); o += (char)(0x80 | ((c >> 6) & 63)); o += (char)(0x80 | (c & 63)); }
    }
    uint32_t hex4() {
        if (p + 4 > s.size()) fail("bad \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[p++]; v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }
    std::string str() {
        if (s[p] != '"') fail("expected string");
        ++p; std::string o;
        while (true) {
            if (p >= s.size()) fail("unterminated string");
            unsigned char c = s[p++];
            if (c == '"') break;
            if (c < 0x20) fail("control character in string");
            if (c != '\\') { o += (char)c; continue; }
            if (p >= s.size()) fail("bad escape");
            char e = s[p++];
            switch (e) {
                case '"': o += '"'; break; case '\\': o += '\\'; break; case '/': o += '/'; break;
                case 'b': o += '\b'; break; case 'f': o += '\f'; break; case 'n': o += '\n'; break;
                case 'r': o += '\r'; break; case 't': o += '\t'; break;
                case 'u': {
                    uint32_t c1 = hex4();
                    if (c1 >= 0xD800 && c1 <= 0xDBFF) {
                        if (!(p + 1 < s.size() && s[p] == '\\' && s[p + 1] == 'u')) fail("lone surrogate");
                        p += 2; uint32_t c2 = hex4();
                        if (c2 < 0xDC00 || c2 > 0xDFFF) fail("bad surrogate pair");
                        c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
                    } else if (c1 >= 0xDC00 && c1 <= 0xDFFF) fail("lone surrogate");
                    put_utf8(o, c1); break;
                }
                default: fail("bad escape");
            }
        }
        return o;
    }
    Json num() {
        size_t b = p; bool is_float = false;
        if (s[p] == '-') ++p;
        if (p >= s.size() || !isdigit((unsigned char)s[p])) fail("bad number");
        if (s[p] == '0') ++p; else while (p < s.size() && isdigit((unsigned char)s[p])) ++p;
        if (p < s.size() && s[p] == '.') { is_float = true; ++p; if (p >= s.size() || !isdigit((unsigned char)s[p])) fail("bad number"); while (p < s.size() && isdigit((unsigned char)s[p])) ++p; }
        if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) { is_float = true; ++p; if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p; if (p >= s.size() || !isdigit((unsigned char)s[p])) fail("bad number"); while (p < s.size() && isdigit((unsigned char)s[p])) ++p; }
        std::string t = s.substr(b, p - b);
        if (!is_float) {
            errno = 0; char* e; long long v = strtoll(t.c_str(), &e, 10);
            if (errno == 0) return Json::integer(v);
        }
        return Json::num(strtod(t.c_str(), nullptr));
    }
    Json val(int depth) {
        if (depth > 512) fail("nesting too deep");
        ws();
        if (p >= s.size()) fail("unexpected end");
        char c = s[p];
        if (c == '{') {
            ++p; Json o = Json::object(); ws();
            if (p < s.size() && s[p] == '}') { ++p; return o; }
            while (true) {
                ws(); std::string k = str(); ws();
                if (p >= s.size() || s[p] != ':') fail("expected ':'");
                ++p; o.set(k, val(depth + 1)); ws();
                if (p < s.size() && s[p] == ',') { ++p; continue; }
                if (p < s.size() && s[p] == '}') { ++p; return o; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++p; Json a = Json::array(); ws();
            if (p < s.size() && s[p] == ']') { ++p; return a; }
            while (true) {
                a.push(val(depth + 1)); ws();
                if (p < s.size() && s[p] == ',') { ++p; continue; }
                if (p < s.size() && s[p] == ']') { ++p; return a; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') return Json::str(str());
        if (lit("true")) return Json::boolean(true);
        if (lit("false")) return Json::boolean(false);
        if (lit("null")) return Json();
        return num();
    }
};

// Python repr() of a float: shortest digits that read back (port of llama.cpp jinja float_repr, MIT)
std::string float_repr(double val) {
    if (!std::isfinite(val)) return std::isnan(val) ? "nan" : val < 0 ? "-inf" : "inf";
    const double mag = std::fabs(val);
    int exp2 = 0;
    const bool is_pow2 = std::frexp(mag, &exp2) == 0.5;
    std::string digits; int exp10 = 0;
    for (int prec = mag < DBL_MIN ? 0 : DBL_DIG - 1; digits.empty(); prec++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.*e", prec, mag);
        const char* e = std::strchr(buf, 'e');
        std::string d(buf, e - buf);
        d.erase(std::remove_if(d.begin(), d.end(), [](char c) { return c < '0' || c > '9'; }), d.end());
        exp10 = std::atoi(e + 1);
        auto reads_back = [&](const std::string& cand) {
            return std::strtod((cand + "e" + std::to_string(exp10 - prec)).c_str(), nullptr) == mag;
        };
        if (prec == 16 || reads_back(d)) digits = d;
        else if (is_pow2 && d.back() != '9') { d.back()++; if (reads_back(d)) digits = d; }
    }
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int n = (int)digits.size(), point = exp10 + 1;
    std::string out = std::signbit(val) ? "-" : "";
    if (point > -4 && point <= 16) {
        if (point <= 0) out += "0." + std::string(-point, '0') + digits;
        else if (point >= n) out += digits + std::string(point - n, '0') + ".0";
        else out += digits.substr(0, point) + "." + digits.substr(point);
    } else {
        char buf[16];
        snprintf(buf, sizeof(buf), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += digits.substr(0, 1) + (n > 1 ? "." + digits.substr(1) : "") + buf;
    }
    return out;
}

void jinja_str(std::string& o, const std::string& v) {
    o += '"';
    for (char c : v) {
        switch (c) {
            case '"': o += "\\\""; break; case '\\': o += "\\\\"; break;
            case '\b': o += "\\b"; break; case '\f': o += "\\f"; break;
            case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", (unsigned char)c); o += b; }
                else o += c;
        }
    }
    o += '"';
}

void to_jinja(std::string& o, const Json& v) {
    switch (v.t) {
        case Json::Null: o += "null"; break;
        case Json::Bool: o += v.b ? "true" : "false"; break;
        case Json::Int: o += std::to_string(v.i); break;
        case Json::Float:
            if (std::isnan(v.d)) o += "NaN";
            else if (std::isinf(v.d)) o += v.d < 0 ? "-Infinity" : "Infinity";
            else o += float_repr(v.d);
            break;
        case Json::String: jinja_str(o, v.s); break;
        case Json::Array:
            o += '[';
            for (size_t k = 0; k < v.arr.size(); ++k) { if (k) o += ", "; to_jinja(o, v.arr[k]); }
            o += ']';
            break;
        case Json::Object:
            o += '{';
            for (size_t k = 0; k < v.obj.size(); ++k) {
                if (k) o += ", ";
                jinja_str(o, v.obj[k].first); o += ": "; to_jinja(o, v.obj[k].second);
            }
            o += '}';
            break;
    }
}

void dump_str(std::string& o, const std::string& v) {
    o += '"';
    for (char c : v) {
        switch (c) {
            case '"': o += "\\\""; break; case '\\': o += "\\\\"; break;
            case '\b': o += "\\b"; break; case '\f': o += "\\f"; break;
            case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", (unsigned char)c); o += b; }
                else o += c;
        }
    }
    o += '"';
}

void dump(std::string& o, const Json& v) {
    switch (v.t) {
        case Json::Null: o += "null"; break;
        case Json::Bool: o += v.b ? "true" : "false"; break;
        case Json::Int: o += std::to_string(v.i); break;
        case Json::Float:
            if (!std::isfinite(v.d)) o += "null";
            else { std::string r = float_repr(v.d); o += r; }
            break;
        case Json::String: dump_str(o, v.s); break;
        case Json::Array:
            o += '[';
            for (size_t k = 0; k < v.arr.size(); ++k) { if (k) o += ','; dump(o, v.arr[k]); }
            o += ']';
            break;
        case Json::Object:
            o += '{';
            for (size_t k = 0; k < v.obj.size(); ++k) {
                if (k) o += ',';
                dump_str(o, v.obj[k].first); o += ':'; dump(o, v.obj[k].second);
            }
            o += '}';
            break;
    }
}
}  // namespace

Json json_parse(const std::string& text) {
    Parser P{text};
    Json v = P.val(0);
    P.ws();
    if (P.p != text.size()) P.fail("trailing characters");
    return v;
}
std::string json_dump(const Json& v) { std::string o; dump(o, v); return o; }
std::string json_to_jinja(const Json& v) { std::string o; to_jinja(o, v); return o; }

}  // namespace ferro
