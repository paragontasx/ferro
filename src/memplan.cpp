#include "memplan.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include "gguf.h"
#include "media.h"
#include "mm_encoder.h"
#include "server.h"

namespace ferro {

static double read_mb_file(const char* path) {
    double v = 0;
    if (FILE* f = fopen(path, "r")) {
        char buf[64] = {0};
        if (fgets(buf, sizeof buf, f) && strncmp(buf, "max", 3) != 0) v = strtod(buf, nullptr) / 1048576.0;
        fclose(f);
    }
    return v;
}

double cgroup_limit_mb() {
    if (const char* e = getenv("FERRO_MEM_LIMIT_MB")) return atof(e);
    double v = read_mb_file("/sys/fs/cgroup/memory.max");
    if (v <= 0) { v = read_mb_file("/sys/fs/cgroup/memory/memory.limit_in_bytes"); if (v > 1e7) v = 0; }   // v1 reports "unlimited" as a huge number
    return v;
}

double cgroup_used_mb() {
    double anon = 0, shmem = 0;
    if (FILE* f = fopen("/sys/fs/cgroup/memory.stat", "r")) {
        char k[64]; unsigned long long v;
        while (fscanf(f, "%63s %llu", k, &v) == 2) { if (!strcmp(k, "anon")) anon = (double)v; else if (!strcmp(k, "shmem")) shmem = (double)v; }
        fclose(f);
        return (anon + shmem) / 1048576.0;
    }
    if (FILE* f = fopen("/proc/self/status", "r")) {                 // no cgroup v2: this process only
        char line[256];
        while (fgets(line, sizeof line, f)) { unsigned long long kb; if (sscanf(line, "RssAnon: %llu kB", &kb) == 1) anon = kb / 1024.0; else if (sscanf(line, "RssShmem: %llu kB", &kb) == 1) shmem = kb / 1024.0; }
        fclose(f);
    }
    return anon + shmem;
}

static double file_mb(const std::string& path) { struct stat st; return !path.empty() && stat(path.c_str(), &st) == 0 ? st.st_size / 1048576.0 : 0.0; }

MemPlan plan_memory(const ServerOptions& o) {
    MemPlan p; p.limit_mb = cgroup_limit_mb();
    auto add = [&](const std::string& what, double mb, const std::string& note = "") { p.lines.push_back({what, mb, note}); p.need_mb += mb; };
    add("runtime (OpenCL driver, buffers, threads)", 700);
    double w_total = 0, big = 0; std::string big_name;
    auto weights = [&](const std::string& what, const std::string& path) {
        if (path.empty()) return;
        const double mb = file_mb(path); if (mb <= 0) { p.lines.push_back({what, 0, "file not found: " + path}); return; }
        add(what + " weights", 1.3 * mb, path.substr(path.find_last_of('/') == std::string::npos ? 0 : path.find_last_of('/') + 1));
        w_total += mb; if (mb > big) { big = mb; big_name = what; }
    };
    weights("embedding model", o.model); weights("embedding mmproj", o.mmproj);
    weights("language model", o.lm);
    std::string d1mm = o.d1_mmproj.empty() && !o.d1.empty() ? find_mmproj_near(o.d1) : o.d1_mmproj;
    weights("d1 decision model", o.d1); weights("d1 mmproj", d1mm);
    double kv_slot_mb = 0, kv_mb = 0;
    if (!o.lm.empty()) {
        GgufFile g; std::string e;
        if (g.open_file(o.lm.c_str(), &e)) {
            const std::string P = g.get_str("general.architecture") + ".";
            const auto kvs = g.get_iarr(P + "attention.head_count_kv");
            const long n_head = g.get_int(P + "attention.head_count"), n_embd = g.get_int(P + "embedding_length");
            long n_attn = 0, n_kv = 0; for (auto v : kvs) if (v > 0) { ++n_attn; n_kv = std::max<long>(n_kv, v); }
            if (n_head > 0 && n_attn > 0) {
                const double per_val = o.kv_dtype == 1 ? 1.125 : o.kv_dtype == 2 ? 2.0 : 4.0;
                kv_slot_mb = (double)n_attn * o.lm_ctx * 2.0 * n_kv * (n_embd / n_head) * per_val / 1048576.0;
                kv_mb = kv_slot_mb * o.lm_slots;
                char nb[96]; snprintf(nb, sizeof nb, "%d slot(s) x %d ctx, %s", o.lm_slots, o.lm_ctx, o.kv_dtype == 1 ? "int8" : o.kv_dtype == 2 ? "f16" : "fp32");
                add("KV cache", kv_mb, nb);
            }
        }
        if (o.prefix_cache_mb > 0) add("prompt-prefix cache", (double)o.prefix_cache_mb, "at its full budget");
    }
    p.fits = p.limit_mb <= 0 || p.need_mb <= p.limit_mb;
    if (!p.fits) {
        const double over = p.need_mb - p.limit_mb; char b[320];
        if (kv_slot_mb > 0 && o.lm_slots > 1) {
            const int fit = (int)std::floor((p.limit_mb - (p.need_mb - kv_mb)) / kv_slot_mb);
            if (fit >= 1) { snprintf(b, sizeof b, "--lm-slots %d fits (each slot costs %.0f MB)", fit, kv_slot_mb); p.advice.push_back(b); }
        }
        if (kv_mb > 0 && o.kv_dtype == 0 && kv_mb * (1 - 1.125 / 4.0) >= over) { snprintf(b, sizeof b, "--kv-dtype int8 saves %.0f MB (small accuracy cost)", kv_mb * (1 - 1.125 / 4.0)); p.advice.push_back(b); }
        if (o.prefix_cache_mb >= over) { snprintf(b, sizeof b, "--prefix-cache-mb %d fits", (int)(o.prefix_cache_mb - over)); p.advice.push_back(b); }
        if (big > 0 && 1.3 * big >= over) { snprintf(b, sizeof b, "dropping the %s (%.0f MB) fits", big_name.c_str(), 1.3 * big); p.advice.push_back(b); }
        if (p.advice.empty()) p.advice.push_back("this combination does not fit: drop a model or raise the container memory limit");
    }
    return p;
}

void print_plan(const MemPlan& p, FILE* out) {
    for (const auto& l : p.lines) fprintf(out, "  %-44s %7.0f MB  %s\n", l.what.c_str(), l.mb, l.note.c_str());
    fprintf(out, "  %-44s %7.0f MB\n", "total needed (estimate)", p.need_mb);
    if (p.limit_mb > 0) fprintf(out, "  %-44s %7.0f MB  -> %s (%.0f MB %s)\n", "container memory limit", p.limit_mb, p.fits ? "fits" : "DOES NOT FIT",
                                std::fabs(p.limit_mb - p.need_mb), p.fits ? "headroom for request bursts" : "short");
    else fprintf(out, "  no container memory limit found: cannot check\n");
    for (const auto& a : p.advice) fprintf(out, "  try: %s\n", a.c_str());
}

// ---------------------------------------------------------------- request cost
static bool image_dims(const std::vector<uint8_t>& b, int* w, int* h) {
    if (b.size() > 24 && b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G') {          // PNG: IHDR
        *w = (b[16] << 24) | (b[17] << 16) | (b[18] << 8) | b[19]; *h = (b[20] << 24) | (b[21] << 16) | (b[22] << 8) | b[23]; return *w > 0 && *h > 0;
    }
    if (b.size() > 4 && b[0] == 0xFF && b[1] == 0xD8) {                                          // JPEG: first SOFn marker
        size_t i = 2;
        while (i + 9 < b.size()) {
            if (b[i] != 0xFF) { ++i; continue; }
            const uint8_t m = b[i + 1];
            if (m == 0xFF) { ++i; continue; }
            if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) { *h = (b[i + 5] << 8) | b[i + 6]; *w = (b[i + 7] << 8) | b[i + 8]; return *w > 0 && *h > 0; }
            if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) { i += 2; continue; }
            i += 2 + ((b[i + 2] << 8) | b[i + 3]);
        }
    }
    return false;
}

double media_cost_mb(const std::string& body) {
    if (body.size() < 2048 || body.find("data:") == std::string::npos) return 0;
    double decoded = 0, tower_mp = 0, other = 0; size_t pos = 0;
    while ((pos = body.find("data:", pos)) != std::string::npos) {
        const size_t semi = body.find(";base64,", pos);
        if (semi == std::string::npos || semi - pos > 64) { pos += 5; continue; }
        const std::string kind = body.substr(pos + 5, std::min<size_t>(semi - pos - 5, 24));
        const size_t b0 = semi + 8; size_t b1 = body.find('"', b0); if (b1 == std::string::npos) b1 = body.size();
        const double bytes_mb = (double)(b1 - b0) * 0.75 / 1048576.0;
        if (kind.compare(0, 5, "image") == 0) {
            int w = 0, h = 0; std::vector<uint8_t> head;
            try { head = base64_decode(body.substr(b0, std::min<size_t>(b1 - b0, 360000))); } catch (...) {}
            double mp = image_dims(head, &w, &h) ? (double)w * h / 1e6 : std::min(40.0, bytes_mb * 4.0);   // unknown format: ~0.25 MB per megapixel
            decoded += 9.0 * mp; tower_mp = std::max(tower_mp, std::min(mp, 2.5));
        } else if (kind.compare(0, 5, "audio") == 0) other += 64 + 12.0 * bytes_mb;
        else if (kind.compare(0, 5, "video") == 0) other += 96 + 20.0 * bytes_mb;
        else other += 16 + 6.0 * bytes_mb;
        pos = b1;
    }
    const double img = (decoded > 0 ? 48 + 180.0 * tower_mp + decoded : 0);
    return img + other;
}

}  // namespace ferro
