// lm_cli.cpp - see lm_cli.h
#include "lm_cli.h"
#include "chat_template.h"
#include "json.h"
#include "model_lfm2_lm.h"
#include "threadpool.h"
#include "tokenizer_bpe.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace ferro {

int run_generate(const GenArgs& a) {
    struct PoolGuard { ~PoolGuard() { Pool::get().shutdown(); } } pool_guard;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); };
    auto t0 = clk::now();
    LmEngine eng; std::string err;
    if (!eng.init(a.lm, a.threads, a.ctx, &err, getenv("FERRO_LM_SELFCHECK") ? 3 : 1)) { fprintf(stderr, "init failed: %s\n", err.c_str()); return 1; }
    BpeTokenizer tok;
    if (!tok.load(eng.model().gguf, &err)) { fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
    fprintf(stderr, "[lm] %d layers (%d conv + %d attention), d=%d, vocab %d, loaded in %.2fs, device=%s\n", eng.model().n_layer, eng.model().n_conv_layers,
            eng.model().n_attn_layers, eng.model().n_embd, eng.model().n_vocab, secs(t0), eng.on_gpu() ? "igpu" : "cpu");

    if (!a.logits_batch.empty()) {                              // score many raw prompts in one model load (parity tooling)
        std::ifstream f(a.logits_batch); std::stringstream ss; ss << f.rdbuf();
        Json arr = json_parse(ss.str());
        std::vector<float> lg((size_t)eng.n_vocab());
        for (size_t i = 0; i < arr.arr.size(); ++i) {
            std::vector<int> pi = tok.tokenize("<|startoftext|>" + arr.arr[i].s, true);
            LmSeq sq; eng.reset(sq);
            auto t1 = clk::now();
            if (!eng.step({&sq}, {pi.data()}, {(int)pi.size()}, lg.data(), &err, true)) { fprintf(stderr, "prompt %zu failed: %s\n", i, err.c_str()); return 1; }
            eng.release(sq);
            fprintf(stderr, "[lm] prompt %zu: %zu tokens, prefill %.3fs (%.0f tok/s)\n", i, pi.size(), secs(t1), pi.size() / secs(t1));
            FILE* o = fopen((a.dump_logits + "." + std::to_string(i) + ".bin").c_str(), "wb"); if (o) { fwrite(lg.data(), 4, lg.size(), o); fclose(o); }
        }
        return 0;
    }
    ChatTemplateOptions copt; copt.typed_content = lfm_template_is_typed(eng.model().gguf.get_str("tokenizer.chat_template"));
    std::string text;
    try {
        if (!a.messages_file.empty()) {
            std::ifstream f(a.messages_file); if (!f) { fprintf(stderr, "cannot read %s\n", a.messages_file.c_str()); return 1; }
            std::stringstream ss; ss << f.rdbuf();
            Json body = json_parse(ss.str());
            const Json* msgs = body.is_array() ? &body : body.find("messages");
            if (!msgs) throw std::invalid_argument("--messages file must be an array of messages or an object with \"messages\"");
            text = lfm_chat_render(*msgs, body.is_object() ? body.find("tools") : nullptr, copt);
        } else if (a.raw) text = "<|startoftext|>" + a.prompt;
        else {
            Json msgs = Json::array();
            if (!a.system.empty()) { Json m = Json::object(); m.set("role", Json::str("system")); m.set("content", Json::str(a.system)); msgs.push(m); }
            Json u = Json::object(); u.set("role", Json::str("user")); u.set("content", Json::str(a.prompt)); msgs.push(u);
            text = lfm_chat_render(msgs, nullptr, copt);
        }
    } catch (const std::exception& e) { fprintf(stderr, "bad prompt: %s\n", e.what()); return 2; }
    std::vector<int> ids = tok.tokenize(text, true);
    if ((int)ids.size() + 1 > eng.max_ctx()) { fprintf(stderr, "prompt is %zu tokens, context is %d\n", ids.size(), eng.max_ctx()); return 2; }
    if (a.print_ids) { fprintf(stderr, "prompt ids (%zu):", ids.size()); for (int i : ids) fprintf(stderr, " %d", i); fprintf(stderr, "\n"); }

    if (getenv("FERRO_LM_SELFCHECK")) {                       // prefill-all must equal chunked prefill and token-by-token decode
        const int N = (int)ids.size(), V = eng.n_vocab();
        std::vector<float> ref(V), cur(V);
        LmSeq s0; eng.reset(s0);
        eng.step({&s0}, {ids.data()}, {N}, ref.data(), &err, true);
        eng.release(s0);
        auto cmp = [&](const char* name, const std::vector<float>& o) {
            double dot = 0, na = 0, nb = 0, mx = 0; int am = 0, bm = 0;
            for (int i = 0; i < V; ++i) { dot += (double)ref[i] * o[i]; na += (double)ref[i] * ref[i]; nb += (double)o[i] * o[i]; mx = std::max(mx, (double)std::fabs(ref[i] - o[i])); if (ref[i] > ref[am]) am = i; if (o[i] > o[bm]) bm = i; }
            fprintf(stderr, "[selfcheck] %-34s cos %.8f  max|diff| %.4g  argmax %d vs %d\n", name, dot / std::sqrt(na * nb), mx, am, bm);
        };
        for (int split : {1, 2, 3, N / 2}) {
            if (split <= 0 || split >= N) continue;
            LmSeq s1; eng.reset(s1);
            eng.step({&s1}, {ids.data()}, {split}, cur.data(), &err, true);
            eng.step({&s1}, {ids.data() + split}, {N - split}, cur.data(), &err, true);
            eng.release(s1);
            char nm[64]; snprintf(nm, sizeof nm, "prefill %d + prefill %d", split, N - split); cmp(nm, cur);
        }
        LmSeq s2; eng.reset(s2);
        for (int i = 0; i < N; ++i) eng.step({&s2}, {ids.data() + i}, {1}, cur.data(), &err);
        eng.release(s2);
        cmp("token-by-token decode", cur);
        // prefix cache: snapshot after `split` tokens, restore into ANOTHER slot (the original keeps running), finish the prompt; then continue by decode on both
        for (int split : {2, N / 2, N - 1}) {
            if (split <= 1 || split >= N || eng.n_slots() < 2) continue;
            LmSeq a; eng.reset(a);
            eng.step({&a}, {ids.data()}, {split}, cur.data(), &err, true);
            auto px = eng.snapshot(a);
            if (!px) { fprintf(stderr, "[selfcheck] snapshot unavailable (CPU path?)\n"); break; }
            LmSeq pad; eng.reset(pad);                                        // occupy slot 0/1 differently so the restore lands in a different slot
            LmSeq b; b.n_past = 0;
            if (!eng.restore(b, *px)) { fprintf(stderr, "[selfcheck] restore failed\n"); return 1; }
            eng.step({&b}, {ids.data() + split}, {N - split}, cur.data(), &err, true);
            char nm[64]; snprintf(nm, sizeof nm, "restore@%d (slot %d->%d) + prefill", split, a.gpu_slot, b.gpu_slot); cmp(nm, cur);
            // the snapshot must be unaffected by what the donor did afterwards, and reusable: restore again after the donor advanced
            eng.step({&a}, {ids.data() + split}, {N - split}, cur.data(), &err, true);
            eng.release(b);
            LmSeq c; if (!eng.restore(c, *px)) { fprintf(stderr, "[selfcheck] 2nd restore failed\n"); return 1; }
            eng.step({&c}, {ids.data() + split}, {N - split}, cur.data(), &err, true);
            snprintf(nm, sizeof nm, "restore@%d again (donor advanced)", split); cmp(nm, cur);
            const int nxt = 1098;                                              // one decode step after the restored prefill == after the donor's
            std::vector<float> d1(V), d2(V);
            eng.step({&c}, {&nxt}, {1}, d1.data(), &err); eng.step({&a}, {&nxt}, {1}, d2.data(), &err);
            double mx = 0; for (int i = 0; i < V; ++i) mx = std::max(mx, (double)std::fabs(d1[i] - d2[i]));
            fprintf(stderr, "[selfcheck] %-34s max|diff| %.4g\n", "decode after restore vs donor", mx);
            eng.release(a); eng.release(c); eng.release(pad);
        }
        return 0;
    }
    LmSeq seq; eng.reset(seq);
    std::vector<float> logits((size_t)eng.n_vocab());
    auto tp = clk::now();
    const int n0 = (int)ids.size();
    if (!eng.step({&seq}, {ids.data()}, {n0}, logits.data(), &err, true)) { fprintf(stderr, "prefill failed: %s\n", err.c_str()); return 1; }
    const double t_pre = secs(tp);
    if (!a.dump_logits.empty()) { FILE* f = fopen(a.dump_logits.c_str(), "wb"); if (f) { fwrite(logits.data(), 4, logits.size(), f); fclose(f); } }

    Sampler sampler(a.sp);
    std::vector<int> hist = ids, gen;
    auto td = clk::now();
    int stop = 0;
    for (int i = 0; i < a.n_predict; ++i) {
        const int nxt = sampler.sample(logits.data(), eng.n_vocab(), hist);
        if (nxt == eng.model().eos_id && !getenv("FERRO_LM_NOEOS")) { stop = 1; break; }     // FERRO_LM_NOEOS=1: timing experiments
        gen.push_back(nxt); hist.push_back(nxt);
        if (!a.quiet) { const std::string s = tok.decode_piece(nxt); fwrite(s.data(), 1, s.size(), stdout); fflush(stdout); }
        if (seq.n_past + 1 >= eng.max_ctx()) { stop = 2; break; }
        if (!eng.step({&seq}, {&nxt}, {1}, logits.data(), &err)) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return 1; }
    }
    const double t_dec = secs(td);
    if (!a.quiet) printf("\n");
    if (a.print_ids) { fprintf(stderr, "generated ids (%zu):", gen.size()); for (int i : gen) fprintf(stderr, " %d", i); fprintf(stderr, "\n"); }
    fprintf(stderr, "[lm] prefill %d tokens in %.3fs (%.1f tok/s) | decode %zu tokens in %.3fs (%.2f tok/s) | stop: %s\n", n0, t_pre, n0 / std::max(t_pre, 1e-9),
            gen.size(), t_dec, gen.size() / std::max(t_dec, 1e-9), stop == 1 ? "eos" : stop == 2 ? "context" : "length");
    return 0;
}

}  // namespace ferro
