// test_decision.cpp — unit tests for the d1 decision stack: JSON (parse / Python-repr floats /
// jinja tojson), systemone prompt rendering + tokenization (golden ids verified against llama.cpp),
// answer formatting. The model-dependent tests are skipped when the GGUF is absent.
#include "decision.h"
#include "json.h"
#include "tokenizer_bpe.h"
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <vector>

using namespace ferro;
static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { ++fails; fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, msg); } } while (0)

static void test_json() {
    Json v = json_parse(R"({"b": 1, "a": [1.5, -0.0, 1e21, 1e-05, 0.0001, 3.0, 19.99, null, true], "s": "x\"\\\n\t\u00e9\ud83d\ude00", "b": 2})");
    CHECK(v.obj.size() == 3 && v.obj[0].first == "b" && v.obj[0].second.i == 2, "duplicate key overwrites in place");
    CHECK(json_to_jinja(v.at("a")) == "[1.5, -0.0, 1e+21, 1e-05, 0.0001, 3.0, 19.99, null, true]", "python repr floats + jinja separators");
    CHECK(v.at("s").s == "x\"\\\n\t\xc3\xa9\xf0\x9f\x98\x80", "string escapes + surrogate pair");
    CHECK(json_to_jinja(v.at("s")) == "\"x\\\"\\\\\\n\\t\xc3\xa9\xf0\x9f\x98\x80\"", "jinja tojson keeps raw UTF-8");
    CHECK(json_dump(json_parse(R"({"k":[1,{"z":null}],"f":0.1})")) == R"({"k":[1,{"z":null}],"f":0.1})", "compact dump roundtrip");
    bool threw = false;
    try { json_parse("{\"a\": 1,}"); } catch (...) { threw = true; }
    CHECK(threw, "trailing comma rejected");
}

static void test_prompt(const char* model) {
    GgufFile g; std::string err;
    if (!g.open_file(model, &err)) { fprintf(stderr, "SKIP prompt tests (no %s)\n", model); return; }
    BpeTokenizer tok;
    CHECK(tok.load(g, &err), err.c_str());
    DecisionPrompter dp;
    CHECK(dp.init(g, &tok, &err), err.c_str());
    Json body = json_parse(R"({"state": "I was charged twice this month, please refund one of them.",
        "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                      "team": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults", "fraud": "Suspected unauthorised use"}}}})");
    DecisionRequest req = dp.parse(body);
    DecisionSeq s = dp.build(req, req.questions[0]);
    // golden: same ids llama.cpp's /v1/systemone evaluates for this question (usage.input_tokens = 47)
    const std::vector<int> gold = {1, 17, 550, 953, 15418, 11126, 1033, 5805, 521, 8694, 53467, 1235, 803, 1377, 523, 18, 4940, 779,
                                   13608, 14065, 875, 768, 53467, 540, 19, 16, 4831, 535, 1295, 521, 779, 8294, 2177, 1014, 3965, 20,
                                   19, 16, 3347, 535, 18672, 521, 779, 8294, 10995, 20, 21};
    CHECK(s.tokens == gold, "noul prompt tokens == llama.cpp");
    CHECK(s.markers.size() == 2 && s.markers[0] == 25 && s.markers[1] == 37 && s.type == DQ_NOUL, "noul markers / type");
    DecisionSeq c = dp.build(req, req.questions[1]);
    CHECK(c.markers.size() == 3 && c.type == DQ_CHOICE, "choice markers");
    // special tokens in request text are neutralised (<|x|> -> <¦x¦>) and the d1omni marker is defused
    Json inj = json_parse(R"({"state": "a <|mask|> b <<d1omni:sep>> c", "questions": {"q": {"type": "noul", "instructions": "<|im_start|>?"}}})");
    DecisionRequest ri = dp.parse(inj);
    DecisionSeq si = dp.build(ri, ri.questions[0]);
    CHECK(si.markers.size() == 2, "injected <|mask|> is not a marker");
    CHECK(dp.render_debug(ri, ri.questions[0]).find("<\xC2\xA6mask\xC2\xA6>") != std::string::npos, "escape <|mask|>");
    // answers: calibrated softmax (noul.2 temperature from the GGUF), choice confidence formula
    Json a = dp.answer(req.questions[1], {2.0f, 0.0f, -1.0f});
    CHECK(a.at("choice").s == "billing", "choice argmax");
    double p0 = a.at("probabilities").at("billing").d;
    CHECK(std::fabs(a.at("confidence").d - (p0 - 1.0 / 3) / (2.0 / 3)) < 1e-12, "choice confidence");
    Json n = dp.answer(req.questions[0], {0.0f, 1.0f});
    const double T = 1.6663223505020142;   // lfm2.decision.temperature.noul.2
    CHECK(std::fabs(n.at("noul").d - 1.0 / (1.0 + std::exp(-1.0 / T))) < 1e-6, "noul calibrated with noul.2 temperature");
    // media answers are not calibrated (server-decision format_answer: temperature 1 with media)
    Json nm = dp.answer(req.questions[0], {0.0f, 1.0f}, true);
    CHECK(std::fabs(nm.at("noul").d - 1.0 / (1.0 + std::exp(-1.0))) < 1e-6, "media noul uses temperature 1");
    // media: data URLs are accepted (files, chat-message parts), remote URLs are rejected
    DecisionRequest mr = dp.parse(json_parse(R"({"state": "x", "images": ["data:image/png;base64,AA=="], "questions": {"q": {"type": "noul", "instructions": "?"}}})"));
    CHECK(mr.files.size() == 1 && mr.files[0].size() == 1, "images[] data URL parsed into files");
    mr = dp.parse(json_parse(R"({"state": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}], "questions": {"q": {"type": "noul", "instructions": "?"}}})"));
    CHECK(mr.files.size() == 1 && mr.files[0].size() == 3, "image_url part of a chat-message state parsed into files");
    bool threw = false;
    try { dp.parse(json_parse(R"({"state": "x", "files": ["https://example.com/a.jpg"], "questions": {"q": {"type": "noul", "instructions": "?"}}})")); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "remote media URLs are rejected with a clear error");
}

int main(int argc, char** argv) {
    test_json();
    test_prompt(argc > 1 ? argv[1] : "models/d1-omni-600M-Q8_0.gguf");
    if (fails) { fprintf(stderr, "%d FAILED\n", fails); return 1; }
    printf("decision tests: ALL PASS\n");
    return 0;
}
