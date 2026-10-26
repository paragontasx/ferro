#!/usr/bin/env python3
"""Verify ferro's Gemma 4 chat rendering + tokenization against llama.cpp on the same GGUF.

Oracle: llama-server /apply-template (the model's own jinja template) + /tokenize (add_special=true,
which prepends <bos> because tokenizer.ggml.add_bos_token = true).
Candidate: `ferro-embed generate --print-ids` (prompt ids printed before the forward pass).

A mismatch here means the model sees a different prompt than llama.cpp would give it, which is the
class of bug that silently degrades instruction following (e.g. Gemma-3 <start_of_turn> markers).
"""
import json, subprocess, sys, urllib.request

PORT = 8098
MODEL = "models/gemma4/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"

CASES = [
    ("user only",            dict(msgs=[{"role": "user", "content": "Hello"}]),                                    []),
    ("system + user",        dict(msgs=[{"role": "system", "content": "Be brief."},
                                        {"role": "user", "content": "Hi"}]),                                       []),
    ("multi-turn",           dict(msgs=[{"role": "user", "content": "A"},
                                        {"role": "assistant", "content": "B"},
                                        {"role": "user", "content": "C"}]),                                        []),
    ("thinking off",         dict(msgs=[{"role": "user", "content": "Hello"}]),                                    ["--no-think"]),
    ("think off + system",   dict(msgs=[{"role": "system", "content": "Be brief."},
                                        {"role": "user", "content": "Hi"}]),                                       ["--no-think"]),
    ("unicode + newlines",   dict(msgs=[{"role": "user", "content": "γειά σου\n\nsecond line ✓"}]),                []),
]


def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def ferro_ids(args):
    cmd = ["./ferro-embed", "generate", "--lm", MODEL, "--n-predict", "0", "--print-ids"] + args
    env = {"FERRO_LM_GPU": "0", "PATH": "/usr/bin:/bin"}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    for line in p.stderr.splitlines():
        if line.startswith("prompt ids"):
            return [int(t) for t in line.split(":", 1)[1].split()]
    raise SystemExit(f"no ids from ferro (args={args}):\n{p.stderr[-800:]}")


def main():
    fails = 0
    for name, kw, extra in CASES:
        req = {"messages": kw["msgs"]}
        if "--no-think" in extra:                     # the model's own template drops <|think|> too
            req["chat_template_kwargs"] = {"enable_thinking": False}
        prompt = post("/apply-template", req)["prompt"]
        ref = post("/tokenize", {"content": prompt, "add_special": True})["tokens"]

        args = []
        if len(kw["msgs"]) == 1:
            args = ["--prompt", kw["msgs"][0]["content"]]
        else:
            args = ["--messages", "/tmp/g4_msgs.json"]
            json.dump({"messages": kw["msgs"]}, open("/tmp/g4_msgs.json", "w"))
        mine = ferro_ids(args + extra)

        ok = mine == ref
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:<20} n={len(ref)}")
        if not ok:
            print(f"   llama: {ref}")
            print(f"   ferro: {mine}")
            for i, (a, b) in enumerate(zip(ref, mine)):
                if a != b:
                    print(f"   first diff at {i}: llama {a} vs ferro {b}")
                    break
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}: {len(CASES) - fails}/{len(CASES)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
