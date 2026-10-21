#!/usr/bin/env python3
"""LFM2 / LFM2.5 language-model parity: next-token distributions of ferro-embed vs a llama.cpp server on the same Q8_0 GGUF.

  llama-server -m LFM2.5-1.2B-Thinking-Q8_0.gguf --port 8092 -c 8192
  python3 scripts/lm_parity.py --model models/LFM2.5-1.2B-Thinking-Q8_0.gguf --llama http://127.0.0.1:8092

For every prompt the reference gives the top-N log-probabilities of the next token (/completion n_probs); ferro dumps its full logits
(`generate --logits-batch`).  Reported per prompt: top-1 agreement, total-variation distance and Jensen-Shannon-style mean |dlogprob| over
the reference's top-N set (probabilities renormalised on that set).  Pass = top-1 agrees (or the two top tokens are within 0.05 nats in the
reference) and TV <= --tv-max for >= 90% of the prompts.
"""
import argparse, json, math, os, struct, subprocess, sys, tempfile, urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>assistant\n<think>",
    "Once upon a time, in a small village by the river, there lived a",
    "Q: What is the boiling point of water at sea level in Celsius?\nA:",
    "Translate to German: 'Good morning, how are you?' ->",
    "The three primary colors are red, blue, and",
    "import numpy as np\n\ndef softmax(x):\n    e = np.exp(x - np.max(x))\n    return e /",
    "1, 1, 2, 3, 5, 8, 13, 21, 34,",
    "<|im_start|>user\nExplain why the sky is blue in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user wants",
    ("In the history of computing, the transition from vacuum tubes to transistors was a pivotal moment. " * 40) + "The next major step was",
]


# NOTE: llama-server adds the BOS token itself. Do NOT put "<|startoftext|>" in the prompt sent to it (that gives a doubled BOS and a
# different input from ferro-embed's single BOS, which showed up as a spurious ~0.1 TV).
def softmax_lp(v):
    m = max(v); lse = m + math.log(sum(math.exp(x - m) for x in v)); return [x - lse for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama", default="http://127.0.0.1:8092")
    ap.add_argument("--ferro", default=os.path.join(os.path.dirname(__file__), "..", "ferro-embed"))
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--tv-max", type=float, default=0.06)
    ap.add_argument("--cpu", action="store_true", help="run ferro on the CPU path (FERRO_LM_GPU=0)")
    ap.add_argument("--cand-url", help="score another llama.cpp server as the candidate instead of ferro (noise-floor calibration, e.g. Q8_0 vs an F16 reference)")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    pf = os.path.join(tmp, "prompts.json"); json.dump(PROMPTS, open(pf, "w"))
    env = dict(os.environ, **({"FERRO_LM_GPU": "0"} if a.cpu else {}))
    r = None if a.cand_url else subprocess.run([a.ferro, "generate", "--lm", a.model, "--logits-batch", pf, "--dump-logits", os.path.join(tmp, "lg")], capture_output=True, text=True, env=env)
    if r is not None and r.returncode: sys.exit("ferro failed:\n" + r.stderr[-2000:])
    V = None; ok_top1 = ok_tv = 0; rows = []
    for i, p in enumerate(PROMPTS):
        req = urllib.request.Request(a.llama + "/completion", json.dumps({"prompt": p, "n_predict": 1, "temperature": 0, "top_k": 1,
                                     "n_probs": a.topn, "cache_prompt": False}).encode(), {"Content-Type": "application/json"})
        cp = json.loads(urllib.request.urlopen(req, timeout=600).read())["completion_probabilities"][0]["top_logprobs"]
        ref = {t["id"]: t["logprob"] for t in cp}
        if a.cand_url:                                              # candidate = another llama.cpp server (top-200 logprobs, missing ids -> -30)
            rq = urllib.request.Request(a.cand_url + "/completion", json.dumps({"prompt": p, "n_predict": 1, "temperature": 0, "top_k": 1,
                                        "n_probs": 200, "cache_prompt": False}).encode(), {"Content-Type": "application/json"})
            cc = {t["id"]: t["logprob"] for t in json.loads(urllib.request.urlopen(rq, timeout=600).read())["completion_probabilities"][0]["top_logprobs"]}
            V = 65536; lp = [-30.0] * V
            for k, v in cc.items(): lp[k] = v
        else:
            raw = open(os.path.join(tmp, f"lg.{i}.bin"), "rb").read(); V = len(raw) // 4
            lp = softmax_lp(struct.unpack(f"<{V}f", raw))
        ids = list(ref)
        pr = [math.exp(ref[k]) for k in ids]; po = [math.exp(lp[k]) for k in ids]
        sr, so = sum(pr), sum(po); pr = [x / sr for x in pr]; po = [x / so for x in po]
        tv = 0.5 * sum(abs(x - y) for x, y in zip(pr, po))
        ours1 = max(range(V), key=lp.__getitem__); ref_sorted = sorted(ids, key=lambda k: -ref[k]); ref1 = ref_sorted[0]
        near = ours1 == ref1 or (ours1 in ref and ref[ref1] - ref[ours1] < 0.05)
        mass = so  # fraction of our probability mass inside the reference's top-N (should be ~1)
        ok_top1 += near; ok_tv += tv <= a.tv_max
        rows.append((i, len(p), ours1 == ref1, near, tv, mass))
        print(f"prompt {i:2d} ({len(p):5d} chars): top1 {'==' if ours1 == ref1 else '!='} (ref {ref1}, ours {ours1})  TV {tv:.4f}  our mass in ref top-{a.topn}: {mass:.3f}")
    n = len(PROMPTS)
    print(f"\ntop-1 agree (within 0.05 nats): {ok_top1}/{n}   TV <= {a.tv_max}: {ok_tv}/{n}   mean TV {sum(r[4] for r in rows) / n:.4f}")
    ok = ok_top1 >= 0.9 * n and ok_tv >= 0.9 * n
    print("LM PARITY", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


main()
