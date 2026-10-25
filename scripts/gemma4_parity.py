#!/usr/bin/env python3
"""Gemma 4 E2B parity: ferro vs llama.cpp, split into two phases so both fit in the 4 GiB cgroup.

  phase 1  python3 scripts/gemma4_parity.py ref  --port 8098 --out /tmp/g4ref.json
  phase 2  python3 scripts/gemma4_parity.py cmp  --ref /tmp/g4ref.json --dumps /tmp/g4lg --tv-max 0.06

Phase 1 asks a running llama-server for the top-N next-token log-probs of each prompt; phase 2
reads ferro's `generate --logits-batch` dumps (full-vocab f32, one file per prompt) and reports
top-1 agreement, total-variation distance, and mean |dlogprob| over the reference's top-N set.
"""
import argparse, json, math, os, struct, subprocess, sys, urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Q: What is the boiling point of water at sea level in Celsius?\nA:",
    "Once upon a time, in a small village by the river, there lived a",
    "The three primary colors are red, blue, and",
    "1, 1, 2, 3, 5, 8, 13, 21, 34,",
]


def softmax_lp(v):
    m = max(v)
    lse = m + math.log(sum(math.exp(x - m) for x in v))
    return [x - lse for x in v]


def phase_ref(a):
    out = []
    for i, p in enumerate(PROMPTS):
        body = json.dumps({"prompt": p, "n_predict": 1, "n_probs": a.topn, "temperature": 0.0}).encode()
        req = urllib.request.Request(a.llama.rstrip("/") + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.loads(r.read().decode())
        probs = j.get("completion_probabilities") or []
        if not probs:
            sys.exit(f"prompt {i}: no completion_probabilities in response")
        cp = probs[0].get("top_logprobs") or probs[0].get("probs") or []
        items = [[int(x["id"]), float(x["logprob"])] for x in cp]
        out.append({"prompt": p, "top": items})
        print(f"ref {i}: {len(items)} entries, top id {items[0][0]} lp {items[0][1]:.4f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}")


def phase_cmp(a):
    ref = json.load(open(a.ref))
    rows, pass_n = [], 0
    for i, r in enumerate(ref):
        fn = f"{a.dumps}.{i}.bin"
        if not os.path.exists(fn):
            sys.exit(f"missing dump {fn}")
        raw = open(fn, "rb").read()
        logits = struct.unpack(f"{len(raw)//4}f", raw)
        lp = softmax_lp(logits)
        top_ids = [t[0] for t in r["top"]]
        mine = sorted(range(len(lp)), key=lambda k: -lp[k])[: a.topn]
        top1_ok = mine[0] == top_ids[0]
        # TV over the reference top-N set, renormalised on that set
        pr = [math.exp(t[1]) for t in r["top"]]
        sr = sum(pr)
        pm = [math.exp(lp[k]) for k in top_ids]
        sm = sum(pm)
        tv = 0.5 * sum(abs(x / sr - y / sm) for x, y in zip(pr, pm))
        dmax = max(abs(t[1] - lp[t[0]]) for t in r["top"])
        rows.append((i, top1_ok, tv, dmax, top_ids[0], mine[0]))
        print(f"prompt {i}: top1 {'OK ' if top1_ok else 'DIFF'} ref {top_ids[0]} mine {mine[0]}  TV {tv:.4f}  max|dlp| {dmax:.3f}")
        if top1_ok and tv <= a.tv_max:
            pass_n += 1
    n = len(rows)
    print(f"\nPARITY {pass_n}/{n} prompts pass (top-1 agrees and TV <= {a.tv_max}); "
          f"mean TV {sum(r[2] for r in rows)/n:.4f}, worst TV {max(r[2] for r in rows):.4f}")
    return 0 if pass_n >= 0.9 * n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ref", "cmp"])
    ap.add_argument("--llama", default="http://127.0.0.1:8098")
    ap.add_argument("--out", default="/tmp/g4ref.json")
    ap.add_argument("--ref", default="/tmp/g4ref.json")
    ap.add_argument("--dumps", default="/tmp/g4lg")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--tv-max", type=float, default=0.06)
    a = ap.parse_args()
    return phase_ref(a) if a.phase == "ref" else phase_cmp(a)


if __name__ == "__main__":
    sys.exit(main())
