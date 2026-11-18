#!/usr/bin/env python3
"""Parity calibration probes.
  probe tok : ferro --print-ids vs llama /tokenize on shared texts
  probe dup : 3 identical prompts in one ferro --logits-batch run -> state leak?
  probe bos : BOS-only forward (empty prompt) vs llama
"""
import argparse, json, math, os, struct, subprocess, sys, tempfile, urllib.request

def softmax_lp(v):
    m = max(v); lse = m + math.log(sum(math.exp(x - m) for x in v)); return [x - lse for x in v]

def llama_tok(url, text):
    req = urllib.request.Request(url + "/tokenize", json.dumps({"content": text}).encode(),
        {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]

def llama_lp(url, text, topn=200):
    req = urllib.request.Request(url + "/completion", json.dumps({"prompt": text, "n_predict": 1,
        "temperature": 0, "top_k": 1, "n_probs": topn, "cache_prompt": False}).encode(),
        {"Content-Type": "application/json"})
    cp = json.loads(urllib.request.urlopen(req, timeout=600).read())["completion_probabilities"][0]["top_logprobs"]
    return {t["id"]: t["logprob"] for t in cp}

def tv(lp, ref):
    ids = list(ref)
    pr = [math.exp(ref[k]) for k in ids]; po = [math.exp(lp[k]) for k in ids]
    sr, so = sum(pr), sum(po)
    pr = [x / sr for x in pr]; po = [x / so for x in po]
    return 0.5 * sum(abs(x - y) for x, y in zip(pr, po)), so

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama", default="http://127.0.0.1:8092")
    ap.add_argument("--ferro", default=os.path.join(os.path.dirname(__file__), "..", "ferro-embed"))
    ap.add_argument("--probe", choices=["tok", "dup", "bos"], required=True)
    a = ap.parse_args()
    env = dict(os.environ, FERRO_LM_GPU="0")

    if a.probe == "tok":
        texts = ["O", "On", "Onc", "Once", "Once u", "Once upo", "Once upon a ", "Once upon a time",
                 "The capital of France is", "Q: What is the boiling point of water at sea level in Celsius?\nA:",
                 "<|im_start|>user\nExplain why the sky is blue in one sentence.<|im_end|>\n<|im_start|>assistant\n"]
        tmp = tempfile.mkdtemp()
        for i, t in enumerate(texts):
            r = subprocess.run([a.ferro, "generate", "--lm", a.model, "--raw", "--print-ids", "--n-predict", "0",
                                "--prompt", t], capture_output=True, text=True, env=env)
            line = [l for l in r.stderr.splitlines() if "prompt ids" in l]
            fid = [int(x) for x in line[0].split(":")[1].split()] if line else []
            lid = llama_tok(a.llama, t)
            print(f"text {i:2d} ({len(t):3d} ch): ferro {len(fid):3d} {fid[:6]}  llama {len(lid):3d} {lid[:6]}  "
                  f"{'SAME' if fid == lid else 'DIFF'}")
            if fid != lid:
                print("   ferro:", fid[:12]); print("   llama:", lid[:12])
        return

    if a.probe == "dup":
        tmp = tempfile.mkdtemp()
        pf = os.path.join(tmp, "dup.json")
        json.dump(["The capital of France is"] * 3, open(pf, "w"))
        lg = os.path.join(tmp, "lg")
        r = subprocess.run([a.ferro, "generate", "--lm", a.model, "--logits-batch", pf, "--dump-logits", lg],
                           capture_output=True, text=True, env=env)
        if r.returncode: sys.exit("ferro failed:\n" + r.stderr[-2000:])
        ref = llama_lp(a.llama, "The capital of France is")
        r1 = sorted(ref, key=lambda k: -ref[k])[0]
        print("ref1:", r1)
        for i in range(3):
            raw = open(f"{lg}.{i}.bin", "rb").read(); V = len(raw) // 4
            lp = softmax_lp(struct.unpack(f"<{V}f", raw))
            t, so = tv(lp, ref)
            print(f"copy {i}: TV {t:.4f} mass {so:.3f} ours1 {max(range(V), key=lp.__getitem__)}")
        b0 = open(f"{lg}.0.bin", "rb").read(); b1 = open(f"{lg}.1.bin", "rb").read(); b2 = open(f"{lg}.2.bin", "rb").read()
        print(f"determinism: 0==1 {b0 == b1}, 1==2 {b1 == b2}")
        return

    if a.probe == "bos":
        tmp = tempfile.mkdtemp()
        pf = os.path.join(tmp, "b.json")
        json.dump([""], open(pf, "w"))
        lg = os.path.join(tmp, "lg")
        r = subprocess.run([a.ferro, "generate", "--lm", a.model, "--logits-batch", pf, "--dump-logits", lg],
                           capture_output=True, text=True, env=env)
        if r.returncode: sys.exit("ferro failed:\n" + r.stderr[-2000:])
        raw = open(f"{lg}.0.bin", "rb").read(); V = len(raw) // 4
        lp = softmax_lp(struct.unpack(f"<{V}f", raw))
        ref = llama_lp(a.llama, "")
        t, so = tv(lp, ref)
        o1 = max(range(V), key=lp.__getitem__); r1 = sorted(ref, key=lambda k: -ref[k])[0]
        print(f"BOS-only: TV {t:.4f} ours1 {o1} ref1 {r1} mass {so:.3f}")
        print("ferro top-10:", [(k, round(lp[k], 3)) for k in sorted(range(V), key=lambda k: -lp[k])[:10]])
        print("ref   top-10:", [(k, round(ref[k], 3)) for k in sorted(ref, key=lambda k: -ref[k])[:10]])

main()
