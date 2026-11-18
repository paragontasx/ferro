#!/usr/bin/env python3
"""Prefix-length sweep for parity calibration.

For a failing prompt, compare ferro vs llama.cpp next-token logits on prefixes of
increasing length.  If TV already diverges at a one-token prefix the bug is in the
single forward (norms / gates / rope); if TV grows with prefix length the bug is
positional (attention / KV / window boundary).

  ./build/bin/llama-server -m LFM3-1.7B-Q8_0.gguf --port 8092 -c 8192   # f32-KV reference
  python3 scripts/calib_prefix.py --model models/LFM3-1.7B-Q8_0.gguf --cpu

Output: per prefix length: TV over reference top-20, top-1 agreement.
"""
import argparse, json, math, os, struct, subprocess, sys, tempfile, urllib.request

BASE_PROMPTS = {
    4: "Once upon a time, in a small village by the river, there lived a",
    5: "Q: What is the boiling point of water at sea level in Celsius?\nA:",
    10: "<|im_start|>user\nExplain why the sky is blue in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user wants",
    0: "The capital of France is",
}


def softmax_lp(v):
    m = max(v); lse = m + math.log(sum(math.exp(x - m) for x in v)); return [x - lse for x in v]


def ref_llama(url, prompt, topn=200):
    req = urllib.request.Request(url + "/completion", json.dumps({"prompt": prompt, "n_predict": 1,
         "temperature": 0, "top_k": 1, "n_probs": topn, "cache_prompt": False}).encode(),
         {"Content-Type": "application/json"})
    cp = json.loads(urllib.request.urlopen(req, timeout=600).read())["completion_probabilities"][0]["top_logprobs"]
    return {t["id"]: t["logprob"] for t in cp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama", default="http://127.0.0.1:8092")
    ap.add_argument("--ferro", default=os.path.join(os.path.dirname(__file__), "..", "ferro-embed"))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--starts", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    a = ap.parse_args()

    jobs = []   # (prompt_id, n_chars, text)
    for pid, p in BASE_PROMPTS.items():
        for k in a.starts:
            if k <= len(p):
                jobs.append((pid, k, p[:k]))

    tmp = tempfile.mkdtemp()
    pf = os.path.join(tmp, "prefixes.json")
    json.dump([t for _, _, t in jobs], open(pf, "w"))
    env = dict(os.environ, **({"FERRO_LM_GPU": "0"} if a.cpu else {}))
    lg = os.path.join(tmp, "lg")
    r = subprocess.run([a.ferro, "generate", "--lm", a.model, "--logits-batch", pf, "--dump-logits", lg],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit("ferro failed:\n" + r.stderr[-3000:])

    print(f"{'p':>2} {'chars':>5} {'TV':>7} {'top1':>6}  ours-in-ref-top20")
    bad = 0
    for idx, (pid, k, _) in enumerate(jobs):
        raw = open(f"{lg}.{idx}.bin", "rb").read()
        V = len(raw) // 4
        lp = softmax_lp(struct.unpack(f"<{V}f", raw))
        ref = ref_llama(a.llama, jobs[idx][2])
        ids = list(ref)
        pr = [math.exp(ref[x]) for x in ids]; po = [math.exp(lp[x]) for x in ids]
        sr, so = sum(pr), sum(po)
        pr = [x / sr for x in pr]; po = [x / so for x in po]
        tv = 0.5 * sum(abs(x - y) for x, y in zip(pr, po))
        ours1 = max(range(V), key=lp.__getitem__)
        ref1 = sorted(ids, key=lambda x: -ref[x])[0]
        agree = "==" if ours1 == ref1 else "!="
        bad += tv > 0.06
        print(f"{pid:2d} {k:5d} {tv:7.4f} {agree:>6}  {so:.3f}  (ours1 {ours1} ref1 {ref1})")
    print(f"\nTV>0.06 prefixes: {bad}/{len(jobs)}")


main()
