#!/usr/bin/env python3
"""End-to-end greedy comparison for one chat prompt: llama.cpp (same GGUF) vs ferro.

Renders the prompt with the model's own template (llama-server /apply-template), runs greedy decoding
on both sides, and compares the produced text token-prefix-wise. Thinking is on for both, so a match
means the prompt, the tokenizer and the forward pass all agree at the start of a reasoning trace.
"""
import json, subprocess, sys, urllib.request

PORT = 8098
MODEL = "models/gemma4/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
NP = 24
MSG = [{"role": "user", "content": sys.argv[1] if len(sys.argv) > 1 else
        "List three prime numbers between 10 and 30, comma separated, nothing else."}]


def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


prompt = post("/apply-template", {"messages": MSG})["prompt"]
ref = post("/completion", {"prompt": prompt, "n_predict": NP, "temperature": 0, "cache_prompt": False,
                           "stop": ["<turn|>"]})
ref_text = ref.get("content") or ""
print(f"llama.cpp ({NP} tok greedy): {ref_text!r}")

json.dump({"messages": MSG}, open("/tmp/g4_e2e_msgs.json", "w"))
env = {"FERRO_LM_GPU": "1", "PATH": "/usr/bin:/bin",
       "OCL_ICD_VENDORS": "/opt/data/neo/icd",
       "LD_LIBRARY_PATH": "/opt/data/neo/root/usr/local/lib:/opt/data/neo/root/usr/lib/x86_64-linux-gnu:"
                          "/opt/data/neo/root/usr/lib/x86_64-linux-gnu/intel-opencl"}
cmd = ["bash", "scripts/gpu_run.sh", "gemma4 e2e greedy", "420", "--", "env"] + \
      [f"{k}={v}" for k, v in env.items()] + \
      ["./ferro-embed", "generate", "--lm", MODEL, "--messages", "/tmp/g4_e2e_msgs.json",
       "--n-predict", str(NP), "--temp", "0"]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
import re as _re; mine = _re.sub(r"gpu_lock:[^\n]*", "", p.stdout).strip("\n")
print(f"ferro    ({NP} tok greedy): {mine!r}")

n = min(len(ref_text), len(mine))
common = 0
while common < n and ref_text[common] == mine[common]:
    common += 1
print(f"\ncommon prefix: {common} chars of {n} compared "
      f"({'IDENTICAL' if common == n and len(ref_text) == len(mine) else 'differs'})")
if common < n:
    print(f"  llama: {ref_text[max(0, common - 20):common + 60]!r}")
    print(f"  ferro: {mine[max(0, common - 20):common + 60]!r}")
sys.exit(0 if common == n and len(ref_text) == len(mine) else 1)
