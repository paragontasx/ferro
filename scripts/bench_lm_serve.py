#!/usr/bin/env python3
"""Concurrency benchmark for `ferro-embed serve --lm ...`: N simultaneous streaming chat requests (stdlib only).

  python3 scripts/bench_lm_serve.py --url http://127.0.0.1:8088 --conc 1 2 4 --tokens 64
Reports per concurrency level: time-to-first-token (median / max), per-stream decode rate, aggregate decode rate.
"""
import argparse, http.client, json, statistics, threading, time, urllib.parse

PROMPTS = ["Write a short poem about the ocean.", "Explain how a transistor works.", "List five uses of copper.", "Describe the water cycle.",
           "What is the difference between TCP and UDP?", "Tell me a fun fact about octopuses.", "How do vaccines work?", "Give me a recipe for pancakes."]

def one(host, port, prompt, tokens, out, i):
    c = http.client.HTTPConnection(host, port, timeout=600)
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "stream": True, "temperature": 0.3, "seed": 1 + i,
                       "stream_options": {"include_usage": True}})
    t0 = time.time(); c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse(); ttft = None; n = 0; usage = None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: "): continue
        p = line[6:]
        if p == "[DONE]": break
        d = json.loads(p)
        if d.get("usage"): usage = d["usage"]
        for ch in d["choices"]:
            dl = ch["delta"]
            if dl.get("content") or dl.get("reasoning_content"):
                if ttft is None: ttft = time.time() - t0
                n += 1
    t1 = time.time()
    out[i] = dict(ttft=ttft, wall=t1 - t0, chunks=n, tokens=(usage or {}).get("completion_tokens", n), start=t0, end=t1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8088"); ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--tokens", type=int, default=64); a = ap.parse_args()
    u = urllib.parse.urlparse(a.url)
    one(u.hostname, u.port, "Hi", 4, [None], 0)                          # warm-up
    print(f"{'conc':>4} {'TTFT med':>9} {'TTFT max':>9} {'per-stream tok/s':>17} {'aggregate tok/s':>16} {'wall s':>7}")
    for n in a.conc:
        out = [None] * n; th = [threading.Thread(target=one, args=(u.hostname, u.port, PROMPTS[i % len(PROMPTS)], a.tokens, out, i)) for i in range(n)]
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
        tot = sum(o["tokens"] for o in out)
        per = statistics.median([(o["tokens"] - 1) / max(o["end"] - o["start"] - o["ttft"], 1e-9) for o in out])
        print(f"{n:4d} {statistics.median(o['ttft'] for o in out):9.2f} {max(o['ttft'] for o in out):9.2f} {per:17.2f} {tot / wall:16.2f} {wall:7.1f}")
main()
