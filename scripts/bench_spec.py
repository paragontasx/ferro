#!/usr/bin/env python3
"""Speculative decoding speed: tokens/s with drafts on vs off on copy-heavy and free-text work (greedy), acceptance, interleaved rounds.
usage: bench_spec.py [--url URL] [--rounds N]"""
import argparse, json, statistics, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8090"); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--extra", default="{}", help="JSON merged into every request, e.g. '{\"thinking_budget\": 0}'"); a = ap.parse_args()
EXTRA = json.loads(a.extra)
def post(b):
    r = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}); t = time.perf_counter()
    o = json.loads(urllib.request.urlopen(r, timeout=900).read()); return time.perf_counter() - t, o
tickets = "\n".join(f"TICKET-{1000 + i}: customer {['Ana', 'Bo', 'Cy', 'Di'][i % 4]} reports {['a refund', 'a late parcel', 'a broken screen', 'a billing error'][i % 4]} on order {7000 + i * 13}" for i in range(14))
digest = "\n".join(f"[{9 + i % 8}:{10 + (i * 7) % 50:02d}] {['Ana', 'Bo', 'Cy'][i % 3]}: ticket TICKET-{1000 + i} moved to queue {i % 4}, owner {['Ana', 'Bo', 'Cy'][(i + 1) % 3]}" for i in range(16))
W = {
    "copy a list verbatim": (f"Repeat this list exactly, line by line, nothing else:\n{tickets}", 150),
    "extract ticket ids": (f"List only the ticket ids from this text, one per line:\n{tickets}", 80),
    "rewrite a log with a fix": (f"Rewrite this log, changing only queue 2 to queue 9. Output the whole log:\n{digest}", 200),
    "free-form answer": ("Explain in a short paragraph why the sky is blue and sunsets are red.", 100),
    "write code": ("Write a Python class for a stack with push, pop and peek methods.", 120),
}
print(f"{'workload':28} {'no drafts':>10} {'drafts':>9} {'speedup':>8}   accepted")
for name, (prompt, n) in W.items():
    res = {True: [], False: []}; acc = [0, 0]
    for _ in range(a.rounds):
        for d in (False, True):
            h0 = json.loads(urllib.request.urlopen(a.url + "/health").read())["speculative"]
            dt, o = post({"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "cache_prompt": False, "draft": d, "ignore_eos": True, **EXTRA})
            res[d].append(o["usage"]["completion_tokens"] / dt)
            h1 = json.loads(urllib.request.urlopen(a.url + "/health").read())["speculative"]
            if d: acc[0] += h1["accepted"] - h0["accepted"]; acc[1] += h1["drafted"] - h0["drafted"]
    off, on = statistics.median(res[False]), statistics.median(res[True])
    print(f"{name:28} {off:8.1f}/s {on:7.1f}/s {on/off:7.2f}x   {acc[0]}/{acc[1]} drafted tokens ({100*acc[0]/max(1,acc[1]):.0f}%)")
