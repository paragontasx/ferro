#!/usr/bin/env python3
"""A stream must decode to the same tokens alone or among N concurrent streams (greedy and seeded sampling).
usage: test_batch_invariance.py [url] [--max-streams N]   (server needs --lm-slots >= max streams)"""
import json, sys, urllib.request, concurrent.futures as cf
url = next((a for a in sys.argv[1:] if a.startswith("http")), "http://127.0.0.1:8090")
maxs = int(sys.argv[sys.argv.index("--max-streams") + 1]) if "--max-streams" in sys.argv else 8


def chat(user, n=60, temp=0.0, seed=None):
    b = {"messages": [{"role": "user", "content": user}], "max_tokens": n, "temperature": temp, "cache_prompt": False}
    if seed is not None:
        b["seed"] = seed
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())["choices"][0]["message"]["content"]


T = ["Explain why the sky is blue.", "Write a short poem about a lighthouse.", "List three uses of a hash map.",
     "What is the capital of Australia and why is it not Sydney?"]
F = [f"Tell me a fact about the number {i*7+3}, in detail." for i in range(maxs)]
fails = 0
for temp, seed in ((0.0, None), (0.8, 1234)):
    solo = [chat(t, 60, temp, seed) for t in T]
    for streams in sorted({2, 3, 4, 5, 6, 8, maxs}):
        if streams > maxs:
            continue
        # fillers get different lengths so the batch composition changes while the targets decode
        jobs = [(t, 60, temp, seed) for t in T[:min(4, streams)]] + [(F[i], 20 + 9 * i, temp, 7 + i) for i in range(max(0, streams - 4))]
        with cf.ThreadPoolExecutor(len(jobs)) as ex:
            outs = list(ex.map(lambda a: chat(*a), jobs))
        ok = [outs[i] == solo[i] for i in range(min(4, streams))]
        print(f"{'ok  ' if all(ok) else 'FAIL'} temp={temp} {streams:2d} streams: {sum(ok)}/{len(ok)} identical to solo")
        fails += not all(ok)
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
