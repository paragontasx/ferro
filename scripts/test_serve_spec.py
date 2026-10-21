#!/usr/bin/env python3
"""Speculative decoding (prompt lookup) must not change a single token: replies with and without drafts are identical (greedy and seeded sampling,
alone and under concurrency), and on copy-heavy prompts drafts are really accepted.
usage: test_serve_spec.py [--url URL]   (server: --lm <chat gguf> --lm-slots 4 [--kv-dtype int8]; speculation on by default)"""
import argparse, concurrent.futures as cf, json, sys, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8090"); U = ap.parse_args().url
fails = 0
def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name, str(info)); fails += not ok
def post(b):
    r = urllib.request.Request(U + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}); t = time.perf_counter()
    o = json.loads(urllib.request.urlopen(r, timeout=600).read()); return (time.perf_counter() - t), o
def chat(user, draft=True, temp=0.0, seed=None, n=120, system=None):
    b = {"messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}], "max_tokens": n, "temperature": temp, "cache_prompt": False, "draft": draft}
    if seed is not None: b["seed"] = seed
    dt, o = post(b); return o["choices"][0]["message"]["content"], dt, o["usage"]["completion_tokens"]
def health(): return json.loads(urllib.request.urlopen(U + "/health").read())["lm"] if False else json.loads(urllib.request.urlopen(U + "/health").read())
tickets = "\n".join(f"TICKET-{1000 + i}: customer {['Ana', 'Bo', 'Cy', 'Di'][i % 4]} reports {['a refund', 'a late parcel', 'a broken screen', 'a billing error'][i % 4]} on order {7000 + i * 13}" for i in range(14))
P = {
    "copy list": f"Repeat this list exactly, line by line, nothing else:\n{tickets}",
    "extract ids": f"List only the ticket ids from this text, one per line:\n{tickets}",
    "quote": "Quote this sentence three times in a row, separated by newlines: 'The quick brown fox jumps over the lazy dog near the river bank.'",
    "free text": "Write two sentences about why the sky is blue.",
    "code": "Write a Python function that returns the sum of a list, then the same function again with a docstring.",
}
h0 = health()["speculative"]
check("server reports the speculative counters", "drafted" in h0, h0)
for temp, seed, label in ((0.0, None, "greedy"), (0.8, 4242, "seeded sampling (T=0.8)")):
    same = True; details = []
    for name, p in P.items():
        a, ta, na = chat(p, True, temp, seed); b, tb, nb = chat(p, False, temp, seed)
        ok = a == b; same &= ok; details.append(f"{name}: {'==' if ok else 'DIFF'} {na}t {na/ta:.0f} vs {nb/tb:.0f} tok/s")
        if not ok:
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))); details.append(f"   first difference at char {k}: ...{a[max(0,k-20):k+20]!r} vs ...{b[max(0,k-20):k+20]!r}")
    check(f"{label}: replies with drafts == replies without", same, "\n      " + "\n      ".join(details))
h1 = health()["speculative"]
check("drafts were really proposed and accepted on the copy-heavy prompts", h1["accepted"] > 20 and h1["drafted"] > h1["accepted"] > 0, f"{h1['accepted']}/{h1['drafted']} accepted over {h1['steps']} verification steps")
# concurrency: several streams (some copy-heavy) at once must each equal their own non-speculative run
names = ["copy list", "extract ids", "quote"]
base = {n: chat(P[n], False)[0] for n in names}
with cf.ThreadPoolExecutor(3) as ex: outs = list(ex.map(lambda n: chat(P[n], True)[0], names))
check("3 concurrent streams with drafts == their solo non-speculative runs", all(o == base[n] for o, n in zip(outs, names)), [n for o, n in zip(outs, names) if o != base[n]])
mixed = [(P["copy list"], True), (P["free text"], True), (P["copy list"], False), (P["code"], True)]
with cf.ThreadPoolExecutor(4) as ex: outs = list(ex.map(lambda a: chat(a[0], a[1])[0], mixed))
check("a mix of speculating and non-speculating streams stays exact", outs[0] == base["copy list"] and outs[2] == base["copy list"], "")
# a stop string / max_tokens boundary inside a draft window
s1, _, _ = chat(P["copy list"], False, n=37); s2, _, _ = chat(P["copy list"], True, n=37)
check("max_tokens inside a verification window cuts at exactly the same token", s1 == s2 and len(s1) > 10, f"{len(s1)} chars")
b = {"messages": [{"role": "user", "content": P["copy list"]}], "max_tokens": 120, "temperature": 0, "stop": ["TICKET-1005"], "cache_prompt": False}
o1 = post(dict(b, draft=False))[1]["choices"][0]["message"]["content"]; o2 = post(dict(b, draft=True))[1]["choices"][0]["message"]["content"]
check("a stop string inside a draft window cuts at the same place", o1 == o2 and "TICKET-1005" not in o2 and "TICKET-1004" in o2, repr(o2[-40:]))
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
