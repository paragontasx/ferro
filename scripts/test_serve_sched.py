#!/usr/bin/env python3
"""Scheduling test: a long prompt is prefilled in slices, so a d1 request, a short chat and a running decode stream are not blocked behind it,
and the sliced prompt still decodes to exactly the tokens it gives when the server is otherwise idle.
usage: test_serve_sched.py [--url URL]   (server: --lm <chat gguf> --d1 <d1 gguf> --lm-slots 4)"""
import argparse, json, sys, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8090"); ap.add_argument("--no-d1", action="store_true"); a = ap.parse_args()
U = a.url
fails = 0
def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name, info); fails += not ok
def post(path, b):
    r = urllib.request.Request(U + path, json.dumps(b).encode(), {"Content-Type": "application/json"}); t = time.perf_counter()
    o = json.loads(urllib.request.urlopen(r, timeout=600).read()); return (time.perf_counter() - t) * 1e3, o
def chat(u, n=8):
    ms, o = post("/v1/chat/completions", {"messages": [{"role": "user", "content": u}], "max_tokens": n, "temperature": 0, "cache_prompt": False})
    return ms, o["choices"][0]["message"]["content"], o["usage"]["prompt_tokens"]
words = "The committee reviewed the quarterly numbers for region {} and found them consistent. "
long_prompt = "Summarise: " + "".join(words.format(i) for i in range(100))
Q = {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}
d1 = lambda: post("/v1/systemone", {"state": "I was charged twice, please refund one.", "questions": Q})[0]

idle_ms, idle_text, ntok = chat(long_prompt, 24)
alone_chat = sorted(chat("Say hi.")[0] for _ in range(3))[1]
alone_d1 = None if a.no_d1 else sorted(d1() for _ in range(3))[1]

gaps = []
def streamer():
    r = urllib.request.Request(U + "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "Count from 1 to 200 slowly."}], "max_tokens": 150, "temperature": 0, "stream": True}).encode(), {"Content-Type": "application/json"})
    last = None
    with urllib.request.urlopen(r, timeout=600) as f:
        for line in f:
            if line.startswith(b"data: {"):
                now = time.perf_counter()
                if last is not None: gaps.append((now - last, now))
                last = now
lat = {}
ts = threading.Thread(target=streamer); ts.start(); time.sleep(0.8)
t0 = time.perf_counter()
def long_req(): lat["long"] = chat(long_prompt, 24)
tl = threading.Thread(target=long_req); tl.start(); time.sleep(0.3)
if not a.no_d1: lat["d1"] = d1()
lat["chat"] = chat("Say hi.")[0]
tl.join(); t1 = time.perf_counter(); ts.join()
during = [g * 1e3 for g, t in gaps if t0 <= t <= t1]
print(f"  {ntok}-token prompt: idle {idle_ms:.0f} ms, under load {lat['long'][0]:.0f} ms; short chat {lat['chat']:.0f} ms (alone {alone_chat:.0f})" + ("" if a.no_d1 else f"; d1 {lat['d1']:.0f} ms (alone {alone_d1:.0f})"))
print(f"  decode stream while the prompt filled: {len(during)} tokens, max gap {max(during, default=0):.0f} ms")
check("sliced prompt decodes to exactly the idle-server tokens", lat["long"][1] == idle_text)
check("a decode stream keeps producing tokens during the prefill", len(during) >= 8, f"{len(during)} tokens")
check("no decode stall longer than ~1.5 s", max(during, default=0) < 1500, f"{max(during, default=0):.0f} ms")
if not a.no_d1: check("d1 request is not queued behind the whole prefill", lat["d1"] < max(1000, 8 * (alone_d1 or 0)), f"{lat['d1']:.0f} ms")
check("short chat is not queued behind the whole prefill", lat["chat"] < 0.8 * lat["long"][0], f"{lat['chat']:.0f} vs {lat['long'][0]:.0f} ms")
h = json.loads(urllib.request.urlopen(U + "/health").read())["lm"]
check("health reports prefilling", "prefilling" in h and h["prefilling"] == 0 and h["active"] == 0, str({k: h[k] for k in ("active", "prefilling", "queued")}))
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
