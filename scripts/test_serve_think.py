#!/usr/bin/env python3
"""Thinking budget / mid-think warning / ignore_eos. Needs the Thinking model: serve --lm LFM2.5-1.2B-Thinking-Q8_0.gguf
usage: test_serve_think.py [--url URL]"""
import argparse, json, sys, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8090"); U = ap.parse_args().url
fails = 0
def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name, info); fails += not ok
def req(b, stream=False):
    r = urllib.request.Request(U + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=600)
def chat(extra, n=400, user="What is 17 * 23? Answer briefly."):
    b = {"messages": [{"role": "user", "content": user}], "max_tokens": n, "temperature": 0}; b.update(extra)
    return json.loads(req(b).read())
base = chat({}, 600)
bm = base["choices"][0]["message"]
check("unbounded thinking: reasoning_content + answer, no warning", bm.get("reasoning_content") and "391" in bm["content"] and "warnings" not in base, f"{base['usage']['completion_tokens']} tokens")
o = chat({"thinking_budget": 24}, 600)
check("thinking_budget closes the reasoning block", len(o["choices"][0]["message"].get("reasoning_content", "")) < len(bm["reasoning_content"]) // 2
      and any("thinking_budget=24" in w for w in o.get("warnings", [])), str(o.get("warnings")))
o0 = chat({"thinking_budget": 0}, 60)
check("thinking_budget 0 leaves no reasoning", not o0["choices"][0]["message"].get("reasoning_content"), str(o0.get("warnings")))
o = chat({}, 20)
check("max_tokens inside <think>: empty content + explicit warning", o["choices"][0]["finish_reason"] == "length" and o["choices"][0]["message"]["content"] == ""
      and any("inside the reasoning block" in w for w in o.get("warnings", [])), str(o.get("warnings")))
# the same warning on the last streamed chunk
last = None
with req({"messages": [{"role": "user", "content": "What is 17 * 23?"}], "max_tokens": 16, "temperature": 0, "stream": True}) as f:
    for line in f:
        if line.startswith(b"data: {"):
            d = json.loads(line[6:])
            if d["choices"] and d["choices"][0].get("finish_reason"): last = d
check("streaming: warning on the finish chunk", last is not None and "warnings" in last, str(last and last.get("warnings")))
# ignore_eos keeps going past the end token (benchmarks need fixed-length runs): a reply that ends by itself after N tokens must run on to N+30
o = chat({}, 600, user="Reply with the single word: pong")
n_stop = o["usage"]["completion_tokens"]
check("without ignore_eos the reply stops at the end token", o["choices"][0]["finish_reason"] == "stop" and n_stop < 600, f"{n_stop} tokens")
o = chat({"ignore_eos": True}, n_stop + 30, user="Reply with the single word: pong")
check("ignore_eos runs past the natural end to max_tokens", o["usage"]["completion_tokens"] == n_stop + 30 and o["choices"][0]["finish_reason"] == "length", f"{o['usage']['completion_tokens']} vs natural {n_stop}")
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
