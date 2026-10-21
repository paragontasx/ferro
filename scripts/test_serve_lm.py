#!/usr/bin/env python3
"""Tests for the language-model endpoints of `ferro-embed serve --lm ...` (stdlib only).

  ferro-embed serve --lm models/LFM2.5-1.2B-Thinking-Q8_0.gguf --port 8088 [--model EG2.gguf]
  python3 scripts/test_serve_lm.py --url http://127.0.0.1:8088 [--llama http://127.0.0.1:8092]

--llama: also check the native chat-template port against llama.cpp's /apply-template (llama-server started with --jinja).
"""
import argparse, concurrent.futures as cf, http.client, json, socket, sys, time, urllib.parse, urllib.request

ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8088"); ap.add_argument("--llama"); ap.add_argument("--no-embed", action="store_true")
A = ap.parse_args(); U = urllib.parse.urlparse(A.url)
fails = []

def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (("  " + str(extra)) if extra != "" else ""), flush=True)
    if not ok: fails.append(name)

def post(path, body, raw=False, timeout=600):
    data = body if isinstance(body, (bytes, str)) else json.dumps(body)
    req = urllib.request.Request(A.url + path, data.encode() if isinstance(data, str) else data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, (r.read() if raw else json.loads(r.read()))
    except urllib.error.HTTPError as e:
        b = e.read()
        try: return e.code, json.loads(b)
        except Exception: return e.code, b

def stream(path, body, timeout=600):
    c = http.client.HTTPConnection(U.hostname, U.port, timeout=timeout)
    c.request("POST", path, json.dumps(dict(body, stream=True, stream_options={"include_usage": True})), {"Content-Type": "application/json"})
    r = c.getresponse(); deltas = []; fin = None; usage = None; done = False
    for raw in r:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: "): continue
        p = line[6:]
        if p == "[DONE]": done = True; break
        d = json.loads(p)
        if d.get("usage"): usage = d["usage"]
        for ch in d["choices"]:
            dl = ch.get("delta", ch)
            for k in ("content", "reasoning_content", "text"):
                if dl.get(k): deltas.append((k, dl[k]))
            if ch.get("finish_reason"): fin = ch["finish_reason"]
    c.close()
    return r.status, deltas, fin, usage, done

def health(): return json.loads(urllib.request.urlopen(A.url + "/health", timeout=10).read())
chat = lambda content, **kw: dict({"messages": [{"role": "user", "content": content}]}, **kw)

# ---------------------------------------------------------------- basics
h = health(); check("health reports the language model", "lm" in h and h["lm"]["slots"] >= 1, h.get("lm"))
slots = h["lm"]["slots"]
m = json.loads(urllib.request.urlopen(A.url + "/v1/models").read())["data"]
check("/v1/models lists a chat model", any("chat" in x.get("capabilities", []) for x in m), [x["id"] for x in m])

s, r = post("/v1/chat/completions", chat("Say OK.", max_tokens=300, temperature=0))
ok = s == 200 and r["object"] == "chat.completion" and r["choices"][0]["message"]["role"] == "assistant"
check("chat completion shape", ok)
u = r["usage"]; check("usage adds up", u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and 0 < u["completion_tokens"] <= 300, u)
check("timings present", r["timings"]["predicted_per_second"] > 0, r["timings"])
msg = r["choices"][0]["message"]; check("answer is non-empty (reasoning split from content)", len(msg["content"].strip()) > 0 or len(msg.get("reasoning_content", "")) > 0)

s, r = post("/v1/chat/completions", chat("Count to 100 slowly.", max_tokens=7, temperature=0))
check("max_tokens is exact and finish_reason=length", r["usage"]["completion_tokens"] == 7 and r["choices"][0]["finish_reason"] == "length", r["usage"])

# ---------------------------------------------------------------- determinism & stream == non-stream
body = chat("Write one sentence about the sea.", max_tokens=60, temperature=0.8, seed=1234)
s1, r1 = post("/v1/chat/completions", body); s2, r2 = post("/v1/chat/completions", body)
get = lambda r: (r["choices"][0]["message"].get("reasoning_content", ""), r["choices"][0]["message"]["content"])
check("same seed -> same output", get(r1) == get(r2))
s3, r3 = post("/v1/chat/completions", dict(body, seed=999))
check("different seed -> (usually) different output", get(r3) != get(r1))
st, deltas, fin, usage, done = stream("/v1/chat/completions", body)
rs = "".join(t for k, t in deltas if k == "reasoning_content"); cs = "".join(t for k, t in deltas if k == "content")
check("stream == non-stream (same seed)", (rs, cs) == get(r1))
check("stream ends with finish_reason, usage chunk and [DONE]", fin in ("stop", "length") and usage and done and usage["completion_tokens"] == r1["usage"]["completion_tokens"], (fin, usage))

# ---------------------------------------------------------------- stop strings / completions
s, r = post("/v1/completions", {"prompt": "Count: 1, 2, 3, 4,", "max_tokens": 40, "temperature": 0, "stop": ["6"]})
t = r["choices"][0]["text"]; check("stop string: output cut before the match", s == 200 and "6" not in t and "5" in t and r["choices"][0]["finish_reason"] == "stop", repr(t))
s, r = post("/v1/completions", {"prompt": "Count: 1, 2, 3, 4,", "max_tokens": 30, "temperature": 0})
check("completions: no stop -> runs on", "6" in r["choices"][0]["text"], repr(r["choices"][0]["text"]))
st, deltas, fin, usage, done = stream("/v1/completions", {"prompt": "Count: 1, 2, 3, 4,", "max_tokens": 40, "temperature": 0, "stop": [", 6"]})
txt = "".join(t for _, t in deltas); check("stop string works while streaming (held-back prefix never leaks)", txt.rstrip().endswith("5") and "6" not in txt and fin == "stop", repr(txt))

# ---------------------------------------------------------------- UTF-8
st, deltas, fin, usage, done = stream("/v1/completions", {"prompt": "日本語で「ありがとう」は英語で", "max_tokens": 40, "temperature": 0})
txt = "".join(t for _, t in deltas)
check("streamed multi-byte text is intact (no U+FFFD, every delta valid UTF-8)", st == 200 and "\ufffd" not in txt and len(txt) > 0, repr(txt[:60]))
s, r = post("/v1/completions", {"prompt": "日本語で「ありがとう」は英語で", "max_tokens": 40, "temperature": 0})
check("non-stream == stream for multi-byte text", r["choices"][0]["text"] == txt)

# ---------------------------------------------------------------- errors
for name, body, want in [("invalid JSON", "{nope", 400), ("no messages", {"model": "x"}, 400), ("empty messages", {"messages": []}, 400),
                         ("bad temperature", chat("hi", temperature=-1), 400), ("n=2", chat("hi", n=2), 400), ("bad top_p", chat("hi", top_p=0), 400),
                         ("prompt longer than the context", chat("word " * 40000), 400), ("image content part", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}, 400)]:
    s, r = post("/v1/chat/completions", body)
    check(f"error: {name} -> {want}", s == want and (isinstance(r, dict) and "error" in r), (s, (r.get("error") or {}).get("message", "")[:70] if isinstance(r, dict) else r[:60]))
req = urllib.request.Request(A.url + "/v1/chat/completions"); 
try: urllib.request.urlopen(req); gs = 200
except urllib.error.HTTPError as e: gs = e.code
check("error: GET on a POST endpoint -> 405", gs == 405, gs)
check("server healthy after the error cases", health()["status"] == "ok")

# ---------------------------------------------------------------- cancellation (client disconnect frees the slot)
c = socket.create_connection((U.hostname, U.port)); payload = json.dumps(chat("Write a very long story.", max_tokens=3000, stream=True)).encode()
c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
got = b""
t0 = time.time()
while b"data:" not in got.split(b"\r\n\r\n", 1)[-1] and time.time() - t0 < 60: got += c.recv(4096)
t0 = time.time()
while health()["lm"]["active"] < 1 and time.time() - t0 < 15: time.sleep(0.1)       # wait until the request really holds a slot
active_during = health()["lm"]["active"]; c.close()
t0 = time.time()
while health()["lm"]["active"] > 0 and time.time() - t0 < 10: time.sleep(0.2)
check("client disconnect cancels generation and frees the slot", active_during >= 1 and health()["lm"]["active"] == 0, f"active {active_during} -> {health()['lm']['active']} in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------- concurrency (+ embeddings in between)
def gen(i):
    s, r = post("/v1/chat/completions", chat(f"Give me fact number {i} about Mars.", max_tokens=48, temperature=0.3, seed=i))
    return s == 200 and r["usage"]["completion_tokens"] > 0
n = slots + 2                                                    # more requests than slots: the extra ones queue
with cf.ThreadPoolExecutor(n + 1) as ex:
    futs = [ex.submit(gen, i) for i in range(n)]
    emb_ms = None
    if not A.no_embed and any("embeddings" in x.get("capabilities", []) for x in m):
        time.sleep(1.0); t0 = time.time(); s, r = post("/v1/embeddings", {"input": "an embedding while the language model is busy"}); emb_ms = (time.time() - t0) * 1000
        check("embeddings still answered while generations run", s == 200 and len(r["data"][0]["embedding"]) > 0, f"{emb_ms:.0f} ms")
    res = [f.result() for f in futs]
check(f"{n} concurrent requests ({slots} slots) all complete", all(res), res)
h = health(); check("server idle and healthy afterwards", h["lm"]["active"] == 0 and h["lm"]["queued"] == 0)

# ---------------------------------------------------------------- chat template vs llama.cpp
if A.llama:
    convs = [[{"role": "user", "content": "Hi"}],
             [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}],
             [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "<think>\nreasoning here\n</think>\n\nA1"}, {"role": "user", "content": "Q2"}],
             [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "<think>x</think>A1"}, {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "<think>y</think>A2"}, {"role": "user", "content": "Q3"}],
             [{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}]]
    # llama.cpp's /apply-template omits the leading <|startoftext|> (its server adds BOS when tokenizing); ferro's includes it
    for i, msgs in enumerate(convs):
        ref = json.loads(urllib.request.urlopen(urllib.request.Request(A.llama + "/apply-template", json.dumps({"messages": msgs}).encode(), {"Content-Type": "application/json"})).read())["prompt"]
        s, mine = post("/apply-template", {"messages": msgs})
        check(f"chat template #{i} is byte-identical to llama.cpp's (jinja)", s == 200 and mine["prompt"].replace("<|startoftext|>", "", 1) == ref, "" if mine.get("prompt", "").replace("<|startoftext|>", "", 1) == ref else (repr(mine.get("prompt"))[:140], repr(ref)[:140]))
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}]
    ref = json.loads(urllib.request.urlopen(urllib.request.Request(A.llama + "/apply-template", json.dumps({"messages": msgs, "tools": tools}).encode(), {"Content-Type": "application/json"})).read())["prompt"]
    s, mine = post("/apply-template", {"messages": msgs, "tools": tools})
    check("chat template with tools is byte-identical to llama.cpp's", s == 200 and mine["prompt"].replace("<|startoftext|>", "", 1) == ref, "" if mine.get("prompt", "").replace("<|startoftext|>", "", 1) == ref else (repr(mine.get("prompt"))[:160], repr(ref)[:160]))

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
