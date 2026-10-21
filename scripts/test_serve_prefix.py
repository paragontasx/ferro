#!/usr/bin/env python3
"""Tests for the LM prompt-prefix cache of `ferro-embed serve --lm ...` (stdlib only).

  ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --lm-slots 4 --port 8090                # default 128 MB cache
  python3 scripts/test_serve_prefix.py --url http://127.0.0.1:8090
  ferro-embed serve --lm ... --port 8091 --prefix-cache-mb 8                                  # tiny cache: LRU eviction
  python3 scripts/test_serve_prefix.py --url http://127.0.0.1:8091 --small-cache

Digest-style workload: one long fixed system prompt + a different user message each time. Checks: the first request pays the full prefill, later ones
restore the cached prefix (cached_tokens, prompt time), answers equal those with cache_prompt=false, opt-out, learned prefixes for /v1/completions,
streaming usage, concurrency, eviction.
"""
import argparse, concurrent.futures as cf, json, sys, time, urllib.request, urllib.error

ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8090"); ap.add_argument("--small-cache", action="store_true")
A = ap.parse_args()
fails = []

def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + ((" " + str(extra)) if extra != "" else ""), flush=True)
    if not ok: fails.append(name)

def post(path, body, timeout=600):
    req = urllib.request.Request(A.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

def health(): 
    with urllib.request.urlopen(A.url + "/health", timeout=30) as r: return json.loads(r.read())["lm"]["prefix_cache"]

def chat(system, user, cache=True, n=24, tools=None, extra=None):
    b = {"messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}], "max_tokens": n, "temperature": 0, "seed": 1, "cache_prompt": cache}
    if tools: b["tools"] = tools
    if extra: b.update(extra)
    st, r = post("/v1/chat/completions", b)
    assert st == 200, (st, r)
    return r

def sysprompt(tag, words=420):
    topics = ["billing", "shipping", "returns", "security", "outages", "onboarding", "invoices", "refunds", "feedback", "accounts"]
    return (f"You are the {tag} assistant that writes a short daily digest for the operations team. Rules: be concise, never invent facts, group items by topic. "
            + " ".join(f"Topic {i}: {topics[i % len(topics)]} items must be summarised in one neutral sentence, with the ticket id in brackets." for i in range(words // 14)))

USERS = ["Ticket 4411: customer charged twice. Ticket 4412: parcel late by 3 days.", "Ticket 5001: login fails after password reset. Ticket 5002: invoice PDF is empty.",
         "Ticket 6120: refund not received after 10 days.", "Ticket 7007: app crashes on start. Ticket 7008: wrong currency shown. Ticket 7009: thanks for the quick fix!",
         "Ticket 8123: outage reported in region EU-West.", "Ticket 9001: please delete my account and all data."]

h0 = health()
check("/health reports a prefix_cache section", all(k in h0 for k in ("entries", "mb", "hits", "misses", "cached_tokens", "evictions")), h0)

if not A.small_cache:
    SYS = sysprompt("support")
    r1 = chat(SYS, USERS[0]); u1 = r1["usage"]
    check("cold request: nothing cached", u1["prompt_tokens_details"]["cached_tokens"] == 0, u1)
    n_sys = u1["prompt_tokens"]
    check("long system prompt (>= 300 tokens)", n_sys >= 300, n_sys)
    h1 = health(); check("the system prompt was stored as a prefix entry", h1["entries"] == h0["entries"] + 1 and h1["mb"] > h0["mb"], h1)
    t_cold = r1["timings"]["prompt_ms"]
    r2 = chat(SYS, USERS[1]); u2 = r2["usage"]
    cached = u2["prompt_tokens_details"]["cached_tokens"]
    check("warm request: the system prompt (not the user turn) is restored from the cache", 0.85 * n_sys < cached < u2["prompt_tokens"] and cached == r2["timings"]["cache_n"], f"cached {cached} of {u2['prompt_tokens']} tokens")
    t_warm = r2["timings"]["prompt_ms"]
    check("warm prefill is much faster than cold", t_warm < 0.4 * t_cold, f"cold {t_cold:.0f} ms -> warm {t_warm:.0f} ms ({t_cold / max(t_warm, 1):.1f}x)")
    h2 = health(); check("hit counted, no extra entry for the same prefix", h2["hits"] == h1["hits"] + 1 and h2["entries"] == h1["entries"], h2)

    # exactness: same greedy answers with and without the cache
    same = 0; diffs = []
    for u in USERS:
        a = chat(SYS, u, cache=True)["choices"][0]["message"]["content"]; b = chat(SYS, u, cache=False)["choices"][0]["message"]["content"]
        if a == b: same += 1
        else: diffs.append((a[:50], b[:50]))
    check("cached answers == uncached answers (greedy, 6 digests)", same == len(USERS), diffs if diffs else "")
    r = chat(SYS, USERS[2], cache=False); check("cache_prompt:false neither uses nor creates entries", r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and health()["entries"] == h2["entries"])

    # a different system prompt misses and becomes its own entry; the first stays usable
    SYS2 = sysprompt("sales", 380)
    r = chat(SYS2, USERS[0]); check("different system prompt: miss", r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0)
    r = chat(SYS2, USERS[1]); check("... and then hits its own entry", r["usage"]["prompt_tokens_details"]["cached_tokens"] > 150)
    r = chat(SYS, USERS[3]); check("the first system prompt still hits", r["usage"]["prompt_tokens_details"]["cached_tokens"] > 150)

    # a prompt that IS the cached prefix + one token must still work (>= 1 token has to be prefilled for the logits)
    r = chat(SYS, "?", n=8); check("short user turn after the cached prefix works", r["usage"]["prompt_tokens_details"]["cached_tokens"] > 150 and r["choices"][0]["message"]["content"] != "")

    # tools are part of the system block
    tools = [{"type": "function", "function": {"name": "get_ticket", "description": "Look up a support ticket by id", "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}}}]
    SYS3 = sysprompt("tools", 300)
    ra = chat(SYS3, USERS[0], tools=tools); rb = chat(SYS3, USERS[1], tools=tools)
    check("system prompt with tools is cached too", ra["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and rb["usage"]["prompt_tokens_details"]["cached_tokens"] > 150, rb["usage"]["prompt_tokens_details"])
    rc = chat(SYS3, USERS[1])   # same system text WITHOUT the tool list renders a different prefix -> must not reuse the tool prefix
    check("a different rendered prefix (no tools) does not reuse it", rc["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, rc["usage"]["prompt_tokens_details"])

    # learned prefixes for raw completions (no system message): the 2nd request teaches the cache, the 3rd benefits
    base = "Quarterly report. " + " ".join(f"Section {i}: the numbers for region {i} were reviewed by the committee and found consistent." for i in range(40)) + " Question: "
    outs = []
    for q in ["what grew?", "what shrank?", "what stayed flat?"]:
        st, r = post("/v1/completions", {"prompt": base + q, "max_tokens": 8, "temperature": 0}); assert st == 200, r; outs.append(r["usage"]["prompt_tokens_details"]["cached_tokens"])
    check("raw completions: shared prefix is learned (miss, miss+learn, hit)", outs[0] == 0 and outs[1] == 0 and outs[2] > 200, outs)

    # streaming reports cached tokens in the usage chunk
    body = {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": USERS[4]}], "max_tokens": 6, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(A.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    usage = None
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                o = json.loads(line[6:])
                if o.get("usage"): usage = o["usage"]
    check("streaming usage chunk carries prompt_tokens_details.cached_tokens", usage is not None and usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) > 150, usage)

    # concurrency: parallel digests sharing the prefix all answer correctly and mostly hit
    before = health()
    with cf.ThreadPoolExecutor(6) as ex: rs = list(ex.map(lambda u: chat(SYS, u), USERS))
    ref = [chat(SYS, u, cache=False)["choices"][0]["message"]["content"] for u in USERS]
    check("6 concurrent digests == uncached answers", [x["choices"][0]["message"]["content"] for x in rs] == ref)
    check("... and all of them hit the cache", health()["hits"] - before["hits"] == len(USERS), health()["hits"] - before["hits"])
    h = health(); check("cache stays within its budget", h["mb"] <= 128.0, h)
else:
    # tiny cache (e.g. 8 MB ~ 300 tokens of KV per entry at ~25 KB/token): distinct long prompts must evict, never exceed the budget
    for i in range(5):
        chat(sysprompt(f"team{i}", 120), USERS[i % len(USERS)], n=4)
    h = health()
    check("evictions happened", h["evictions"] >= 1, h)
    check("cache stays within the 8 MB budget", h["mb"] <= 8.0, h)
    r = chat(sysprompt("team4", 120), USERS[0], n=4); check("the most recent prefix is still cached", r["usage"]["prompt_tokens_details"]["cached_tokens"] > 50, r["usage"]["prompt_tokens_details"])
    r = chat(sysprompt("team0", 120), USERS[0], n=4); check("the oldest was evicted (miss, but the request still works)", r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and r["choices"][0]["message"]["content"] != "")

print("\nALL PASS" if not fails else "\nFAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)
