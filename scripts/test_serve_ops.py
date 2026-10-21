#!/usr/bin/env python3
"""Ops surface: API keys (several, labelled, file + SIGHUP reload, x-api-key), /tokenize + /detokenize (vLLM fields), /metrics (Prometheus), live rates.
Spawns its own server.   usage: test_serve_ops.py [--bin ./ferro-embed] [--lm models/LFM2.5-350M-Q8_0.gguf]"""
import argparse, json, os, re, signal, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error
ap = argparse.ArgumentParser(); ap.add_argument("--bin", default="./ferro-embed"); ap.add_argument("--lm", default="models/LFM2.5-350M-Q8_0.gguf"); a = ap.parse_args()
fails = 0
def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name, str(info)); fails += not ok
env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines", "llama.cpp", "build", "bin"))
PORT = 8191; U = f"http://127.0.0.1:{PORT}"
kf = tempfile.NamedTemporaryFile("w", suffix=".keys", delete=False); kf.write("# team keys\nbob: key-b\nkey-c-bare\n"); kf.close()
logf = tempfile.mktemp(suffix=".log")
p = subprocess.Popen([a.bin, "serve", "--port", str(PORT), "--lm", a.lm, "--lm-ctx", "2048", "--lm-slots", "2", "--api-key", "alice: key-a", "--api-key-file", kf.name], stdout=open(logf, "w"), stderr=subprocess.STDOUT, env=env)
def call(path, body=None, key=None, hdr="auth", method=None):
    h = {"Content-Type": "application/json"}
    if key: h["Authorization" if hdr == "auth" else "x-api-key"] = ("Bearer " + key) if hdr == "auth" else key
    r = urllib.request.Request(U + path, json.dumps(body).encode() if body is not None else None, h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as f: raw = f.read(); return f.status, raw
    except urllib.error.HTTPError as e: return e.code, e.read()
try:
    for _ in range(120):
        try: urllib.request.urlopen(U + "/health", timeout=2); break
        except Exception: time.sleep(1)
    chat = {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4, "temperature": 0}
    check("no key: 401 on the API", call("/v1/chat/completions", chat)[0] == 401)
    check("/health stays open without a key", call("/health")[0] == 200)
    check("--api-key 'label: key' works", call("/v1/chat/completions", chat, "key-a")[0] == 200)
    check("key from --api-key-file (labelled) works", call("/v1/chat/completions", chat, "key-b")[0] == 200)
    check("bare key from the file via x-api-key works", call("/v1/chat/completions", chat, "key-c-bare", "x")[0] == 200)
    check("a wrong key is refused", call("/v1/chat/completions", chat, "key-b-")[0] == 401 and call("/v1/chat/completions", chat, "key")[0] == 401)
    check("/metrics needs a key when keys are set", call("/metrics")[0] == 401 and call("/metrics", key="key-a")[0] == 200)
    time.sleep(0.5); log = open(logf).read()
    check("the log names which key was used", "key=alice" in log and "key=bob" in log and "key=key3" in log, [m for m in re.findall(r"key=\w+", log)][:4])
    open(kf.name, "w").write("carol: key-d\n"); p.send_signal(signal.SIGHUP); time.sleep(0.3)
    check("SIGHUP re-reads the key file: new key works, removed key stops", call("/v1/chat/completions", chat, "key-d")[0] == 200 and call("/v1/chat/completions", chat, "key-b")[0] == 401 and call("/v1/chat/completions", chat, "key-a")[0] == 200)
    K = "key-a"
    # ---- tokenize / detokenize: must match what generation tokenizes
    st, raw = call("/tokenize", {"prompt": "Hello world, this is a test."}, K); t = json.loads(raw)
    st2, raw2 = call("/v1/completions", {"prompt": "Hello world, this is a test.", "max_tokens": 1, "temperature": 0}, K); n_comp = json.loads(raw2)["usage"]["prompt_tokens"]
    check("/tokenize (prompt) matches the token count completions uses; BOS first; max_model_len", st == 200 and t["count"] == n_comp == len(t["tokens"]) and t["tokens"][0] == 1 and t["max_model_len"] == 2048, f"{t['count']} tokens")
    t2 = json.loads(call("/tokenize", {"prompt": "Hello world, this is a test.", "add_special_tokens": False}, K)[1])
    check("add_special_tokens=false drops the BOS", t2["tokens"] == t["tokens"][1:])
    st, raw = call("/tokenize", {"messages": chat["messages"]}, K); tm = json.loads(raw)
    n_chat = json.loads(call("/v1/chat/completions", chat, K)[1])["usage"]["prompt_tokens"]
    check("/tokenize (messages) applies the chat template: count equals chat's prompt_tokens", tm["count"] == n_chat, f"{tm['count']} vs {n_chat}")
    ts = json.loads(call("/tokenize", {"prompt": "Hello world", "return_token_strs": True}, K)[1])
    check("return_token_strs gives one string per token", len(ts["token_strs"]) == ts["count"] and "Hello" in "".join(ts["token_strs"]), ts["token_strs"])
    dt = json.loads(call("/detokenize", {"tokens": t2["tokens"]}, K)[1])
    check("/detokenize round-trips", dt["prompt"] == "Hello world, this is a test.", repr(dt["prompt"]))
    check("/detokenize rejects bad ids", call("/detokenize", {"tokens": [10**9]}, K)[0] == 400 and call("/tokenize", {}, K)[0] == 400)
    # ---- live rates while a stream decodes
    seen = []
    def stream():
        b = dict(chat, max_tokens=120, stream=True, messages=[{"role": "user", "content": "Count from 1 to 100."}]); b["ignore_eos"] = True
        r = urllib.request.Request(U + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + K})
        with urllib.request.urlopen(r, timeout=120) as f:
            for _ in f: pass
    th = threading.Thread(target=stream); th.start()
    while th.is_alive():
        time.sleep(0.4); seen.append(json.loads(call("/health")[1]).get("lm", {}) and json.loads(call("/health")[1])["live"]["decode_tokens_per_second"])
    th.join()
    check("/health live decode rate is reported during generation", max(seen or [0]) > 5, f"peak {max(seen or [0]):.1f} tok/s")
    time.sleep(4.0)
    check("...and drops to 0 once idle", json.loads(call("/health")[1])["live"]["decode_tokens_per_second"] == 0)
    # ---- prometheus text
    st, raw = call("/metrics", key=K); text = raw.decode()
    bad = [l for l in text.splitlines() if l and not l.startswith("#") and not re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? [-+0-9.eE]+$", l)]
    vals = {l.split(" ")[0]: float(l.split(" ")[1]) for l in text.splitlines() if l and not l.startswith("#")}
    buckets = [vals[k] for k in vals if k.startswith("ferro_lm_ttft_seconds_bucket")]
    check("/metrics is valid Prometheus text", st == 200 and not bad and "# TYPE ferro_lm_requests_total counter" in text, bad[:2])
    check("counters moved", vals["ferro_lm_requests_total"] >= 8 and vals["ferro_lm_generated_tokens_total"] > 100 and vals["ferro_http_requests_total"] > 20, {k: vals[k] for k in ("ferro_lm_requests_total", "ferro_lm_generated_tokens_total")})
    check("TTFT histogram is cumulative and ends at the count", buckets == sorted(buckets) and buckets[-1] == vals["ferro_lm_ttft_seconds_count"] > 0, buckets)
    check("memory gauges are present", vals["ferro_memory_used_bytes"] > 0 and "ferro_memory_limit_bytes" in vals)
finally:
    p.send_signal(signal.SIGTERM)
    try: p.wait(30)
    except Exception: p.kill()
    os.unlink(kf.name)
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
