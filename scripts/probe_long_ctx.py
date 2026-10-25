#!/usr/bin/env python3
"""Long-context behaviour of the block cache: ~15k tokens on the N100.

Answers the questions a long prompt raises that the short tests cannot:
  1. does a 15k prompt capture all its blocks, and how big is the chain on disk
  2. after a restart, does the chain restore and is the answer identical to a cold prefill
  3. what does the chain cost in time (restore vs re-prefill) at that length
  4. does the tip still fire, and does the capture buffer stay within its bound

Usage: probe_long_ctx.py [--tokens 15000] [--model path] [--ctx 16384] [--kv f32|int8]
"""
import argparse, json, os, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "ferro-embed")
BLK = 128

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=15000)
ap.add_argument("--model", default=os.path.join(ROOT, "models", "LFM2.5-350M-Q8_0.gguf"))
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--kv", default="f32")
ap.add_argument("--port", type=int, default=8501)
ap.add_argument("--dir", default="/tmp/ferro-longctx")
a = ap.parse_args()

SENT = ("The homelab runs Jellyfin for four rooms, an R/Shiny stack for analysis, and a local inference engine on an "
        "Intel N100 with 16 GB of RAM. Prompts are cached as blocks so a repeated prefix never has to be recomputed. ")
SYS = SENT * max(1, a.tokens // 40)          # ~40 tokens per repetition


def srv(port, fresh=False):
    if fresh:
        subprocess.run(["rm", "-rf", a.dir])
    p = subprocess.Popen([BIN, "serve", "--lm", a.model, "--lm-ctx", str(a.ctx), "--lm-slots", "1",
                          "--port", str(port), "--snapshot-dir", a.dir, "--snapshot-mb", "4096",
                          "--block-cache", str(BLK), "--kv-dtype", a.kv],
                         cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(1200):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server died")
            time.sleep(0.25)
    raise RuntimeError("no server")


def tokenize(port, text):
    r = urllib.request.Request("http://127.0.0.1:%d/tokenize" % port,
                               data=json.dumps({"prompt": text, "add_special_tokens": False}).encode(),
                               headers={"Content-Type": "application/json"})
    o = json.load(urllib.request.urlopen(r, timeout=300))
    if isinstance(o.get("count"), int): return o["count"]
    if isinstance(o.get("n_tokens"), int): return o["n_tokens"]
    return len(o.get("tokens") or o.get("ids") or [])


def size_prompt(port, target):
    """Grow the system prompt until it really is `target` tokens: a fixed repetition count overshoots."""
    text = SENT * max(1, target // 40)
    n = tokenize(port, text)
    for _ in range(6):
        if abs(n - target) <= target * 0.01:
            break
        reps = max(1, int(len(text) / len(SENT) * target / max(n, 1)))
        text = SENT * reps
        n = tokenize(port, text)
    return text, n


def chat(port, sysprompt, n=16):
    r = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                               data=json.dumps({"messages": [{"role": "system", "content": sysprompt},
                                                             {"role": "user", "content": "Name one file system, briefly."}],
                                                "max_tokens": n, "temperature": 0, "cache_prompt": True}).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        o = json.load(urllib.request.urlopen(r, timeout=1800))
    except urllib.error.HTTPError as e:
        print("   HTTP %d: %s" % (e.code, e.read().decode()[:300]))
        raise
    m = o["choices"][0]["message"]
    t = o.get("timings", {})
    u = o["usage"]
    return ((m.get("reasoning_content") or "") + (m.get("content") or ""), u["prompt_tokens"],
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), t.get("prompt_ms", 0))


def bc(port):
    return json.load(urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=10)).get("block_cache") or {}


def settle(port, tries=400):
    for _ in range(tries):
        h = bc(port)
        if h.get("saved", 0) > 0:
            time.sleep(1.0)
            return h
        time.sleep(0.25)
    return bc(port)


try:
    print("model %s | ctx %d | kv %s | block %d" % (os.path.basename(a.model), a.ctx, a.kv, BLK))
    p = srv(a.port, fresh=True)
    try:
        sysp, want = size_prompt(a.port, a.tokens)
        print("       system prompt sized to %d tokens (target %d)" % (want, a.tokens))
        ans, ntok, cached, ms = chat(a.port, sysp)
        h = settle(a.port)
        print("cold : prompt %d tok | prefill %6.0f ms (%.0f tok/s) | cached %d" % (ntok, ms, ntok / (ms / 1000 + 1e-9), cached))
        print("       blocks %d + tips %d | %.1f MB on disk (%.2f MB/block)"
              % (h.get("entries", 0), h.get("tips", 0), h.get("mb", 0.0), (h.get("blocks_mb", 0) / max(h.get("entries", 1), 1))))
        print("       answer %r" % ans[:60])
    finally:
        p.terminate(); p.wait(timeout=60)

    p = srv(a.port)                                     # restart: only disk survives
    try:
        ans2, ntok2, cached2, ms2 = chat(a.port, sysp)
        h2 = bc(a.port)
        print("warm : prompt %d tok | prefill %6.0f ms (%.0f tok/s) | cached %d | disk hits %d"
              % (ntok2, ms2, ntok2 / (ms2 / 1000 + 1e-9), cached2, h2.get("disk_hits", 0)))
        print("       re-prefilled %d tokens | saved %.1f s (%.1fx faster)" % (ntok2 - cached2, (ms - ms2) / 1000, ms / max(ms2, 1)))
        print("       identical answer: %s" % (ans == ans2))
    finally:
        p.terminate(); p.wait(timeout=60)
    total = 0
    for root, _, files in os.walk(os.path.join(a.dir, "blocks")):
        for f in files:
            if f.endswith(".blk"):
                total += os.path.getsize(os.path.join(root, f))
    print("disk: %.1f MB total for %d blocks + 1 tip" % (total / 2**20, h.get("entries", 0)))
except Exception as e:
    print("PROBE FAILED: %s" % e)
    sys.exit(1)
