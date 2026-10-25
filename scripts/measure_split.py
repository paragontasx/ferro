#!/usr/bin/env python3
"""Where does a warm long-context turn actually spend its time?

Sends a 15k-token prompt twice against a warm chain and prints the server's own per-phase split
(queue wait, disk read, upload, restore, tail prefill, TTFT) plus the wall time. Run for each block size, since block
granularity sets how much of the prompt falls below the last boundary and has to be re-prefilled.

Usage: measure_split.py <block_size> [<snapshot_dir>]
A directory that already holds a chain in that block size skips the cold warm-up pass.
"""
import json, os, sys, subprocess, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("FERRO_BIN") or os.path.join(ROOT, "ferro-embed")
MODEL = os.path.join(ROOT, "models", "LFM2.5-350M-Q8_0.gguf")
SENT = ("The homelab runs Jellyfin for four rooms, an R/Shiny stack for analysis, and a local inference engine on an "
        "Intel N100 with 16 GB of RAM. Prompts are cached as blocks so a repeated prefix is never recomputed. ")
MSG = [{"role": "system", "content": SENT * 267}, {"role": "user", "content": "Name one file system."}]
BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DIR = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ferro-split-b%d" % BLOCK
PORT = 8707


def start():
    p = subprocess.Popen([BIN, "serve", "--lm", MODEL, "--lm-ctx", "16384", "--lm-slots", "2", "--port", str(PORT),
                          "--snapshot-dir", DIR, "--snapshot-mb", "4096", "--block-cache", str(BLOCK)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(1200):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=1).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server died")
            time.sleep(0.25)
    raise RuntimeError("no server")


def chat(n=8):
    r = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                               data=json.dumps({"messages": MSG, "max_tokens": n, "temperature": 0,
                                                "cache_prompt": True}).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    o = json.load(urllib.request.urlopen(r, timeout=1800))
    return time.time() - t0, o


def health():
    return json.load(urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=10)).get("block_cache") or {}


p = start()
try:
    warm = (health().get("entries") or 0) > 0
    print("block %d, dir %s (%s)" % (BLOCK, DIR, "already warm" if warm else "cold: warming now"))
    if not warm:
        t, o = chat()
        print("  cold pass: %.1fs, %d tokens" % (t, o["usage"]["prompt_tokens"]))
        for _ in range(600):
            if (health().get("saved") or 0) > 0:
                break
            time.sleep(0.25)
        time.sleep(2.0)
    for i in range(2):
        t, o = chat()
        cs = o.get("cache_split") or {}
        u = o["usage"]
        print("  warm %d: wall %6.2fs | ttft %6.1f ms = queue %6.1f + disk %5.1f + upload %5.1f + restore %5.1f + prefill %5.1f"
              % (i + 1, t, cs.get("ttft_ms", -1), cs.get("queue_ms", -1), cs.get("disk_ms", -1),
                 cs.get("upload_ms", -1), cs.get("restore_ms", -1), cs.get("prefill_ms", -1)))
        print("          cached %d of %d, %d blocks in the store, %.1f MB"
              % (u["prompt_tokens_details"]["cached_tokens"], u["prompt_tokens"], health().get("entries") or 0,
                 health().get("mb") or 0))
finally:
    p.terminate(); p.wait(timeout=60)
