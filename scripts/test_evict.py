#!/usr/bin/env python3
"""Eviction policy tests: leaves before ancestors, orphans reclaimed, TTL per class.

The discriminating case is the leaf test. With an LRU-over-everything policy (the old behaviour) a budget squeeze evicts
the oldest blocks first, and those are the *ancestors* of every chain - so the whole store goes dead at once. With
leaf-preferring eviction the newest branch's tail blocks go instead and the shared prefix still serves.
"""
import json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("FERRO_BIN") or os.path.join(ROOT, "ferro-embed")
MODEL = os.environ.get("FERRO_TEST_LM") or os.path.join(ROOT, "models", "LFM2.5-350M-Q8_0.gguf")
PASS, FAIL = 0, 0


def check(name, ok, info=""):
    global PASS, FAIL
    print(("PASS " if ok else "FAIL ") + name + (("  ->  " + str(info)) if info != "" else ""))
    if ok: PASS += 1
    else: FAIL += 1


def start(dirp, budget_mb, port, extra=(), keep=False):
    if not keep:
        subprocess.run(["rm", "-rf", dirp])
    p = subprocess.Popen([BIN, "serve", "--lm", MODEL, "--lm-ctx", "8192", "--lm-slots", "2", "--port", str(port),
                          "--snapshot-dir", dirp, "--snapshot-mb", str(budget_mb), "--block-cache", "128"] + list(extra),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(1200):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server died")
            time.sleep(0.25)
    raise RuntimeError("no server")


def start_existing(dirp, budget_mb, port):
    return start(dirp, budget_mb, port, keep=True)


def chat(port, text, n=8):
    r = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                               data=json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": n,
                                                "temperature": 0, "cache_prompt": True}).encode(),
                               headers={"Content-Type": "application/json"})
    o = json.load(urllib.request.urlopen(r, timeout=900))
    return o["usage"]["prompt_tokens_details"]["cached_tokens"], o["usage"]["prompt_tokens"]


def bc(port):
    return json.load(urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=10)).get("block_cache") or {}


def settle(port, want_saved):
    """Wait for the writer to finish: `saved` must reach want_saved and then stop moving for a second."""
    for _ in range(1200):
        if (bc(port).get("saved") or 0) >= want_saved:
            break
        time.sleep(0.25)
    last, still = -1, 0
    for _ in range(120):
        cur = bc(port).get("saved") or 0
        still = still + 1 if cur == last else 0
        last = cur
        if still >= 4:
            return True
        time.sleep(0.25)
    return False


A = ("The rack holds a switch, a NAS and a small server. " * 90)          # ~1.2k tokens
B = A + (" Extra detail that only the second conversation carries. " * 40)  # A plus a branch of its own

# ---------------- 1. leaf preference under a budget squeeze
# Measured after a restart, because within one server the in-RAM prefix cache answers first and would hide what the disk
# store actually kept.
p = start("/tmp/ferro-evict1", 30, 8711)
try:
    chat(8711, A); settle(8711, 1)
    chat(8711, B); settle(8711, 2)
    st1 = bc(8711)
finally:
    p.terminate(); p.wait(timeout=60)
check("the budget was enforced", (st1.get("evicted") or 0) > 0, "evicted=%s orphans=%s entries=%s %.1f MB"
      % (st1.get("evicted"), st1.get("orphans"), st1.get("entries"), st1.get("mb") or 0))
check("no block is left with a missing parent", (st1.get("dangling") or 0) == 0, "dangling=%s" % st1.get("dangling"))
p = start_existing("/tmp/ferro-evict1", 4096, 8714)          # rebuild from disk, no further eviction, cold RAM cache
try:
    cached, total = chat(8714, A)
    st = bc(8714)
    check("the shared ancestors survived the squeeze", cached >= 1100,
          "cached %d of %d after restart (LRU-everything cuts the chain at the evicted ancestor)" % (cached, total))
    check("the reloaded store has no dangling block either", (st.get("dangling") or 0) == 0, st.get("dangling"))
finally:
    p.terminate(); p.wait(timeout=60)

# ---------------- 2. TTL per class: tips expire, durable blocks do not
p = start("/tmp/ferro-evict2", 4096, 8712, extra=["--tip-ttl-h", "0.0056", "--block-ttl-h", "24"])   # tips: ~20 s
try:
    chat(8712, A)
    for _ in range(120):
        if (bc(8712).get("tips") or 0) > 0: break
        time.sleep(0.25)
    st = bc(8712)
    check("a tip is written", (st.get("tips") or 0) > 0, st.get("tips"))
    blocks_before = st.get("entries") or 0
    time.sleep(24)                                             # let the tip's TTL lapse
    # a save is what runs enforcement, so the trigger has to be long enough to write a block
    chat(8712, "A different conversation entirely, long enough to fill at least one block. " * 12); settle(8712, 2)
    st = bc(8712)
    # the trigger conversation writes a tip of its own, so one tip remains; what matters is that the old one was reclaimed
    # (expired counted it) instead of accumulating alongside the new one
    check("the old tip expired on its own clock", (st.get("tips") or 0) == 1 and (st.get("expired") or 0) >= 1,
          "tips=%s (just the trigger's) expired=%s" % (st.get("tips"), st.get("expired")))
    check("durable blocks survived the tip TTL", (st.get("entries") or 0) >= blocks_before, "%s -> %s" % (blocks_before, st.get("entries")))
    check("still no dangling block", (st.get("dangling") or 0) == 0, st.get("dangling"))
finally:
    p.terminate(); p.wait(timeout=60)

# ---------------- 3. an unlimited budget must not evict anything
p = start("/tmp/ferro-evict3", 4096, 8713)
try:
    chat(8713, A); settle(8713, 1)
    chat(8713, B); settle(8713, 2)
    st = bc(8713)
    check("a budget nothing reaches keeps everything", (st.get("evicted") or 0) == 0 and (st.get("entries") or 0) > 5,
          "entries=%s evicted=%s" % (st.get("entries"), st.get("evicted")))
    cached, total = chat(8713, B)
    check("and the second conversation restores from disk", cached > 0, "cached %d of %d" % (cached, total))
finally:
    p.terminate(); p.wait(timeout=60)

print()
print("ALL PASS" if FAIL == 0 else "FAILED: %d check(s)" % FAIL)
sys.exit(1 if FAIL else 0)
