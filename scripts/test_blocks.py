#!/usr/bin/env python3
"""Block-granular prompt cache: exactness, sharing, and storage on local NVMe.

Claims under test:
  1. exactness  -- a chain restored from disk produces the same tokens as a cold prefill (greedy: any drift shows)
  2. sharing    -- a sibling conversation reuses the blocks of the shared prefix instead of storing its own copy
  3. storage    -- a growing conversation costs O(unique tokens) on disk, not O(sum of prefix lengths)

Usage: test_blocks.py [--keep]
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.environ.get("FERRO_BIN", os.path.join(ROOT, "ferro-embed"))
MODEL = os.environ.get("FERRO_TEST_LM", os.path.join(ROOT, "models", "LFM2.5-350M-Q8_0.gguf"))
BLK = int(os.environ.get("FERRO_TEST_BLK", "128"))
fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (("  ->  " + str(info)) if info != "" else ""))
    if not cond:
        fails.append(name)


def start(port, snapdir, block=BLK):
    args = [BIN, "serve", "--lm", MODEL, "--lm-ctx", "2048", "--lm-slots", "2", "--port", str(port),
            "--snapshot-dir", snapdir, "--snapshot-mb", "512"]
    if block:
        args += ["--block-cache", str(block)]
    p = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(900):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server died: " + (p.stdout.read() or "")[-900:])
            time.sleep(0.2)
    raise RuntimeError("server did not come up")


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=25)
    except subprocess.TimeoutExpired:
        p.kill()


def chat(port, messages, n=24):
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps({"messages": messages, "max_tokens": n, "temperature": 0,
                                                  "cache_prompt": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    o = json.load(urllib.request.urlopen(req, timeout=300))
    u = o["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return o["choices"][0]["message"]["content"], u["prompt_tokens"], cached


def bc(port):
    return json.load(urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5)).get("block_cache") or {}


def settle(port, want=1, tries=200):
    for _ in range(tries):
        if bc(port).get("saved", 0) >= want:
            time.sleep(0.5)          # let the writer thread finish the last file
            return True
        time.sleep(0.1)
    return False


SYS = ("You are a terse assistant for a homelab. Answer in one sentence. "
       "Context: this machine runs Jellyfin, an R/Shiny stack, and a local inference engine on an Intel N100. "
       "Rules: prefer metric units, never apologise, and keep replies concrete. " * 10)

TURNS = ["What is a safe idle temperature for an N100?",
         "How much RAM should I reserve for the OS?",
         "Which filesystem suits a media library?",
         "Is ECC memory worth it for a NAS?",
         "How do I check NVMe wear?",
         "Should snapshots use zstd or zlib?"]


def convo(upto, q):
    out = [{"role": "system", "content": SYS}]
    for t in TURNS[:upto]:
        out += [{"role": "user", "content": t}, {"role": "assistant", "content": "Keep it simple."}]
    out.append({"role": "user", "content": q})
    return out


def main():
    keep = "--keep" in sys.argv
    tmp = tempfile.mkdtemp(prefix="ferro-blocks-")
    port = 8301
    try:
        # ---------------- 1. capture + exactness ----------------
        p = start(port, tmp)
        try:
            a, ntok, cached = chat(port, convo(0, TURNS[0]))
            ok = settle(port)
            check("blocks are captured while prefilling", ok, bc(port))
            check("the first request is a miss", cached == 0, "cached=%d of %d" % (cached, ntok))
            stop(p)
            p = start(port, tmp)
            b, ntok2, cached2 = chat(port, convo(0, TURNS[0]))
            check("a restarted server restores a chain from disk", bc(port).get("disk_hits", 0) >= 1, bc(port))
            check("the restored prefix is actually reused", cached2 >= BLK, "cached=%d" % cached2)
            check("restore is exact (same tokens, cold vs restored)", a == b, repr(a[:70]) + " vs " + repr(b[:70]))

            # ---------------- 2. sharing ----------------
            before = bc(port).get("entries", 0)
            c, _, cached3 = chat(port, convo(0, TURNS[1]))     # same system prompt, different question
            settle(port, before + 1)
            grew = bc(port).get("entries", 0) - before
            check("a sibling conversation reuses the shared blocks", cached3 >= BLK, "cached=%d of %d tokens" % (cached3, ntok2))
            check("a sibling stores only its own new blocks", grew <= 2, "%d new blocks (its suffix is shorter than one block)" % grew)

            # the shared-prefix answer must match a cold prefill
            stop(p)
            colddir = os.path.join(tmp, "cold")
            p2 = start(port, colddir, block=0)
            d, _, dcached = chat(port, convo(0, TURNS[1]))
            stop(p2)
            check("a sibling served from shared blocks matches a cold prefill", c == d, repr(c[:70]) + " vs " + repr(d[:70]))
            check("(control) the cold server really had no cache", dcached == 0, "cached=%d" % dcached)

            p = start(port, tmp)
        finally:
            stop(p)

        # ---------------- 3. storage for a growing conversation (own directory: only this conversation) ----------------
        p = start(port, os.path.join(tmp, "store"))
        try:
            sum_tokens = 0
            for i in range(len(TURNS)):
                _, ntok, _ = chat(port, convo(i, TURNS[i]))
                sum_tokens += ntok
            settle(port, bc(port).get("saved", 0))
            h = bc(port)
            blocks, mb = h.get("entries", 0), h.get("mb", 0.0)
            per_block = ((mb - h.get("tips_mb", 0.0)) * 2**20) / max(blocks, 1)
            per_tok = per_block / BLK
            unique = max(h.get("saved", 0), 1)
            naive = sum_tokens * per_tok
            actual = mb * 2**20                                  # blocks + tips: what the store really occupies
            print()
            print("  %d turns, %d prompt tokens summed, final prefix ~%d tokens" % (len(TURNS), sum_tokens, ntok))
            print("  blocks on disk: %d (%.2f MB, %.3f MB each = %.1f KB/token)" % (blocks, mb, per_block / 2**20, per_tok / 1024))
            print("  whole-prefix snapshots would need: %.1f MB" % (naive / 2**20))
            print("  block-chained actually uses:       %.1f MB" % (actual / 2**20))
            print("  saving: %.2fx" % (naive / actual if actual else 0))
            check("storage grew with unique tokens, not with history", actual < naive * 0.75, "%.2fx" % (naive / actual if actual else 0))
        finally:
            stop(p)

        # ---------------- 4. the tip: restart, then continue the same conversation ----------------
        tipdir = os.path.join(tmp, "tip")
        p = start(port, tipdir)
        try:
            a1, p1, _ = chat(port, convo(0, TURNS[0]))
            settle(port, 1)
            h1 = bc(port)
            check("a tip is stored for the conversation", h1.get("tips", 0) >= 1, h1)
        finally:
            stop(p)
        turn2 = convo(0, TURNS[0]) + [{"role": "assistant", "content": a1},
                                      {"role": "user", "content": TURNS[1]}]
        p = start(port, tipdir)                       # restart: only the disk tier survives
        try:
            c1, p2, cached = chat(port, turn2)
            blockonly = (p1 - 1) // BLK * BLK
            check("a continuation after a restart reuses the conversation end", cached > blockonly,
                  "cached %d of %d (blocks alone would be %d, turn-1 prompt was %d)" % (cached, p2, blockonly, p1))
        finally:
            stop(p)
        p = start(port, os.path.join(tmp, "tipcold"), block=0)
        try:
            d1, _, dcached = chat(port, turn2)
            check("the tip-served continuation matches a cold prefill", c1 == d1, repr(c1[:60]) + " vs " + repr(d1[:60]))
            check("(control) the cold server had no cache", dcached == 0, "cached=%d" % dcached)
        finally:
            stop(p)
    finally:
        if keep or fails:
            print("kept %s" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print()
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
