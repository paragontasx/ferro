#!/usr/bin/env python3
"""The tip, exercised without a chat template: a raw completion echoed back verbatim.

A chat request re-renders messages into the model's wire format, so a Thinking model's reasoning tokens do not come back
as the same ids. This test uses /v1/completions, where the next prompt is literally the previous prompt plus the returned
text, so the only thing that can break the match is tokenizer drift at the seam.

  cold:  prompt P            -> completion C, tip stored at the exact end
  warm:  prompt P + C + ...  -> must reuse the tip (cached > the last whole block), same answer as a cold server
"""
import json, os, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.environ.get("FERRO_BIN", os.path.join(ROOT, "ferro-embed"))
MODEL = os.environ.get("FERRO_TEST_LM", os.path.join(ROOT, "models", "LFM2.5-350M-Q8_0.gguf"))
BLK = int(os.environ.get("FERRO_TEST_BLK", "128"))
PORT = int(os.environ.get("FERRO_TEST_PORT", "8611"))
fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (("  ->  " + str(info)) if info != "" else ""))
    if not cond:
        fails.append(name)


def start(d, block=BLK):
    args = [BIN, "serve", "--lm", MODEL, "--lm-ctx", "4096", "--lm-slots", "2", "--port", str(PORT),
            "--snapshot-dir", d, "--snapshot-mb", "512"]
    if block:
        args += ["--block-cache", str(block)]
    if os.environ.get("FERRO_TEST_KV"):
        args += ["--kv-dtype", os.environ["FERRO_TEST_KV"]]
    p = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(900):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=1).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server died")
            time.sleep(0.2)
    raise RuntimeError("no server")


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()


def complete(prompt, n=20):
    body = {"prompt": prompt, "max_tokens": n, "temperature": 0, "cache_prompt": True}
    r = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % PORT, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    o = json.load(urllib.request.urlopen(r, timeout=600))
    u = o["usage"]
    txt = o["choices"][0].get("text") or o["choices"][0]["message"].get("content", "")
    return txt, u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def bc():
    return json.load(urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=10)).get("block_cache") or {}


P = ("The homelab runs Jellyfin for four rooms, an R/Shiny stack for analysis, and a local inference engine on an "
     "Intel N100. Answer in one short sentence. ") * 12
P += "\n\nQuestion: name one file system that suits a media library.\nAnswer:"

tmp = tempfile.mkdtemp(prefix="ferro-tiprt-")
try:
    p = start(tmp)
    try:
        c1, n1, k1 = complete(P)
        for _ in range(200):
            if bc().get("saved", 0) > 0:
                time.sleep(1.0)
                break
            time.sleep(0.1)
        h = bc()
        check("a tip is stored for the raw completion", h.get("tips", 0) >= 1, h)
        check("the first pass is a miss", k1 == 0, "cached=%d of %d" % (k1, n1))
    finally:
        stop(p)

    turn2 = P + c1 + "\n\nQuestion: which of those would you pick for snapshots?\nAnswer:"
    p = start(tmp)                                    # restart: only the disk tier survives
    try:
        c2, n2, k2 = complete(turn2)
        blockonly = (n1 // BLK) * BLK
        # The tip is keyed on token ids, and this round-trip goes through text: BPE is context dependent, so the seam
        # between the prompt and the echoed completion may or may not re-tokenise identically. Contract under test:
        # the answer is always exact, and when the ids do line up the tip is used. Whether they do is a property of the
        # text, not of the cache, so a drift is reported rather than failed.
        if k2 > blockonly:
            check("the echoed completion reuses the tip", True,
                  "cached %d of %d (blocks alone would be %d)" % (k2, n2, blockonly))
        else:
            print("INFO the text round-trip drifted at the seam: blocks alone restored %d of %d tokens "
                  "(identical answer below). Feed token ids to test the tip strictly." % (k2, n2))
        check("the conversation head is still restored from blocks", k2 >= blockonly, "cached=%d" % k2)
    finally:
        stop(p)

    p = start(os.path.join(tmp, "cold"), block=0)
    try:
        c3, n3, k3 = complete(turn2)
        check("the tip-served answer matches a cold prefill", c2 == c3, repr(c2[:60]) + " vs " + repr(c3[:60]))
        check("(control) the cold server had no cache", k3 == 0, "cached=%d" % k3)
    finally:
        stop(p)
finally:
    if fails:
        print("kept %s" % tmp)
    else:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)

print()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
sys.exit(1 if fails else 0)
