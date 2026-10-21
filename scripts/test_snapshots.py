#!/usr/bin/env python3
"""Persistent prefix snapshots: written by one server process, hit on the FIRST request of the next; exact; safe against corruption,
foreign (other model / dtype) snapshots and over-budget entries. Spawns its own servers.
usage: test_snapshots.py [--bin ./ferro-embed] [--lm models/LFM2.5-350M-Q8_0.gguf]"""
import argparse, glob, json, os, shutil, signal, subprocess, sys, tempfile, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--bin", default="./ferro-embed"); ap.add_argument("--lm", default="models/LFM2.5-350M-Q8_0.gguf"); a = ap.parse_args()
fails = 0
def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name, str(info)); fails += not ok
env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines", "llama.cpp", "build", "bin"))
PORT = 8181
class Srv:
    def __init__(self, d, extra=()):
        self.log = open(tempfile.mktemp(suffix=".log"), "w")
        self.p = subprocess.Popen([a.bin, "serve", "--port", str(PORT), "--lm", a.lm, "--lm-ctx", "4096", "--lm-slots", "2", "--snapshot-dir", d, *extra], stdout=self.log, stderr=subprocess.STDOUT, env=env)
        for _ in range(120):
            try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); return
            except Exception: time.sleep(1)
        raise SystemExit("server did not start: " + open(self.log.name).read()[-500:])
    def stop(self): self.p.send_signal(signal.SIGTERM); self.p.wait(30)
    def health(self): return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health").read())
    def chat(self, system, user, cache=True, n=24):
        b = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": n, "temperature": 0, "cache_prompt": cache}
        t = time.perf_counter(); o = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=300).read())
        return (time.perf_counter() - t) * 1e3, o["choices"][0]["message"]["content"], o["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0), o["usage"]["prompt_tokens"]
    def wait_saved(self, n=1, t=20):
        for _ in range(t * 5):
            h = self.health().get("prefix_snapshots", {})
            if h.get("saved", 0) >= n: return h
            time.sleep(0.2)
        return self.health().get("prefix_snapshots", {})
SYS = "You triage support tickets for a hardware shop. " + " ".join(f"Rule {i}: tickets about item {i*7} go to queue {i%5} unless the customer is {i*3} days late." for i in range(80))
U1, U2 = "Ticket: customer wants a refund for item 14, 9 days late.", "Ticket: customer asks when item 35 ships."
D = tempfile.mkdtemp(prefix="ferro-snap-")
try:
    s = Srv(D)
    ms_cold, ans_cold, c0, ntok = s.chat(SYS, U1)
    h = s.wait_saved()
    check("cold request: nothing cached; a snapshot is written to disk in the background", c0 == 0 and h.get("saved", 0) >= 1 and len(glob.glob(D + "/*.fsnap")) >= 1, f"{ntok} tokens, {ms_cold:.0f} ms, {h}")
    s.stop()
    files = sorted(glob.glob(D + "/*.fsnap")); size = sum(os.path.getsize(f) for f in files)
    s = Srv(D)
    h = s.health()["prefix_snapshots"]
    check("restart: the snapshot is found on disk", h["entries"] >= 1, h)
    ms_warm, ans_warm, c1, _ = s.chat(SYS, U2)
    _, ans_ref, c2, _ = s.chat(SYS, U2, cache=False)
    check("the FIRST request after a restart hits the snapshot", c1 >= 400 and s.health()["prefix_snapshots"]["disk_hits"] == 1, f"cached {c1} of {ntok} tokens, {ms_warm:.0f} ms vs cold {ms_cold:.0f} ms")
    check("answer from a disk-restored prefix equals the uncached answer", ans_warm == ans_ref and c2 == 0, repr(ans_warm[:60]))
    s.stop()
    # corruption: flip bytes in the middle of the payload
    f = files[0]; b = bytearray(open(f, "rb").read()); mid = len(b) // 2
    for k in range(64): b[mid + k] ^= 0xFF
    open(f, "wb").write(bytes(b))
    s = Srv(D)
    ms_c, ans_c, c3, _ = s.chat(SYS, U2)
    check("a corrupt snapshot is ignored (CRC) and the answer is still right", c3 == 0 and ans_c == ans_ref, f"cached {c3}")
    s.wait_saved(); s.stop()
    s = Srv(D); _, ans_d, c4, _ = s.chat(SYS, U2)
    check("...and it was rewritten: the next restart hits again", c4 >= 400 and ans_d == ans_ref, f"cached {c4}"); s.stop()
    # foreign identity: same directory, different KV dtype -> never loaded, never deleted
    before = sorted(os.path.basename(x) for x in glob.glob(D + "/*.fsnap"))
    s = Srv(D, ["--kv-dtype", "int8"]); h = s.health()["prefix_snapshots"]
    _, _, c5, _ = s.chat(SYS, U2)
    check("snapshots of another KV dtype are not loaded", h["entries"] == 0 and c5 == 0, h)
    s.wait_saved(); s.stop()
    after = sorted(os.path.basename(x) for x in glob.glob(D + "/*.fsnap"))
    check("...and the other dtype's files are left alone", set(before) <= set(after) and len(after) > len(before), f"{len(before)} -> {len(after)} files")
    # budget: an entry bigger than the budget is not written
    D2 = tempfile.mkdtemp(prefix="ferro-snap-small-")
    s = Srv(D2, ["--snapshot-mb", "1"]); s.chat(SYS, U1); time.sleep(1.5); h = s.health()["prefix_snapshots"]
    check("an entry larger than --snapshot-mb is not stored", h["saved"] == 0 and not glob.glob(D2 + "/*.fsnap"), h); s.stop(); shutil.rmtree(D2, ignore_errors=True)
finally:
    shutil.rmtree(D, ignore_errors=True)
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
