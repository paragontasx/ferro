#!/usr/bin/env python3
"""Memory planning and admission: `plan` exit codes, startup refusal / --no-memory-check, and 413 for a media request that cannot fit.
Spawns its own servers (FERRO_MEM_LIMIT_MB simulates a small container; real memory is not touched).
usage: test_memory.py [--bin ./ferro-embed] [--models models]"""
import argparse, base64, io, json, os, subprocess, sys, time, urllib.request, urllib.error, signal
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--bin", default="./ferro-embed"); ap.add_argument("--models", default="models"); a = ap.parse_args()
fails = 0
def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name, str(info)); fails += not ok
M = a.models
env0 = dict(os.environ, LD_LIBRARY_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines", "llama.cpp", "build", "bin"))
def run(args, limit=None, timeout=60):
    e = dict(env0); 
    if limit: e["FERRO_MEM_LIMIT_MB"] = str(limit)
    return subprocess.run([a.bin] + args, capture_output=True, text=True, env=e, timeout=timeout)
LM = ["--lm", f"{M}/LFM2.5-350M-Q8_0.gguf"]
r = run(["plan"] + LM, limit=4096)
check("plan: a fitting configuration exits 0 and says so", r.returncode == 0 and "fits" in r.stderr and "KV cache" in r.stderr, r.stderr.strip().splitlines()[-1][:90])
r = run(["plan"] + LM, limit=1000)
check("plan: an over-committed configuration exits 3 with advice", r.returncode == 3 and "DOES NOT FIT" in r.stderr and "try:" in r.stderr, [l for l in r.stderr.splitlines() if "try:" in l][:1])
r = run(["plan", "--lm-slots", "8", "--lm-ctx", "8192"] + LM, limit=2000)
check("plan: advice names a slot count that fits", r.returncode == 3 and "--lm-slots" in r.stderr, [l.strip() for l in r.stderr.splitlines() if "try:" in l][:2])
r = run(["serve", "--port", "8171"] + LM, limit=1000, timeout=30)
check("serve refuses to start when the plan does not fit", r.returncode == 1 and "refusing to start" in r.stderr and "--no-memory-check" in r.stderr)

def start(args, limit, port):
    e = dict(env0, FERRO_MEM_LIMIT_MB=str(limit)); log = open(os.environ.get("TMPDIR", "/tmp") + f"/mem_{port}.log", "w")
    p = subprocess.Popen([a.bin, "serve", "--port", str(port)] + args, stdout=log, stderr=subprocess.STDOUT, env=e)
    for _ in range(180):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); return p
        except Exception: time.sleep(1)
        if p.poll() is not None: break
    p.kill(); raise SystemExit("server did not start: " + open(log.name).read()[-600:])
def stop(p): p.send_signal(signal.SIGTERM); p.wait(20)
def jpeg(w, h):
    rng = np.random.default_rng(1); x = np.linspace(0, 255, w)[None, :, None] * np.ones((h, 1, 3)); n = rng.normal(0, 12, (h, w, 3))
    b = io.BytesIO(); Image.fromarray(np.clip(x + n, 0, 255).astype("uint8")).save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def post(port, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try: return 200, json.loads(urllib.request.urlopen(r, timeout=300).read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())
Q = {"cat": {"type": "noul", "instructions": "Is there a cat?"}}
D1 = ["--no-memory-check", "--d1", f"{M}/d1-omni-600M-Q8_0.gguf"]
p = start(D1, 100000, 8172)                                                                  # phase 1: how much of the container is resident with d1 loaded?
try: used = json.loads(urllib.request.urlopen("http://127.0.0.1:8172/health").read())["memory"]["used_mb"]
finally: stop(p)
LIMIT = int(used + 330)                                                                      # phase 2: a "container" with ~330 MB left (~230 MB after the 96 MB reserve)
print(f"  d1 resident: container uses {used:.0f} MB; simulated limit {LIMIT} MB")
p = start(D1, LIMIT, 8172)
try:
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8172/health").read())
    check("health reports the memory limit, usage and headroom", h["memory"]["limit_mb"] == LIMIT and h["memory"]["used_mb"] > 0 and "headroom_mb" in h["memory"], str(h["memory"]))
    st, o = post(8172, {"state": "Is this a cat?", "questions": Q})
    check("a text-only request is always admitted", st == 200, st)
    st, o = post(8172, {"state": "Look.", "files": [jpeg(256, 256)], "questions": Q})
    check("a small image fits and is admitted", st == 200, f"{st} {o if st != 200 else ''}")
    st, o = post(8172, {"state": "Look.", "files": [jpeg(1920, 1080)], "questions": Q})
    check("a 1080p whole frame (needs ~440 MB, more than is free) is rejected with 413 and guidance", st == 413 and "not enough memory" in o["error"]["message"] and "crop" in o["error"]["message"], f"{st} {o.get('error', {}).get('message', '')[:120]}")
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8172/health").read())
    check("the rejection is counted and the server stays healthy", h["memory"]["rejected"] == 1 and h["status"] == "ok", str(h["memory"]))
    st, o = post(8172, {"state": "Look.", "files": [jpeg(256, 256)], "questions": Q})
    check("requests after a rejection still work", st == 200)
finally:
    stop(p)
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
