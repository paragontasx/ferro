#!/usr/bin/env python3
"""d1-omni head-to-head: ferro-embed (iGPU / CPU) vs llama.cpp llama-server (CPU / Vulkan iGPU), same GGUF, same requests.
Throughput: all requests at once (ferro: one batch; llama-server: -np 8, concurrent clients).
Latency: one request at a time. Interleaved rounds with cooldowns (N100 DVFS)."""
import json, subprocess, sys, time, os, urllib.request, concurrent.futures as cf
M = "models/d1-omni-600M-Q8_0.gguf"
REQ = sys.argv[1] if len(sys.argv) > 1 else "bench/d1/tickets16.json"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
reqs = json.load(open(REQ))
LB = "baselines/llama.cpp"

def ferro(env, seq):
    e = dict(os.environ); e.update(env)
    if seq: e["FERRO_DECISION_SEQUENTIAL"] = "1"
    p = subprocess.run(["./ferro-embed", "bench", "--model", M, "--request", REQ, "--reps", "4"], capture_output=True, text=True, env=e)
    line = [l for l in p.stderr.splitlines() if "BEST" in l]
    best_ms = float(line[0].split("BEST")[1].split("ms")[0])
    ntok = int([l for l in p.stderr.splitlines() if "question sequences" in l][0].split(",")[2].split()[0])
    return best_ms, ntok

def server_run(binpath, extra_env, np_, concurrent):
    e = dict(os.environ); e.update(extra_env)
    port = 8095
    srv = subprocess.Popen([binpath, "-m", M, "-b", "4096", "-ub", "4096", "-t", "2", "-np", str(np_), "--host", "127.0.0.1",
                            "--port", str(port), "-ngl", "99"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=e)
    try:
        for _ in range(240):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read(): break
            except Exception: pass
            time.sleep(1)
        def post(r):
            q = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(r).encode(), headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(q, timeout=1800))
        post(reqs[0])  # warmup
        best = 1e30; ntok = 0
        for rep in range(2):
            t0 = time.time()
            if concurrent:
                with cf.ThreadPoolExecutor(len(reqs)) as ex: res = list(ex.map(post, reqs))
            else:
                res = [post(r) for r in reqs]
            best = min(best, (time.time() - t0) * 1e3)
            ntok = sum(x["usage"]["input_tokens"] for x in res)
        return best, ntok
    finally:
        srv.terminate(); srv.wait()

configs = [
    ("ferro iGPU", lambda c: ferro({}, not c)),
    ("ferro CPU (2t)", lambda c: ferro({"FERRO_GPU": "0"}, not c)),
    ("llama.cpp Vulkan iGPU", lambda c: server_run(f"{LB}/build-vk/bin/llama-server", {"GGML_VK_VISIBLE_DEVICES": "0"}, 8 if c else 1, c)),
    ("llama.cpp CPU (2t)", lambda c: server_run(f"{LB}/build/bin/llama-server", {}, 8 if c else 1, c)),
]
results = {}
for rnd in range(ROUNDS):
    for mode in ("throughput", "latency"):
        for name, fn in configs:
            time.sleep(10)
            ms, ntok = fn(mode == "throughput")
            results.setdefault((name, mode), []).append(ms)
            print(f"round {rnd} {mode:10s} {name:24s} {ms:9.0f} ms  {ntok / ms * 1e3:6.0f} tok/s  ({ntok} tokens)", flush=True)
print("\nBEST")
for (name, mode), v in sorted(results.items(), key=lambda x: (x[0][1], min(x[1]))):
    print(f"{mode:10s} {name:24s} {min(v):9.0f} ms")
