#!/bin/bash
# One-shot validation window: oracle + parity (f32/int8/f16) + serve suites + #3 A/B.
# Ends by exec'ing production, so the tracked background job becomes the live server.
set -u
cd /opt/data/ferro || exit 3
LOG=/opt/data/cache/scratch/window.log
: > "$LOG"
say() { echo "== $*" >> "$LOG"; }
export FERRO_GPU=1

say "window start $(date +%H:%M:%S)  headroom $(( (4294967296 - $(cat /sys/fs/cgroup/memory.current)) / 1048576 )) MB"
pkill -TERM -f 'ferro-embed serve'; sleep 5

# ---- 1. oracle: wait for the build, then serve llama.cpp CPU on :8095
for i in $(seq 1 300); do [ -x baselines/llama.cpp/build-cpu/bin/llama-server ] && break; sleep 5; done
if [ ! -x baselines/llama.cpp/build-cpu/bin/llama-server ]; then say "ORACLE BUILD MISSING - aborting to production"; exec ./scripts/serve_prod.sh; fi
baselines/llama.cpp/build-cpu/bin/llama-server -m models/LFM2.5-350M-Q8_0.gguf -c 4096 -b 4096 -ub 4096 -t 2 \
  --host 127.0.0.1 --port 8095 >> "$LOG" 2>&1 &
ORACLE=$!
for i in $(seq 1 90); do curl -sf -m2 127.0.0.1:8095/health >/dev/null 2>&1 && break; sleep 2; done
say "oracle: $(curl -s -m2 127.0.0.1:8095/health | head -c 100)"

# ---- 2. LM parity, same Q8_0 file as the oracle, one dtype per run
for dt in f32 int8 f16; do
  out=$(FERRO_KV=$dt timeout 600 python3 scripts/lm_parity.py --model models/LFM2.5-350M-Q8_0.gguf \
        --llama http://127.0.0.1:8095 --ferro ./ferro-embed 2>&1)
  say "parity $dt: $(echo "$out" | grep -Ei 'top-1|mean TV|TV |PASS|FAIL|error' | tail -4 | tr '\n' ' | ')"
done

# ---- 3. serving suites against a throwaway instance (own port, small prefix cache)
FERRO_KV=f32 ./ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --lm-slots 8 --lm-ctx 4096 \
  --prefix-cache-mb 64 --snapshot-dir /opt/data/ferro-snapshots-test --host 127.0.0.1 --port 8096 >> "$LOG" 2>&1 &
FE=$!
for i in $(seq 1 90); do curl -sf -m2 127.0.0.1:8096/health >/dev/null 2>&1 && break; sleep 2; done
say "test server: $(curl -s -m2 127.0.0.1:8096/health | head -c 90)"
for t in test_batch_invariance test_serve_spec test_serve_lm test_serve_prefix; do
  out=$(timeout 420 python3 scripts/$t.py http://127.0.0.1:8096 2>&1)
  say "$t: $(echo "$out" | grep -Ei 'pass|fail|error|identical|ALL' | tail -3 | tr '\n' ' | ')"
done
kill $FE 2>/dev/null; wait $FE 2>/dev/null

# ---- 4. #3 A/B: six turns that extend the previous prompt, old binary vs new
for bin in /opt/data/ferro-embed.bak-20261022-1732 ./ferro-embed; do
  FERRO_KV=int8 $bin serve --lm models/LFM2.5-350M-Q8_0.gguf --lm-slots 2 --lm-ctx 4096 \
    --prefix-cache-mb 256 --host 127.0.0.1 --port 8097 >> "$LOG" 2>&1 &
  B=$!
  for i in $(seq 1 90); do curl -sf -m2 127.0.0.1:8097/health >/dev/null 2>&1 && break; sleep 2; done
  out=$(timeout 300 python3 - "http://127.0.0.1:8097" 2>&1 <<'PY'
import json, sys, time, urllib.request
U = sys.argv[1] + "/v1/chat/completions"
sysmsg = "You are a terse operations assistant. Answer in one short line. " * 40
msgs = [{"role": "system", "content": sysmsg}]
rows = []
for turn in range(6):
    msgs.append({"role": "user", "content": "Turn %d: list item %d of the checklist in five words." % (turn, turn + 1)})
    body = {"messages": msgs, "max_tokens": 24, "temperature": 0}
    r = urllib.request.Request(U, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=600)); dt = time.time() - t
    u = d.get("usage", {}); pd = u.get("prompt_tokens_details", {}) or {}
    rows.append("t%d prompt=%s cached=%s %.0fms" % (turn + 1, u.get("prompt_tokens"), pd.get("cached_tokens"), dt * 1000))
    msgs.append({"role": "assistant", "content": d["choices"][0]["message"].get("content") or "ok"})
print(" ; ".join(rows))
PY
)
  say "#3 $(basename $bin): $out"
  kill $B 2>/dev/null; wait $B 2>/dev/null
done

kill $ORACLE 2>/dev/null; wait $ORACLE 2>/dev/null
say "window end $(date +%H:%M:%S)  oom_kill $(grep oom_kill /sys/fs/cgroup/memory.events | head -1)"
say "done"
exec ./scripts/serve_prod.sh
