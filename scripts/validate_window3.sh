#!/bin/bash
# window3: the gate for re-enabling tail snapshots - test_serve_prefix with tails ON, full logs, plus a conversation that uses them.
set -u
cd /opt/data/ferro || exit 3
L=/opt/data/cache/scratch/window3.log; P1=/opt/data/cache/scratch/window3_prefix.log; P2=/opt/data/cache/scratch/window3_small.log
: > "$L"
say() { echo "== $*" >> "$L"; }
pkill -TERM -f 'ferro-embed ser[v]e'; sleep 5
say "start $(date +%H:%M:%S) head=$(git rev-parse --short HEAD) tails=ON cache=128MB f32"
FERRO_KV=f32 ./ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --lm-slots 4 --lm-ctx 4096 \
  --prefix-cache-mb 128 --host 127.0.0.1 --port 8096 >> "$L" 2>&1 &
P=$!
for i in $(seq 1 90); do curl -sf -m2 127.0.0.1:8096/health >/dev/null 2>&1 && break; sleep 2; done
say "test server: $(curl -s -m2 127.0.0.1:8096/health | head -c 55)"
timeout 480 python3 scripts/test_serve_prefix.py --url http://127.0.0.1:8096 > "$P1" 2>&1
say "prefix suite exit=$? pass=$(grep -ciE '^PASS' "$P1") fail=$(grep -ciE '^FAIL' "$P1")"
grep -iE '^FAIL' "$P1" | cut -c1-150 | head -4 >> "$L"
timeout 480 python3 scripts/test_serve_prefix.py --url http://127.0.0.1:8096 --small-cache > "$P2" 2>&1
say "small-cache exit=$? pass=$(grep -ciE '^PASS' "$P2") fail=$(grep -ciE '^FAIL' "$P2")"
grep -iE '^FAIL' "$P2" | cut -c1-150 | head -4 >> "$L"
say "conversation+health: $(timeout 300 python3 - http://127.0.0.1:8096 2>&1 <<'PY'
import json, sys, urllib.request
B, H = sys.argv[1], sys.argv[1] + "/health"
sysmsg = "You are a terse operations assistant. " * 40
msgs = [{"role": "system", "content": sysmsg}]
rows = []
for turn in range(5):
    msgs.append({"role": "user", "content": "Turn %d: name item %d in three words." % (turn, turn + 1)})
    r = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps({"messages": msgs, "max_tokens": 16, "temperature": 0}).encode(),
                               headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    u = d.get("usage", {}); pd = u.get("prompt_tokens_details", {}) or {}
    rows.append("t%d %s/%s" % (turn + 1, pd.get("cached_tokens"), u.get("prompt_tokens")))
    msgs.append({"role": "assistant", "content": d["choices"][0]["message"].get("content") or "ok"})
h = json.load(urllib.request.urlopen(H, timeout=30))["lm"]["prefix_cache"]
print("%s | entries %s %.1fMB hits %s evict %s" % (", ".join(rows), h["entries"], h["mb"], h["hits"], h["evictions"]))
PY
)"
kill $P 2>/dev/null; wait $P 2>/dev/null
say "end $(date +%H:%M:%S) $(grep oom_kill /sys/fs/cgroup/memory.events | head -1)"
exec ./scripts/serve_prod.sh
