#!/bin/bash
# window 2: serving suites on --url, both KV dtypes, small-cache prefix, penalty A/B. Ends as production.
set -u
cd /opt/data/ferro || exit 3
LOG=/opt/data/cache/scratch/window2.log; : > "$LOG"
say() { echo "== $*" >> "$LOG"; }
pkill -TERM -f 'ferro-embed serve'; sleep 5
start() { FERRO_KV=$1 ./ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --lm-slots 4 --lm-ctx 4096 \
            --prefix-cache-mb $2 --host 127.0.0.1 --port 8096 >> "$LOG" 2>&1 & echo $!; }
waitit() { for i in $(seq 1 90); do curl -sf -m2 127.0.0.1:8096/health >/dev/null 2>&1 && return; sleep 2; done; }
say "window2 start $(date +%H:%M:%S)"

P=$(start f32 128); waitit
say "spec f32: $(timeout 420 python3 scripts/test_serve_spec.py --url http://127.0.0.1:8096 2>&1 | grep -Ei 'pass|fail|identical|error' | tail -3 | tr '\n' ' ')"
say "prefix: $(timeout 420 python3 scripts/test_serve_prefix.py --url http://127.0.0.1:8096 2>&1 | grep -Ei 'pass|fail|error|cached' | tail -3 | tr '\n' ' ')"
say "serve_lm: $(timeout 420 python3 scripts/test_serve_lm.py --url http://127.0.0.1:8096 2>&1 | grep -Ei 'pass|fail|error' | tail -3 | tr '\n' ' ')"
say "penalty A/B: $(timeout 300 python3 - http://127.0.0.1:8096 2>&1 <<'PY'
import json, sys, urllib.request
U = sys.argv[1] + "/v1/chat/completions"
IDS = ["ACCT-77120", "INV-2026-0041", "svc/ledger/apply.py", "cfg/limits.yaml",
       "ZQ-8842", "node-17b", "retry_policy=3", "READ_TIMEOUT_MS=2500"]
def run(pen):
    body = {"messages": [{"role": "system", "content": "You copy identifiers exactly as given."},
                         {"role": "user", "content": "Copy these identifiers exactly, comma separated, nothing else: " + ", ".join(IDS)}],
            "max_tokens": 80, "temperature": 0, "penalize_prompt": pen}
    r = urllib.request.Request(U, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    t = d["choices"][0]["message"].get("content") or ""
    return sum(1 for i in IDS if i in t), t.replace("\n", " ")[:90]
a, ta = run(True); b, tb = run(False)
print("default %d/%d [%s] | penalize_prompt=false %d/%d [%s]" % (a, len(IDS), ta, b, len(IDS), tb))
PY
)"
kill $P 2>/dev/null; wait $P 2>/dev/null

P=$(start int8 128); waitit
say "spec int8: $(timeout 420 python3 scripts/test_serve_spec.py --url http://127.0.0.1:8096 2>&1 | grep -Ei 'pass|fail|identical|error' | tail -3 | tr '\n' ' ')"
kill $P 2>/dev/null; wait $P 2>/dev/null

P=$(start f32 8); waitit
say "prefix small-cache: $(timeout 420 python3 scripts/test_serve_prefix.py --url http://127.0.0.1:8096 --small-cache 2>&1 | grep -Ei 'pass|fail|error|evict' | tail -3 | tr '\n' ' ')"
kill $P 2>/dev/null; wait $P 2>/dev/null

say "done $(date +%H:%M:%S) $(grep oom_kill /sys/fs/cgroup/memory.events | head -1)"
exec ./scripts/serve_prod.sh
