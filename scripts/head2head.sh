#!/bin/bash
M=models/embeddinggemma-2-Q8_0.gguf
T=2
FBS=(); LBS=(); LTPS=()
for round in 1 2 3 4; do
  out=$(./ferro-embed bench --model $M --tokens bench/tokens_16.txt --threads $T --reps 3 2>&1 | tail -1)
  ms=$(echo "$out" | grep -oP 'BEST \K[0-9.]+')
  FBS+=($ms); echo "round $round ferro: $ms ms"; sleep 12
  out=$(./baselines/llama.cpp/build/bin/llama-bench -m $M -p 188 -n 0 -r 3 -t $T 2>/dev/null | grep pp188)
  tps=$(echo "$out" | awk -F'|' '{print $(NF-1)}' | tr -d ' ' | cut -d'±' -f1)
  LTPS+=($tps); LBS+=($(python3 -c "print(round(188/$tps,1))")); echo "round $round llama: ${LBS[-1]} ms ($tps tok/s)"; sleep 12
done
python3 - "${FBS[@]}" -- "${LBS[@]}" -- "${LTPS[@]}" <<'PYEOF'
import sys
a=sys.argv[1:]; f=a[:a.index('--')]; a=a[a.index('--')+1:]; l=a[:a.index('--')]; t=a[a.index('--')+1:]
f=list(map(float,f)); l=list(map(float,l)); t=list(map(float,t))
print(f"ferro: BEST {min(f):.1f} ms avg {sum(f)/len(f):.1f} ms | {188000/(sum(f[1:])/3):.0f} tok/s (excl warmup)")
print(f"llama: BEST {min(l):.1f} ms avg {sum(l)/len(l):.1f} ms | {sum(t[1:])/3:.1f} tok/s (excl warmup)")
print(f"SPEEDUP ferro vs llama: {sum(l[1:])/sum(f[1:]):.2f}x")
PYEOF
