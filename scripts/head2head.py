#!/usr/bin/env python3
"""Interleaved head-to-head: ferro-embed vs llama.cpp llama-bench on N100. Cooldown between runs."""
import subprocess, time, re, sys, statistics
M = "models/embeddinggemma-2-Q8_0.gguf"
T = sys.argv[1] if len(sys.argv) > 1 else "2"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ferro, llama = [], []
for r in range(R):
    o = subprocess.run(["./ferro-embed","bench","--model",M,"--tokens","bench/tokens_16.txt",
                        "--threads",T,"--reps","3"], capture_output=True, text=True).stdout + \
        subprocess.run(["./ferro-embed","bench","--model",M,"--tokens","bench/tokens_16.txt",
                        "--threads",T,"--reps","3"], capture_output=True, text=True).stderr
    m = re.findall(r"BEST ([0-9.]+) ms", o)
    ferro.append(float(m[-1]))
    time.sleep(10)
    o = subprocess.run(["./baselines/llama.cpp/build/bin/llama-bench","-m",M,"-p","188","-n","0",
                        "-r","3","-t",T], capture_output=True, text=True).stdout
    row = [ln for ln in o.splitlines() if 'pp188' in ln][0]
    cols = [c.strip() for c in row.split('|') if c.strip()]
    tps = float(re.match(r"([0-9.]+)", cols[-1]).group(1))
    llama.append(188.0/tps*1000)
    print(f"round {r+1}: ferro {ferro[-1]:.1f} ms | llama {llama[-1]:.1f} ms ({tps:.1f} tok/s)")
    time.sleep(10)
fb, lb = min(ferro), min(llama)
fa, la = statistics.mean(ferro[1:]) if R>1 else ferro[0], statistics.mean(llama[1:]) if R>1 else llama[0]
print(f"\nferro: BEST {fb:.1f} ms avg {fa:.1f} ms -> {188000/fa:.0f} tok/s")
print(f"llama: BEST {lb:.1f} ms avg {la:.1f} ms -> {188000/la:.0f} tok/s")
print(f"SPEEDUP: {la/fa:.2f}x avg, {lb/fb:.2f}x best")
