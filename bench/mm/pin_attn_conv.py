#!/usr/bin/env python3
"""Pin the chunked-local-attention geometry and the causal depthwise conv against the ggml dumps (bench/mm/ggml_ops_ref).
Usage: pin_attn_conv.py OPS_DIR      Exit 0 only when all three ops match their numpy specs."""
import numpy as np, sys

D = sys.argv[1]
def load(name):
    raw = open(f"{D}/{name}.bin", "rb").read()
    return np.frombuffer(raw[:32], dtype=np.int64), np.frombuffer(raw[32:], dtype=np.float32)
def rel(a, b): return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)
ok = True

# ---- K/V block-context extraction. ggml K ne = [D, H, N] -> memory K[n][h][d]. Output ne = [D, S, B, H] -> out[h][b][s][d].
# block b, slot s holds position (b*C - P + s) of the sequence (zero when outside [0, N)).
ne, k = load("blk.k"); Dd, H, N = (int(v) for v in ne[:3]); K = k.reshape(N, H, Dd)
ne, o = load("blk.out"); _, S, B, _ = (int(v) for v in ne[:4]); O = o.reshape(H, B, S, Dd)
C, P = 12, 12
spec = np.zeros((H, B, S, Dd), dtype=np.float32)
for b in range(B):
    for s in range(S):
        pos = b * C - P + s
        if 0 <= pos < N: spec[:, b, s, :] = K[pos]
r = rel(O, spec); print(f"blocks  (K window, overlapping view)   relL2 {r:.3e}"); ok &= r < 1e-6

# ---- blocked relative shift. in ne = [R, C, B, H] -> memory in[h][b][c][r]; out ne = [S, C, B, H] -> out[h][b][c][s].
# Transformer-XL shift per block: row c of the [C][R] matrix is placed so that out[c][s] = in[c][ s - (C-1 - c) ... ] -- derive
# by brute force below from the reshape trick: flatten rows padded to S+1, drop the tail, reshape to S wide.
ne, bi = load("shift.in"); R, C2, B2, H2 = (int(v) for v in ne[:4]); BD = bi.reshape(H2, B2, C2, R)
ne, so = load("shift.out"); S2 = int(ne[0]); SO = so.reshape(H2, B2, C2, S2)
spec = np.zeros_like(SO)
for h in range(H2):
    for b in range(B2):
        padded = np.zeros((C2, S2 + 1), dtype=np.float32); padded[:, :R] = BD[h, b]     # pad each row to S+1
        flat = padded.reshape(-1)[: C2 * S2]                                              # drop the tail
        spec[h, b] = flat.reshape(C2, S2)
r = rel(SO, spec); print(f"shift   (blocked relative shift)       relL2 {r:.3e}"); ok &= r < 1e-6

# ---- causal depthwise conv. x ne = [NPOS, CH] -> x[ch][t]; w ne = [KS, CH] -> w[ch][k]; out ne = [CH, NPOS] -> out[t][ch].
# left-padded by KS-1: out[t][ch] = sum_k w[ch][k] * x[ch][t - (KS-1) + k]
ne, xx = load("ssm.x"); NP, CH = int(ne[0]), int(ne[1]); X = xx.reshape(CH, NP)
ne, ww = load("ssm.w"); KS = int(ne[0]); W = ww.reshape(CH, KS)
ne, oo = load("ssm.out"); OO = oo.reshape(int(ne[1]), int(ne[0]))                          # [t][ch]
spec = np.zeros((NP, CH), dtype=np.float32)
for t in range(NP):
    for k in range(KS):
        src = t - (KS - 1) + k
        if src >= 0: spec[t] += W[:, k] * X[:, src]
r = rel(OO, spec); print(f"ssm     (causal depthwise conv, k={KS})    relL2 {r:.3e}"); ok &= r < 1e-5

print("ALL OPS MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
