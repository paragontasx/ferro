#!/usr/bin/env python3
"""compare two mel dumps (bench/mm/mel_ref format): int32 n_chunks, per chunk int32 n_mel, int32 n_len, f32[n_mel*n_len]"""
import struct, sys, numpy as np

def load(p):
    f = open(p, 'rb'); n, = struct.unpack('<i', f.read(4)); out = []
    for _ in range(n):
        nm, nl = struct.unpack('<ii', f.read(8))
        out.append(np.frombuffer(f.read(4 * nm * nl), dtype=np.float32).reshape(nm, nl))
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"chunks ref={len(a)} ferro={len(b)}")
bad = len(a) != len(b)
for i, (x, y) in enumerate(zip(a, b)):
    if x.shape != y.shape:
        print(f"  chunk {i}: SHAPE ref {x.shape} ferro {y.shape}"); bad = True; continue
    d = np.abs(x - y)
    print(f"  chunk {i}: {x.shape[1]} frames  max|diff| {d.max():.3e}  mean|diff| {d.mean():.3e}  "
          f"relL2 {np.linalg.norm(x - y) / np.linalg.norm(x):.3e}  (ref range {x.min():.2f}..{x.max():.2f})")
    bad = bad or d.max() > 1e-3
print("MEL MATCH" if not bad else "MEL MISMATCH")
sys.exit(1 if bad else 0)
