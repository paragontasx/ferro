#!/usr/bin/env python3
"""compare two media-embedding dumps (mm_ref / ferro mm-embed format): per-chunk token count, cosine, rel. L2"""
import struct, sys, numpy as np
def load(p):
    f = open(p, 'rb'); n, = struct.unpack('<i', f.read(4)); out = []
    for _ in range(n):
        t, d = struct.unpack('<ii', f.read(8)); out.append(np.frombuffer(f.read(4 * t * d), dtype=np.float32).reshape(t, d))
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"chunks ref={len(a)} ferro={len(b)}")
worst = 1.0
for i, (x, y) in enumerate(zip(a, b)):
    if x.shape != y.shape: print(f"  chunk {i}: SHAPE ref {x.shape} ferro {y.shape}"); worst = -1; continue
    cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-30)
    rel = np.linalg.norm(x - y) / np.linalg.norm(x)
    worst = min(worst, cos.min())
    print(f"  chunk {i}: {x.shape[0]} tok  cos min {cos.min():.6f} mean {cos.mean():.6f}  relL2 {rel:.4e}")
print(f"WORST cos {worst:.6f}")
