#!/usr/bin/env python3
"""Compare my gemma4 forward (G4_DUMP_DIR/mine_*.bin) against llama.cpp graph dumps
(G4_DUMP_DIR/call<N>_<name>.ne<a>_<b>_<c>.bin) stage by stage."""
import glob, os, re, struct, sys

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/g4ref"
mine = {}
for f in glob.glob(os.path.join(d, "mine_*.bin")):
    name = os.path.basename(f)[5:-4]
    raw = open(f, "rb").read()
    mine[name] = struct.unpack(f"{len(raw)//4}f", raw)

ref = {}
for f in glob.glob(os.path.join(d, "call*_*.bin")):
    b = os.path.basename(f)
    m = re.match(r"call(\d+)_(.+?)\.ne(\d+)_(\d+)_(\d+)\.bin$", b)
    if not m:
        continue
    call, name, n0, n1, n2 = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
    if "view" in name or "permuted" in name or "normed" in name:
        continue
    raw = open(f, "rb").read()
    v = struct.unpack(f"{len(raw)//4}f", raw)
    # transpose [n0][n1][n2] -> sequence-major [n2][n1][n0]
    out = []
    for i2 in range(n2):
        for i1 in range(n1):
            base = (i2 * n1 + i1) * n0
            out.extend(v[base:base + n0])
    ref.setdefault(name, []).append((call, n2 * n1, out))


def best(name, want_len):
    cands = [c for c in ref.get(name, []) if len(c[2]) == want_len]
    if not cands:
        return None
    return max(cands, key=lambda c: c[0])


def rel(a, b):
    na = sum(x * x for x in a)
    nd = sum((x - y) * (x - y) for x, y in zip(a, b))
    return (nd / na) ** 0.5 if na > 0 else float("nan")


order = ["inp_scaled", "inp_per_layer"] + \
        [f"{k}-{i}" for i in range(35) for k in ("Qcur_pos", "attn_wo", "attn_out", "ffn_out", "pre_ple", "l_out")] + \
        ["result_norm", "result_output_pre"]
for name in order:
    if name not in mine:
        print(f"{name:22s} MISSING in mine")
        continue
    m = mine[name]
    r = best(name, len(m))
    if r is None:
        print(f"{name:22s} no ref with len {len(m)} (ref lens: {sorted({len(c[2]) for c in ref.get(name, [])})})")
        continue
    call, rows, rv = r
    print(f"{name:22s} call{call} rows={rows:3d}  rel-L2 {rel(m, rv):.3e}  max|d| {max(abs(x-y) for x,y in zip(m,rv)):.4e}  |mine|max {max(abs(x) for x in m):.4e}")
