#!/usr/bin/env python3
"""Pin the Gemma4 SSCP layout against the ggml dump before writing C++: a numpy reference of
conv2d(3x3, s2, p1) -> LayerNorm over channels (weight, no bias) -> ReLU, twice, then flatten (ch, freq) -> [time][ch*freq].
Compares with bench/mm/ggml_ops_ref output. Prints the max diff; a layout mistake shows up as O(1) error."""
import numpy as np, sys

D = sys.argv[1]
IM2COL_F16 = not (len(sys.argv) > 2 and sys.argv[2] == 'f32')   # ggml_conv_2d rounds BOTH the im2col patches and the kernel to fp16
def load(name):
    raw = open(f"{D}/{name}.bin", "rb").read()
    ne = np.frombuffer(raw[:32], dtype=np.int64)
    return ne, np.frombuffer(raw[32:], dtype=np.float32)

ne_x, x = load("sscp.in")          # ggml ne = [F(mel, fast), T(time), 1]
ne_w0, w0 = load("sscp.w0")        # ggml ne = [3(kw), 3(kh), 1(cin), 128(cout)]
ne_w1, w1 = load("sscp.w1")        # ggml ne = [3, 3, 128, 32]
_, n0 = load("sscp.n0"); _, n1 = load("sscp.n1")
ne_o, ref = load("sscp.out")       # ggml ne = [ch*freq, time]
F, T = int(ne_x[0]), int(ne_x[1])
print("in  ne", ne_x[:3], " w0 ne", ne_w0, " w1 ne", ne_w1, " out ne", ne_o[:2])

# memory order: ggml ne0 is fastest. x[t, f] = x_flat[t*F + f]
X = x.reshape(T, F)                                         # [time][mel]
def conv_layer(inp, w, ne_w, norm_w):
    """inp: [C][H=time][W=mel]  ->  [Cout][H'][W'] ; weight memory is [Cout][Cin][KH][KW] (ne = KW, KH, Cin, Cout)"""
    kw, kh, cin, cout = [int(v) for v in ne_w[:4]]
    W = w.reshape(cout, cin, kh, kw)
    if IM2COL_F16: W = W.astype(np.float16).astype(np.float32)
    C, H, Wd = inp.shape
    Ho, Wo = (H - 1) // 2 + 1, (Wd - 1) // 2 + 1
    pad = np.zeros((C, H + 2, Wd + 2), dtype=np.float32); pad[:, 1:H + 1, 1:Wd + 1] = inp
    out = np.zeros((cout, Ho, Wo), dtype=np.float32)
    for y in range(Ho):
        for xx in range(Wo):
            patch = pad[:, 2 * y:2 * y + 3, 2 * xx:2 * xx + 3]          # [C][3][3]
            if IM2COL_F16: patch = patch.astype(np.float16).astype(np.float32)   # F16_IM2COL
            out[:, y, xx] = np.tensordot(W, patch, axes=([1, 2, 3], [0, 1, 2]))
    # LayerNorm over channels at each (y, x), weight only, eps 1e-6
    mu = out.mean(0, keepdims=True); var = ((out - mu) ** 2).mean(0, keepdims=True)
    out = (out - mu) / np.sqrt(var + 1e-6) * norm_w[:, None, None]
    return np.maximum(out, 0)

h = conv_layer(X[None], w0, ne_w0, n0)
h = conv_layer(h, w1, ne_w1, n1)                            # [32][T'][F']
Tp, Fp = h.shape[1], h.shape[2]
# flatten: build() permutes (1,2,0,3) then reshape_2d(ne0*ne1, ne2): feature index = ch * ? + ? ; test both orders
cand = {
    "ch-major (ch, freq)": h.transpose(1, 0, 2).reshape(Tp, -1),
    "freq-major (freq, ch)": h.transpose(1, 2, 0).reshape(Tp, -1),
}
refm = ref.reshape(int(ne_o[1]), int(ne_o[0]))              # [time][ch*freq]
best = None
for name, c in cand.items():
    # gate on relative L2, not max-abs: with fp16-rounded operands a few elements flip across a rounding boundary
    # (0.2% of elements > 1e-4, median diff exactly 0) and dominate max-abs without indicating any layout error
    d = np.linalg.norm(c - refm) / np.linalg.norm(refm)
    print(f"  flatten order {name:24s} relL2 {d:.3e}   max|diff| {np.abs(c - refm).max():.2e}")
    if best is None or d < best[1]: best = (name, d)
ok = best[1] < 1e-4
print("BEST:", best[0], f"relL2 {best[1]:.2e}", "OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
