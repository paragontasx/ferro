#!/usr/bin/env python3
"""golden.py — numpy reference forward for gemma-embedding2 GGUF (mirrors llama.cpp graph)."""
import struct, sys, json, math
import numpy as np

def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n).decode('utf-8')

class GGUF:
    def __init__(self, path):
        self.f = open(path, 'rb')
        assert self.f.read(4) == b'GGUF'
        self.ver = struct.unpack('<I', self.f.read(4))[0]
        self.n_tensors = struct.unpack('<Q', self.f.read(8))[0]
        n_kv = struct.unpack('<Q', self.f.read(8))[0]
        self.kv = {}
        for _ in range(n_kv):
            k = rd_str(self.f); t = struct.unpack('<I', self.f.read(4))[0]
            self.kv[k] = self.rd_val(t)
        self.tensors = {}
        for _ in range(self.n_tensors):
            nm = rd_str(self.f); nd = struct.unpack('<I', self.f.read(4))[0]
            dims = [struct.unpack('<Q', self.f.read(8))[0] for _ in range(nd)]
            ty = struct.unpack('<I', self.f.read(4))[0]
            off = struct.unpack('<Q', self.f.read(8))[0]
            self.tensors[nm] = (dims, ty, off)
        al = self.kv.get('general.alignment', 32)
        self.data_start = (self.f.tell() + al - 1) // al * al
        self.mm = open(path, 'rb').read()

    def rd_val(self, t):
        if t == 0: return struct.unpack('<B', self.f.read(1))[0]
        if t == 1: return struct.unpack('<b', self.f.read(1))[0]
        if t == 2: return struct.unpack('<H', self.f.read(2))[0]
        if t == 3: return struct.unpack('<h', self.f.read(2))[0]
        if t == 4: return struct.unpack('<I', self.f.read(4))[0]
        if t == 5: return struct.unpack('<i', self.f.read(4))[0]
        if t == 6: return struct.unpack('<f', self.f.read(4))[0]
        if t == 7: return struct.unpack('<B', self.f.read(1))[0]
        if t == 8: return rd_str(self.f)
        if t == 9:
            et = struct.unpack('<I', self.f.read(4))[0]; n = struct.unpack('<Q', self.f.read(8))[0]
            return [self.rd_val(et) for _ in range(n)]
        if t == 10: return struct.unpack('<d', self.f.read(8))[0]
        if t == 11: return struct.unpack('<Q', self.f.read(8))[0]
        raise ValueError(t)

    def tensor(self, name):
        dims, ty, off = self.tensors[name]
        base = self.data_start + off
        n = 1
        for d in dims: n *= d
        if ty == 0:  # f32
            v = np.frombuffer(self.mm, dtype=np.float32, count=n, offset=base).copy()
        elif ty == 1:  # f16
            v = np.frombuffer(self.mm, dtype=np.float16, count=n, offset=base).astype(np.float32)
        elif ty == 30:  # bf16
            u = np.frombuffer(self.mm, dtype=np.uint16, count=n, offset=base).astype(np.uint32)
            v = u.view(np.float32) if False else (u << 16).view(np.float32)
        elif ty == 8:  # q8_0 — ggml layout: 2-byte fp16 scale FIRST, then 32 int8 quants
            assert dims[0] % 32 == 0
            nb = n // 32
            blk = np.frombuffer(self.mm, dtype=np.uint8, count=nb*34, offset=base).reshape(nb, 34)
            ds = blk[:, 0:2].copy().view(np.uint16).view(np.float16).astype(np.float32).reshape(nb, 1)
            qs = blk[:, 2:].astype(np.int8).astype(np.float32)
            v = (qs * ds).reshape(n)
        else:
            raise ValueError(f'unsupported type {ty} for {name}')
        # ggml ne = [ne0, ne1]; ne0 contiguous -> rows = ne1, cols = ne0
        if len(dims) == 2: return v.reshape(dims[1], dims[0])
        return v

def rms(x, w, eps):
    inv = 1.0 / np.sqrt((x*x).mean(-1, keepdims=True) + eps)
    return x * inv * w

def rope(x, pos, base, rot):
    # x: [T, nh, hd]; rotate first rot dims in rotate_half pairs
    T, nh, hd = x.shape
    out = x.copy()
    half = rot // 2
    idx = np.arange(half, dtype=np.float64)
    inv = 1.0 / (base ** (2*idx/rot))
    p = pos.astype(np.float64)[:, None] * inv[None, :]        # [T, half]
    c, s = np.cos(p)[:, None, :], np.sin(p)[:, None, :]       # [T,1,half]
    x1 = x[:, :, :half]; x2 = x[:, :, half:rot]
    out[:, :, :half]  = x1*c - x2*s
    out[:, :, half:rot] = x2*c + x1*s
    return out

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x); return e / e.sum(-1, keepdims=True)

def forward(g, seqs, dbg=None):
    import os
    P = g.kv['general.architecture'] + '.'
    I = lambda k, d=0: int(g.kv.get(P+k, d))
    F = lambda k, d=0.0: float(g.kv.get(P+k, d))
    n_layer, n_embd, n_ff = I('block_count'), I('embedding_length'), I('feed_forward_length')
    n_pl = I('embedding_length_per_layer_input'); n_out = I('embedding_length_out', n_embd)
    eps = F('attention.layer_norm_rms_epsilon', 1e-6)
    swa = I('attention.sliding_window', 0)
    rb, rbs = F('rope.freq_base', 1e6), F('rope.freq_base_swa', 1e6)
    hd_full = I('attention.key_length'); hd_swa = I('attention.key_length_swa', hd_full)
    rot_full = I('rope.dimension_count', hd_full); rot_swa = I('rope.dimension_count_swa', hd_swa)
    kvh = g.kv[P+'attention.head_count_kv']; pat = g.kv[P+'attention.sliding_window_pattern']
    n_head = I('attention.head_count')

    te = g.tensor('token_embd.weight')                      # [vocab, 512]
    plm = g.tensor('per_layer_model_proj.weight')            # [512*24, 512]
    plmn = g.tensor('per_layer_proj_norm.weight')            # [512]
    onorm = g.tensor('output_norm.weight')
    outw = g.tensor('output.weight')                         # [768, 512]

    ids = np.array([t for s in seqs for t in s], dtype=np.int64)
    seg = np.array([i for i, s in enumerate(seqs) for _ in s], dtype=np.int64)
    pos = np.array([i for s in seqs for i in range(len(s))], dtype=np.int64)
    T = len(ids)
    x = te[ids] * np.sqrt(n_embd)                            # [T, 512]
    if dbg: dbg('embd', x)
    # per-layer inputs
    y = x @ plm.T / np.sqrt(n_embd)                          # [T, 512*24]
    y = y.reshape(T, n_layer, n_pl).transpose(1, 0, 2)       # [L, T, n_pl]
    if dbg: dbg('plm_pre', y.transpose(1, 0, 2).reshape(T, n_layer * n_pl))
    if dbg: dbg('l0_h0_pre', y[0]); dbg('plm_norm', plmn)
    H = np.stack([rms(y[l], plmn, eps) for l in range(n_layer)])
    for il in range(n_layer):
        is_swa = bool(pat[il])
        nh, nkv = n_head, int(kvh[il])
        hd = hd_swa if is_swa else hd_full
        rot = rot_swa if is_swa else rot_full
        base = rbs if is_swa else rb
        L = f'blk.{il}.'
        an = g.tensor(L+'attn_norm.weight')
        wq = g.tensor(L+'attn_q.weight'); wk = g.tensor(L+'attn_k.weight'); wv = g.tensor(L+'attn_v.weight')
        wo = g.tensor(L+'attn_output.weight')
        qn = g.tensor(L+'attn_q_norm.weight'); kn = g.tensor(L+'attn_k_norm.weight')
        apn = g.tensor(L+'post_attention_norm.weight')
        fn = g.tensor(L+'ffn_norm.weight')
        fg = g.tensor(L+'ffn_gate.weight'); fu = g.tensor(L+'ffn_up.weight'); fd = g.tensor(L+'ffn_down.weight')
        fpn = g.tensor(L+'post_ffw_norm.weight')
        ig = g.tensor(L+'inp_gate.weight'); pj = g.tensor(L+'proj.weight'); pn = g.tensor(L+'post_norm.weight')
        os_ = g.tensor(L+'layer_output_scale.weight').reshape(-1)[0]

        b = rms(x, an, eps)
        if dbg and il == 0: dbg('l0_normed', b)
        if dbg and il == 0 and os.environ.get('GOLDEN_DBGW'): dbg('l0_wqkv', np.concatenate([wq, wk, wv], axis=0))
        q = (b @ wq.T).reshape(T, nh, hd)
        k = (b @ wk.T).reshape(T, nkv, hd)
        v = (b @ wv.T).reshape(T, nkv, hd)
        if dbg and il == 0: dbg('l0_qkv_pre', np.concatenate([q, k, v], axis=1))
        q = rms(q, qn, eps); k = rms(k, kn, eps)
        v = v / np.sqrt((v*v).mean(-1, keepdims=True) + eps)
        q = rope(q, pos, base, rot); k = rope(k, pos, base, rot)
        if dbg and il == 0: dbg('l0_qkv', np.concatenate([q, k, v], axis=1))
        # non-causal attention, per-sequence, symmetric swa
        o = np.zeros((T, nh, hd), dtype=np.float32)
        for s in range(len(seqs)):
            idx = np.where(seg == s)[0]
            q[idx] = q[idx]
            qh = q[idx].transpose(1, 0, 2)                   # [nh, n, hd]
            kvmap = [h // (nh // nkv) for h in range(nh)]
            kh = k[idx][:, kvmap, :].transpose(1, 0, 2)          # [nh, n, hd]
            vh = v[idx][:, kvmap, :].transpose(1, 0, 2)
            sc = qh @ kh.transpose(0, 2, 1)                  # [nh, n, n]
            if is_swa and swa > 0:
                d = np.abs(idx[:, None] - idx[None, :])
                sc = np.where(d < swa, sc, -1e30)
            a = softmax(sc) @ vh                             # [nh, n, hd]
            o[idx] = a.transpose(1, 0, 2)
        attn = o.reshape(T, nh*hd) @ wo.T
        if dbg and il == 0: dbg('l0_attn', o.reshape(T, nh*hd))
        attn = rms(attn, apn, eps)
        if dbg and il == 0: dbg('l0_attnpost', attn)
        ao = attn + x
        b = rms(ao, fn, eps)
        up = b @ fu.T; gate = b @ fg.T
        if dbg: dbg(f'L{il}_gateup', np.concatenate([gate, up], axis=1))
        gate = gate * 0.5 * (1 + np.vectorize(math.erf)(gate/np.sqrt(2)))
        if dbg: dbg(f'L{il}_gelumul', up*gate)
        ff = (up*gate) @ fd.T
        ff = rms(ff, fpn, eps)
        if dbg: dbg(f'L{il}_ffnpost', ff)
        pe = ff + ao
        if dbg: dbg(f'L{il}_pein', pe)
        b = rms(pe, np.ones(n_embd, dtype=np.float32), eps) if False else pe
        g2 = pe @ ig.T                                        # [T, n_pl]
        if dbg: dbg(f'L{il}_gate_pre', g2); dbg(f'L{il}_gelu_post', 0.5*g2*(1+np.vectorize(math.erf)(g2/np.sqrt(2))))
        g2 = 0.5*g2*(1+np.vectorize(math.erf)(g2/np.sqrt(2)))
        if dbg: dbg(f'L{il}_h0', H[il])
        g2 = g2 * H[il]
        ple = g2 @ pj.T
        ple = rms(ple, pn, eps)
        if dbg: dbg(f'L{il}_pleout', ple)
        x = (pe + ple) * os_
        if dbg: dbg(f'layer{il}_out', x)
    z = rms(x, onorm, eps)
    e = z @ outw.T                                            # [T, 768]
    embs = []
    for s in range(len(seqs)):
        m = e[seg == s].mean(0)
        embs.append(m / (np.linalg.norm(m) + 1e-12))
    return np.stack(embs)

if __name__ == '__main__':
    model, tokfile, outjson = sys.argv[1], sys.argv[2], sys.argv[3]
    g = GGUF(model)
    seqs = [[int(t) for t in l.split()] for l in open(tokfile) if l.strip()]
    import os
    dbgdir = os.environ.get('GOLDEN_DBGDIR')
    def dbg(tag, arr):
        if not dbgdir: return
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        with open(os.path.join(dbgdir, tag + '.bin'), 'wb') as f:
            f.write(struct.pack('<Q', a.size)); f.write(a.tobytes())
    e = forward(g, seqs, dbg=dbg)
    json.dump({'dim': e.shape[1], 'embd': e.tolist()}, open(outjson, 'w'))
    print('golden wrote', outjson, e.shape)
