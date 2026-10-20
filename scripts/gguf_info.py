#!/usr/bin/env python3
"""Minimal GGUF inspector: dumps KV metadata and tensor layout."""
import struct, sys, json

TYPE_FMT = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'bool',8:'Q',9:'q',10:'d'}
ARR, STR, TENSOR = 4, 8, 9  # gguf type ids: 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,8 string,9 arr

def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

def rd_val(f, t):
    if t == 8: return rd_str(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if et == 8:
            # special-case huge string arrays: summarize
            if n > 32:
                head = [rd_str(f) for _ in range(4)]
                for _ in range(n - 8): rd_str(f)
                tail = [rd_str(f) for _ in range(4)]
                return f"<str[{n}] head={head} tail={tail}>"
            return [rd_str(f) for _ in range(n)]
        fmt = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',8:'Q',9:'q',10:'d'}[et]
        cnt = struct.unpack(f'<{n}{fmt}', f.read(n*struct.calcsize(fmt)))
        return list(cnt[:16]) + ([f'...({n} total)'] if n > 16 else [])
    fmt = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',8:'Q',9:'q',10:'d'}[t]
    return struct.unpack(f'<{fmt}', f.read(struct.calcsize(fmt)))[0]

def main(path, show_tensors=40):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', 'not gguf'
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    print(f'# {path}\nversion={ver} n_tensors={n_tensors} n_kv={n_kv}')
    meta = {}
    for _ in range(n_kv):
        k = rd_str(f); t = struct.unpack('<I', f.read(4))[0]
        v = rd_val(f, t)
        meta[k] = v
        s = str(v)
        print(f'  {k} = {s[:110]}')
    align = meta.get('general.alignment', 32)
    # tensor infos
    from collections import Counter
    types = Counter(); names = []
    for i in range(n_tensors):
        name = rd_str(f)
        nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        dt = struct.unpack('<I', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        types[dt] += 1
        if i < show_tensors:
            names.append(f'  [{i:3d}] {name:55s} dims={dims} dtype={dt} off={off}')
    print('tensor dtype histogram:', dict(types))
    for l in names: print(l)
    # compute data start
    # (approximate: alignment after tensor infos — recompute by seeking)
    f.close()

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
