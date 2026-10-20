#!/usr/bin/env python3
"""EmbeddingGemma-2 multimodal parity: ferro-embed `embed-media` vs llama.cpp `llama-server --mmproj --embeddings`.
Usage: python3 scripts/eg2_parity_mm.py [server_url] [--only NAME[,NAME]] [--gate 0.999]
Server: llama-server -m embeddinggemma-2-Q8_0.gguf --mmproj mmproj-embeddinggemma-2-Q8_0.gguf --embeddings --pooling mean \
        -t 2 -b 8192 -ub 8192 -c 8192 --port 8091
Reference vectors are cached (EG2_MM_REF_CACHE) by request hash so reruns do not need the server.
The final metric is the cosine of the L2-normalised 768-d embedding (what a retrieval index stores)."""
import base64, hashlib, json, os, subprocess, sys, time, struct, urllib.request, math

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
URL = args[0] if args else "http://127.0.0.1:8091"
GATE = float(opts.get("--gate", "0.999"))
ONLY = set(opts["--only"].split(",")) if "--only" in opts else None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(ROOT, "bench/d1/media")
MODEL = os.path.join(ROOT, "models/embeddinggemma-2-Q8_0.gguf")
MMPROJ = os.path.join(ROOT, "models/mmproj-embeddinggemma-2-Q8_0.gguf")
CACHE = os.environ.get("EG2_MM_REF_CACHE", "/opt/data/cache/scratch/eg2_mm_ref_cache.json")
FERRO_EXTRA = os.environ.get("FERRO_ARGS", "").split()
MIME = {"jpg": "image/jpeg", "png": "image/png", "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}

def b64(name):
    return base64.b64encode(open(os.path.join(MEDIA, name), "rb").read()).decode()

# (name, [(kind, value)...]) kind in text|image|audio|video ; value = text or a file in bench/d1/media
CASES = [
    ("text",          [("text", "a photo of two cats lying on a pink couch")]),
    ("img_cats",      [("image", "cats.jpg")]),
    ("img_tiny",      [("image", "tiny.png")]),
    ("img_big",       [("image", "big.png")]),
    ("img_text",      [("text", "two cats on a sofa"), ("image", "cats.jpg")]),
    ("audio_jfk",     [("audio", "jfk.wav")]),
    ("audio_jfk_mp3", [("audio", "jfk.mp3")]),
    ("audio_jfk_flac", [("audio", "jfk_44k.flac")]),
    ("audio_long",    [("audio", "long.wav")]),
    ("audio_text",    [("text", "a speech about the nation"), ("audio", "jfk.wav")]),
    ("video_cats",    [("video", "clip_cats.mp4")]),
    ("mixed",         [("text", "cats"), ("image", "cats.jpg"), ("audio", "jfk.wav")]),
]

def load_cache():
    try: return json.load(open(CACHE))
    except Exception: return {}

def data_uri(name):
    ext = name.rsplit(".", 1)[1]
    mime = {"jpg": "image/jpeg", "png": "image/png", "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "mp4": "video/mp4"}[ext]
    return f"data:{mime};base64," + b64(name)

def ref_embed(server_marker, items, cache):
    """Reference vector from llama-server POST /v1/embeddings with interleaved content parts (text/image/audio/video).
    NOTE: the legacy /embedding + image_data shape silently returns a TEXT-ONLY vector - never use it."""
    key = hashlib.sha256(json.dumps(items).encode() + open(MMPROJ, "rb").read(4096)).hexdigest()[:24]
    if key in cache: return cache[key]
    parts = []
    for kind, v in items:
        if kind == "text": parts.append({"type": "text", "text": v})
        elif kind == "image": parts.append({"type": "image_url", "image_url": {"url": data_uri(v)}})
        elif kind == "audio":
            ext = v.rsplit(".", 1)[1]
            parts.append({"type": "input_audio", "input_audio": {"data": b64(v), "format": ext}})
        elif kind == "video": parts.append({"type": "input_video", "input_video": {"url": data_uri(v)}})
    body = json.dumps({"input": [{"content": parts}]}).encode()
    req = urllib.request.Request(URL + "/v1/embeddings", body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    vec = r["data"][0]["embedding"]
    cache[key] = vec
    os.makedirs(os.path.dirname(CACHE), exist_ok=True); json.dump(cache, open(CACHE, "w"))
    return vec

def oracle_sane():
    """A text-only fallback vector would make every media case 'pass' trivially: require image(cats) closer to a matching
    caption than to an unrelated one, and two different images to differ."""
    def emb(parts):
        body = json.dumps({"input": [{"content": parts}]} if isinstance(parts, list) else {"input": parts}).encode()
        req = urllib.request.Request(URL + "/v1/embeddings", body, {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["data"][0]["embedding"]
    cats = emb([{"type": "image_url", "image_url": {"url": data_uri("cats.jpg")}}])
    tiny = emb([{"type": "image_url", "image_url": {"url": data_uri("tiny.png")}}])
    match, other = emb("two cats lying on a couch"), emb("a diesel truck on a highway")
    a, b, c = cos(cats, match), cos(cats, other), cos(cats, tiny)
    ok = a > b + 0.1 and c < 0.999
    print(f"oracle sanity: cats~match {a:.3f}  cats~unrelated {b:.3f}  cats~tiny {c:.3f}  -> {'OK' if ok else 'ORACLE IS NOT SEEING THE MEDIA'}")
    return ok

def ferro_embed(items):
    cmd = [os.path.join(ROOT, "ferro-embed"), "embed-media", "--model", MODEL, "--mmproj", MMPROJ] + FERRO_EXTRA
    for kind, v in items:
        cmd += ["--item", f"{kind}:{v}" if kind == "text" else f"{kind}:{os.path.join(MEDIA, v)}"]
    out = "/opt/data/cache/scratch/_eg2_ferro.bin"
    cmd += ["--out", out]
    t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200); dt = time.time() - t
    if p.returncode: return None, dt, p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "failed"
    raw = open(out, "rb").read(); vec = list(struct.unpack(f"<{len(raw)//4}f", raw))
    tok = [l for l in p.stderr.splitlines() if "[embed-media]" in l]
    return vec, dt, tok[-1] if tok else ""

def cos(a, b):
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb + 1e-30)

def main():
    cache = load_cache()
    marker = "<__media__>"
    try:
        props = json.loads(urllib.request.urlopen(URL + "/props", timeout=5).read()); marker = props.get("media_marker", marker)
    except Exception: pass
    os.makedirs("/opt/data/cache/scratch", exist_ok=True)
    try:
        if not oracle_sane(): sys.exit(3)
    except Exception as e:
        print(f"oracle sanity skipped (server unreachable / cached refs only): {type(e).__name__}")
    worst, rows, fails = 1.0, [], 0
    for name, items in CASES:
        if ONLY and name not in ONLY: continue
        if any(k == "video" for k, _ in items):
            rows.append((name, "-", "-", "SKIP (video reference handled by --video-ref)")); continue
        try: ref = ref_embed(marker, items, cache)
        except Exception as e:
            rows.append((name, "-", "-", f"REF FAIL {e}")); fails += 1; continue
        vec, dt, info = ferro_embed(items)
        if vec is None: rows.append((name, "-", f"{dt:.1f}s", f"FERRO FAIL {info}")); fails += 1; continue
        c = cos(vec, ref); worst = min(worst, c)
        ok = c >= GATE; fails += (not ok)
        rows.append((name, f"{c:.6f}", f"{dt:.1f}s", ("PASS " if ok else "FAIL ") + info))
    print(f"{'case':16} {'cos':>9} {'ferro':>7}  note")
    for r in rows: print(f"{r[0]:16} {r[1]:>9} {r[2]:>7}  {r[3]}")
    print(f"WORST cos {worst:.6f}  gate {GATE}  failures {fails}")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
