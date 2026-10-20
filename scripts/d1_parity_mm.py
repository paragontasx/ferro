#!/usr/bin/env python3
"""d1-omni multimodal parity: ferro-embed `decide` vs llama.cpp `llama-server --mmproj` (/v1/systemone).
Usage: python3 scripts/d1_parity_mm.py [models/d1-omni-600M-Q8_0.gguf] [server_url]
Server: llama-server -m d1.gguf --mmproj mmproj.gguf -b 4096 -ub 4096 --port 8089
Reference answers are cached (D1_MM_REF_CACHE) by request hash so reruns do not need the server."""
import base64, hashlib, json, os, subprocess, sys, urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/d1-omni-600M-Q8_0.gguf"
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8089"
MEDIA = "bench/d1/media"
CACHE = os.environ.get("D1_MM_REF_CACHE", "/opt/data/cache/scratch/d1_mm_ref_cache.json")
EXTRA = os.environ.get("FERRO_ARGS", "").split()

def uri(name):
    mime = {"jpg": "image/jpeg", "png": "image/png", "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}[name.rsplit(".", 1)[1]]
    return f"data:{mime};base64," + base64.b64encode(open(f"{MEDIA}/{name}", "rb").read()).decode()

TEAM = {"type": "choice", "instructions": "What is shown?", "criteria": {"cats": "One or more cats", "dogs": "One or more dogs",
        "people": "People", "food": "Food or drinks", "vehicle": "A car, bus or train"}}
REQS = [
    # 0 single image, choice + noul + score
    {"state": "Photo uploaded to the family album.", "files": [uri("cats.jpg")], "questions": {
        "subject": TEAM,
        "indoor": {"type": "noul", "instructions": "Was the photo taken indoors?"},
        "count": {"type": "score", "instructions": "How many animals are visible?", "criteria": ["none", "one", "two", "three or more"]}}},
    # 1 null state, image only, noul without criteria -> short yes/no form
    {"state": None, "images": [uri("cats.jpg")], "questions": {
        "sleep": {"type": "noul", "instructions": "Are the animals sleeping or lying down?"},
        "remote": {"type": "noul", "instructions": "Is there a remote control in the picture?"}}},
    # 2 tiny image (upscaled to the minimum size)
    {"state": "Thumbnail from a camera roll.", "files": [uri("tiny.png")], "questions": {"subject": TEAM}},
    # 3 large image (tiled: 4x2 tiles + thumbnail)
    {"state": "Scan from the NAS photo library.", "files": [uri("big.png")], "questions": {
        "subject": TEAM, "pink": {"type": "noul", "instructions": "Is there a pink blanket?", "criteria": {"true": "a pink blanket or sofa is visible", "false": "no pink"}}}},
    # 4 two images in one request
    {"state": "Two pictures from the same folder.", "files": [uri("cats.jpg"), uri("tiny.png")], "questions": {
        "same": {"type": "noul", "instructions": "Do both pictures show the same scene?"}}},
    # 5 chat-message state with an image_url part
    {"state": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Is this safe to post publicly?"},
                                                          {"type": "image_url", "image_url": {"url": uri("cats.jpg")}}]}]},
     "questions": {"safe": {"type": "noul", "instructions": "Is the image safe for a public audience?"}}},
    # 6 audio, text state, choice (audio: option_NNN labels) + noul (audio: yes/no) + score
    {"state": "Recording from a public event.", "files": [uri("jfk.wav")], "questions": {
        "speaker": {"type": "choice", "instructions": "Who is speaking?", "criteria": {"politician": "A politician giving a speech",
                    "singer": "A singer performing", "child": "A child talking", "announcer": "A sports announcer"}},
        "speech": {"type": "noul", "instructions": "Is someone giving a speech?", "criteria": {"true": "ignored for audio", "false": "also ignored"}},
        "energy": {"type": "score", "instructions": "How emotional is the delivery?", "criteria": ["flat", "calm", "passionate"]}}},
    # 7 audio, null state (template renders {})
    {"state": None, "files": [uri("jfk.wav")], "questions": {
        "country": {"type": "noul", "instructions": "Does the speaker mention their country?"}}},
    # 8 mp3 + flac decoding/resampling paths
    {"state": "Voice memo.", "files": [uri("jfk.mp3")], "questions": {"speech": {"type": "noul", "instructions": "Is this a speech?"}}},
    {"state": "Voice memo.", "files": [uri("jfk_44k.flac")], "questions": {"speech": {"type": "noul", "instructions": "Is this a speech?"}}},
    # 10 audio longer than 30 s (cut to 30 s)
    {"state": "Long recording from the NAS.", "files": [uri("long.wav")], "questions": {
        "repeat": {"type": "noul", "instructions": "Does the recording repeat the same sentence several times?"}}},
    # 11 chat-message state with input_audio
    {"state": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(open(f"{MEDIA}/jfk.wav", "rb").read()).decode(), "format": "wav"}},
                                             {"type": "text", "text": "Classify this clip."}]}],
     "questions": {"music": {"type": "noul", "instructions": "Is this music?"}}},
]
ERR_REQS = [
    {"state": "x", "files": [uri("cats.jpg"), uri("jfk.wav")], "questions": {"q": {"type": "noul", "instructions": "?"}}},   # image + audio
    {"state": "x", "files": ["https://example.com/a.jpg"], "questions": {"q": {"type": "noul", "instructions": "?"}}},       # remote URL
]

def key(r): return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()

def ref(r, cache):
    k = key(r)
    if k not in cache:
        q = urllib.request.Request(URL + "/v1/systemone", data=json.dumps(r).encode(), headers={"Content-Type": "application/json"})
        try:
            cache[k] = json.load(urllib.request.urlopen(q, timeout=3600))
        except urllib.error.HTTPError as e:
            cache[k] = {"error": json.loads(e.read().decode() or "{}")}
        json.dump(cache, open(CACHE, "w"))
    return cache[k]

def flat(a):
    out = {}
    for qid, x in a.get("answers", {}).items():
        if x["type"] == "noul": out[qid] = x["noul"]
        else:
            for k, v in x["probabilities"].items(): out[f"{qid}.{k}"] = v
    return out

def top(a):
    out = {}
    for qid, x in a.get("answers", {}).items():
        out[qid] = (x["noul"] > 0.5) if x["type"] == "noul" else x.get("choice", max(x["probabilities"], key=x["probabilities"].get))
    return out

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
refs = [ref(r, cache) for r in REQS]
json.dump(REQS, open("/opt/data/cache/scratch/d1_mm_req.json", "w"))
p = subprocess.run(["./ferro-embed", "decide", "--model", MODEL, "--request", "/opt/data/cache/scratch/d1_mm_req.json"] + EXTRA,
                   capture_output=True, text=True)
if p.returncode != 0:
    print(p.stderr[-3000:]); sys.exit(1)
got = json.loads(p.stdout)
agree = total = 0; worst = 0.0; tok_eq = 0
for i, (a, b) in enumerate(zip(refs, got)):
    if "error" in a: print(f"req {i:2d}: reference error {a['error']}"); continue
    ta, tb = a["usage"]["input_tokens"], b["usage"]["input_tokens"]
    tok_eq += ta == tb
    fa, fb = flat(a), flat(b)
    d = max(abs(fa[k] - fb[k]) for k in fa)
    worst = max(worst, d)
    A, B = top(a), top(b)
    ag = sum(A[q] == B[q] for q in A); agree += ag; total += len(A)
    print(f"req {i:2d}: tokens ferro={tb:5d} llama={ta:5d}  max|dp|={d:.4f}  top {ag}/{len(A)}" + ("" if ag == len(A) and ta == tb else "  <-- check"))
    if os.environ.get("VERBOSE"): print("    ref  ", {k: round(v, 4) for k, v in fa.items()}, "\n    ferro", {k: round(v, 4) for k, v in fb.items()})
print(f"SUMMARY: {len(REQS)} media requests, input_tokens equal {tok_eq}/{len(REQS)}, top answers agree {agree}/{total}, worst |dprob| {worst:.4f}")
# error handling: both must reject
for r in ERR_REQS:
    a = ref(r, cache)
    json.dump(r, open("/opt/data/cache/scratch/d1_mm_err.json", "w"))
    p = subprocess.run(["./ferro-embed", "decide", "--model", MODEL, "--request", "/opt/data/cache/scratch/d1_mm_err.json"] + EXTRA, capture_output=True, text=True)
    print(f"reject: llama={'error' in a} ferro={p.returncode != 0}  ({p.stdout.strip()[:110]})")
