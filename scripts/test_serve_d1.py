#!/usr/bin/env python3
"""Tests for `ferro-embed serve --d1 d1-omni.gguf` (POST /v1/systemone, llama.cpp's decision endpoint). stdlib + optional Pillow.

  ferro-embed serve --d1 models/d1-omni-600M-Q8_0.gguf --port 8090
  python3 scripts/test_serve_d1.py --url http://127.0.0.1:8090 [--cli ./ferro-embed --d1 models/d1-omni-600M-Q8_0.gguf]

--cli: also check that the server answers are identical to the one-shot `ferro-embed decide` CLI on the same requests.
Crop tests need Pillow (a server-side crop must give the same answer as sending a pre-cropped PNG).
"""
import argparse, base64, concurrent.futures as cf, io, json, math, os, subprocess, sys, tempfile, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8090"); ap.add_argument("--cli"); ap.add_argument("--d1")
ap.add_argument("--media", default=os.path.join(os.path.dirname(__file__), "..", "bench", "d1", "media"))
A = ap.parse_args()
fails = []

def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (("  " + str(extra)) if extra != "" else ""), flush=True)
    if not ok: fails.append(name)

def post(body, path="/v1/systemone", timeout=600):
    data = body if isinstance(body, (bytes, str)) else json.dumps(body)
    req = urllib.request.Request(A.url + path, data.encode() if isinstance(data, str) else data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        b = e.read()
        try: return e.code, json.loads(b)
        except Exception: return e.code, b.decode(errors="replace")

def get(path):
    with urllib.request.urlopen(A.url + path, timeout=30) as r: return json.loads(r.read())

def durl(path, mime): return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
M = A.media
PNG = os.path.join(M, "cats624.png")

def same(a, b, tol=1e-6):          # JSON structural equality, floats within tol
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list): return isinstance(b, list) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float): return isinstance(b, (int, float)) and abs(a - b) <= tol * max(1.0, abs(a))
    return a == b

Q = {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
     "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults"}},
     "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "Today", "Blocking the customer now"]}}
IQ = {"cat": {"type": "noul", "instructions": "Is there a cat in the image?"}, "dog": {"type": "noul", "instructions": "Is there a dog in the image?"},
      "scene": {"type": "choice", "instructions": "What does the image show?", "criteria": {"animals": "Animals", "food": "Food", "vehicles": "Vehicles"}}}
TEXT = {"state": "I was charged twice this month, please refund one of them.", "questions": Q}
BATCH = [TEXT, {"state": {"subject": "App crashes on login", "body": "Since the update the app closes right after I log in."}, "questions": Q}]
IMG = {"state": "Look at the picture.", "files": [durl(PNG, "image/png")], "questions": IQ}
AUD = {"state": "Listen to the clip.", "files": [durl(os.path.join(M, "jfk.wav"), "audio/wav")],
       "questions": {"speech": {"type": "noul", "instructions": "Is someone speaking?"},
                     "topic": {"type": "choice", "instructions": "What is the topic?", "criteria": {"politics": "Politics, government, country", "music": "Music", "sport": "Sports"}}}}

# ---- discovery
models = get("/v1/models")["data"]
d1m = [m for m in models if "decision" in m.get("capabilities", [])]
check("/v1/models lists the decision model", len(d1m) == 1, d1m[0]["id"] if d1m else models)
has_img = "image" in d1m[0]["modalities"] if d1m else False; has_aud = "audio" in d1m[0]["modalities"] if d1m else False
h = get("/health"); check("/health has a d1 section", "d1" in h, h.get("d1"))

# ---- text
st, r = post(TEXT)
check("text request -> 200 with typed answers", st == 200 and set(r["answers"]) == set(Q) and r["answers"]["refund"]["type"] == "noul" and r["answers"]["team"]["choice"] == "billing", r if st != 200 else r["answers"]["team"]["choice"])
check("noul probability is high for the refund question", st == 200 and r["answers"]["refund"]["noul"] > 0.9, r["answers"]["refund"] if st == 200 else "")
check("usage reports input tokens, 0 output", st == 200 and r["usage"]["input_tokens"] > 0 and r["usage"]["output_tokens"] == 0, r.get("usage"))
st, rb = post(BATCH)
check("array request -> array of answers", st == 200 and isinstance(rb, list) and len(rb) == 2 and same(rb[0], r), st)
st, r2 = post(TEXT); check("deterministic (same request twice)", same(r, r2, 0))

# ---- CLI parity
def cli(req):
    if not (A.cli and A.d1): return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(req, f); p = f.name
    try:
        out = subprocess.run([A.cli, "decide", "--model", A.d1, "--request", p], capture_output=True, text=True).stdout
        return json.loads(out)
    finally: os.unlink(p)
if A.cli and A.d1:
    for nm, rq in [("text", TEXT), ("batch", BATCH), ("image", IMG), ("audio", AUD)]:
        want = cli(rq); st, got = post(rq)
        check(f"server == one-shot CLI ({nm})", st == 200 and same(want, got, 1e-6))
else:
    print("SKIP server == CLI (pass --cli and --d1 to enable)")

# ---- media
if has_img:
    st, ri = post(IMG)
    check("image request: cat detected, no dog", st == 200 and ri["answers"]["cat"]["noul"] > 0.9 and ri["answers"]["dog"]["noul"] < 0.1, ri["answers"] if st == 200 else ri)
    check("image tokens are counted in usage", st == 200 and ri["usage"]["input_tokens"] > 100, ri.get("usage"))
    st, rj = post({**IMG, "files": [durl(os.path.join(M, "cats.jpg"), "image/jpeg")]})
    check("JPEG works too", st == 200 and rj["answers"]["cat"]["noul"] > 0.9)
    st, rc = post({"questions": IQ, "state": {"messages": [{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": durl(PNG, "image/png")}}]}]}})
    check("image_url part inside a chat-message state works", st == 200 and rc["answers"]["cat"]["noul"] > 0.9, rc if st != 200 else "")
else: print("SKIP image tests (server has no vision projector)")
if has_aud:
    st, ra = post(AUD)
    check("audio request: speech detected, politics", st == 200 and ra["answers"]["speech"]["noul"] > 0.5 and ra["answers"]["topic"]["choice"] == "politics", ra["answers"] if st == 200 else ra)
else: print("SKIP audio tests")

# ---- crops (server-side region of a full frame == the same region sent as its own image)
try:
    from PIL import Image
except Exception:
    Image = None
if has_img and Image:
    im = Image.open(PNG).convert("RGB"); W, H = im.size
    def png_of(box):
        b = io.BytesIO(); im.crop(box).save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    def ask(files, extra=None):
        return post({"state": "Look at the picture.", "files": files, "questions": IQ, **(extra or {})})
    box = (W // 8, H // 8, W // 8 + W // 2, H // 8 + H // 2)                         # x0,y0,x1,y1 px
    st, want = ask([png_of(box)])
    c_xywh = {"x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]}
    st1, g1 = ask([{"url": durl(PNG, "image/png"), "crop": c_xywh}])
    check("crop {x,y,w,h} == pre-cropped PNG", st == 200 and st1 == 200 and same(want, g1, 1e-6), g1 if st1 != 200 else "")
    st2, g2 = ask([{"url": durl(PNG, "image/png"), "crop": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}}])
    check("crop {x1,y1,x2,y2} == {x,y,w,h}", st2 == 200 and same(g1, g2, 0), g2 if st2 != 200 else "")
    st3, g3 = ask([{"url": durl(PNG, "image/png"), "crop": {"x": box[0] / W, "y": box[1] / H, "w": (box[2] - box[0]) / W, "h": (box[3] - box[1]) / H, "unit": "norm"}}])
    check("normalised crop == pixel crop", st3 == 200 and same(g1, g3, 1e-6), g3 if st3 != 200 else "")
    st4, g4 = ask([{"url": durl(PNG, "image/png"), "crop": {"x": W // 4, "y": H // 4, "w": W // 4, "h": H // 4, "pad": 0.5}}])
    st4w, w4 = ask([png_of((W // 4 - W // 8, H // 4 - H // 8, W // 4 + W // 4 + W // 8, H // 4 + H // 4 + H // 8))])
    check("pad grows the box on every side (== manually padded crop)", st4 == 200 and st4w == 200 and same(w4, g4, 1e-6))
    st5, g5 = ask([{"url": durl(PNG, "image/png"), "crop": {"x": -50, "y": -50, "w": 150, "h": 150}}])
    st5w, w5 = ask([png_of((0, 0, 100, 100))])
    check("a box that sticks out of the frame is clipped", st5 == 200 and st5w == 200 and same(w5, g5, 1e-6))
    # several detector boxes on one frame -> one image per box, one request
    b2 = (W // 2, H // 2, W, H)
    st6, g6 = ask([{"url": durl(PNG, "image/png"), "crops": [c_xywh, {"x": b2[0], "y": b2[1], "w": b2[2] - b2[0], "h": b2[3] - b2[1]}]}])
    st6w, w6 = ask([png_of(box), png_of(b2)])
    check("crops [a, b] == the two crops sent as two images", st6 == 200 and st6w == 200 and same(w6, g6, 1e-6) and g6["usage"]["input_tokens"] > g1["usage"]["input_tokens"], g6 if st6 != 200 else "")
    st7, g7 = ask([{"url": durl(PNG, "image/png"), "crop": c_xywh, "include_full": True}])
    st7w, w7 = ask([durl(PNG, "image/png"), png_of(box)])
    check("include_full adds the whole frame first", st7 == 200 and st7w == 200 and same(w7, g7, 1e-6), g7 if st7 != 200 else "")
    # image_url parts accept the same fields (reference = the same chat-message state with the region sent as its own image)
    def msg_state(url_obj): return {"messages": [{"role": "user", "content": [{"type": "text", "text": "Look."}, {"type": "image_url", "image_url": url_obj}]}]}
    st8, g8 = post({"state": msg_state({"url": durl(PNG, "image/png"), "crop": c_xywh}), "questions": IQ})
    st8w, w8 = post({"state": msg_state({"url": png_of(box)}), "questions": IQ})
    check("image_url part with a crop == the same message with a pre-cropped image", st8 == 200 and st8w == 200 and same(w8, g8, 1e-6), g8 if st8 != 200 else "")
    # errors
    for nm, spec, frag in [("box outside the image", {"x": W + 10, "y": 0, "w": 50, "h": 50}, "outside"), ("zero width", {"x": 0, "y": 0, "w": 0, "h": 5}, "positive"),
                           ("bad unit", {"x": 0, "y": 0, "w": 5, "h": 5, "unit": "cm"}, "unit"), ("missing field", {"x": 0, "y": 0, "w": 5}, "required"),
                           ("negative pad", {"x": 0, "y": 0, "w": 5, "h": 5, "pad": -1}, "pad"), ("not an object", [1, 2, 3, 4], "object")]:
        s_, e_ = ask([{"url": durl(PNG, "image/png"), "crop": spec}])
        check(f"crop error: {nm} -> 400", s_ == 400 and frag in json.dumps(e_), (s_, str(e_)[:120]))
    s_, e_ = ask([{"url": durl(PNG, "image/png"), "crops": [c_xywh] * 9}]); check("more than 8 crops -> 400", s_ == 400 and "maximum" in json.dumps(e_), (s_, str(e_)[:100]))
    s_, e_ = ask([{"url": durl(PNG, "image/png"), "crops": []}]); check("empty crops array -> 400", s_ == 400)
elif has_img: print("SKIP crop tests (pip install pillow)")
if has_aud:
    s_, e_ = post({**AUD, "files": [{"url": AUD["files"][0], "crop": {"x": 0, "y": 0, "w": 5, "h": 5}}]}); check("crop on audio -> 400", s_ == 400 and "image" in json.dumps(e_), (s_, str(e_)[:100]))

# ---- request errors
for nm, body, frag in [("invalid JSON", "{nope", ""), ("no state", {"questions": Q}, "state"), ("no questions", {"state": "x"}, "questions"),
                       ("empty array", [], "empty"), ("videos", {**TEXT, "videos": ["data:video/mp4;base64,AAAA"]}, "videos"),
                       ("remote image URL", {**TEXT, "files": ["http://example.com/a.png"]}, "data URL"), ("bad question type", {"state": "x", "questions": {"q": {"type": "essay", "instructions": "?"}}}, "type")]:
    s_, e_ = post(body); check(f"error: {nm} -> 400", s_ == 400 and frag in json.dumps(e_), (s_, str(e_)[:100]))
try:
    urllib.request.urlopen(A.url + "/v1/systemone", timeout=10); check("GET -> 405", False)
except urllib.error.HTTPError as e: check("GET /v1/systemone -> 405", e.code == 405, e.code)
s_, _ = post(TEXT); check("server healthy after the error cases", s_ == 200)

# ---- concurrency: answers do not depend on who they are batched with; queued requests share passes
reqs = [TEXT, BATCH[1], TEXT, BATCH[1]] * 4 + ([IMG] * 2 if has_img else [])
before = get("/health")["d1"]
with cf.ThreadPoolExecutor(len(reqs)) as ex: res = list(ex.map(post, reqs))
ok = all(st == 200 for st, _ in res)
alone = {id(TEXT): post(TEXT)[1], id(BATCH[1]): post(BATCH[1])[1]}
same_all = ok and all(same(alone[id(rq)], rs, 2e-3) for rq, (st, rs) in zip(reqs, res) if id(rq) in alone)
check(f"{len(reqs)} concurrent requests all answered", ok)
check("concurrent answers == answers when alone (batching does not change them; tol 2e-3)", same_all)
after = get("/health")["d1"]
check("queued requests shared forward passes", after["passes"] - before["passes"] <= after["requests"] - before["requests"], f"{after['requests'] - before['requests']} requests in {after['passes'] - before['passes']} passes")
h = get("/health"); check("server idle and healthy afterwards", h["status"] == "ok" and h["d1"]["queued"] == 0)

print("\nALL PASS" if not fails else "\nFAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)
