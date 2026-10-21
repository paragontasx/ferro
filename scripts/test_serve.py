#!/usr/bin/env python3
"""End-to-end checks for `ferro-embed serve` (stdlib only).

  ferro-embed serve --model models/embeddinggemma-2-Q8_0.gguf --mmproj models/mmproj-embeddinggemma-2-Q8_0.gguf --port 8088 &
  python3 scripts/test_serve.py [--url http://127.0.0.1:8088] [--api-key K]

Ground truth = the CLI (`embed-media`), which scripts/eg2_parity_mm.py validates against llama.cpp (gate: cosine >= 0.999).
Server vs CLI must agree to cosine >= 0.9999 (they share the engine; the only difference is how sequences are packed).
"""
import argparse, base64, json, math, os, struct, subprocess, sys, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8088')
ap.add_argument('--api-key', default='')
ap.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ap.add_argument('--no-cli', action='store_true', help='skip the CLI cross-checks (no models on this machine)')
a = ap.parse_args()
ROOT = os.path.abspath(a.root); MEDIA = os.path.join(ROOT, 'bench/d1/media')
MODEL = os.path.join(ROOT, 'models/embeddinggemma-2-Q8_0.gguf'); MMPROJ = os.path.join(ROOT, 'models/mmproj-embeddinggemma-2-Q8_0.gguf')
fails = []

def call(path, body=None, method=None, key=None, raw=False, headers=None):
    req = urllib.request.Request(a.url + path, data=None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode()),
                                 method=method or ('POST' if body is not None else 'GET'))
    req.add_header('Content-Type', 'application/json')
    k = a.api_key if key is None else key
    if k: req.add_header('Authorization', 'Bearer ' + k)
    for hk, hv in (headers or {}).items(): req.add_header(hk, hv)
    try:
        with urllib.request.urlopen(req, timeout=600) as r: data = r.read(); code = r.status
    except urllib.error.HTTPError as e: data = e.read(); code = e.code
    return code, (data if raw else json.loads(data))

def cos(x, y): return sum(p * q for p, q in zip(x, y)) / (math.sqrt(sum(p * p for p in x)) * math.sqrt(sum(q * q for q in y)))
def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}"); 
    if not ok: fails.append(name)
def cli(items, fps=None):
    out = '/tmp/_srv_ref.bin'
    cmd = [os.path.join(ROOT, 'ferro-embed'), 'embed-media', '--model', MODEL, '--mmproj', MMPROJ, '--out', out]
    for it in items: cmd += ['--item', it]
    subprocess.run(cmd, check=True, capture_output=True, env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(ROOT, 'baselines/llama.cpp/build/bin')))
    raw = open(out, 'rb').read(); return list(struct.unpack('<%df' % (len(raw) // 4), raw))
def data_url(path, mime): return f'data:{mime};base64,' + base64.b64encode(open(path, 'rb').read()).decode()
def emb(resp, i=0): return resp['data'][i]['embedding']

print('endpoints')
c, h = call('/health', key=''); check('GET /health', c == 200 and h['status'] == 'ok', f"model={h.get('model')} multimodal={h.get('multimodal')}")
c, m = call('/v1/models'); check('GET /v1/models', c == 200 and m['data'][0]['dimensions'] == 768, str(m['data'][0]['modalities']))
mm = h.get('multimodal')

print('text')
T = 'a photo of two cats lying on a pink couch'
c, r = call('/v1/embeddings', {'model': 'x', 'input': T})
check('single text -> 768 floats, unit norm', c == 200 and len(emb(r)) == 768 and abs(math.sqrt(sum(v * v for v in emb(r))) - 1) < 1e-3, f"usage={r.get('usage')}")
if not a.no_cli:
    check('text matches the CLI', cos(emb(r), cli([f'text:{T}'])) >= 0.9999, f"cos={cos(emb(r), cli([f'text:{T}'])):.6f}")
texts = ['the quick brown fox jumps over the lazy dog', 'embeddings turn text into vectors', 'a photo of two cats lying on a pink couch', 'x', 'Ωmega ünïcode テスト ' * 30]
c, rb = call('/v1/embeddings', {'input': texts})
check('batch of 5 -> 5 vectors in order', c == 200 and [d['index'] for d in rb['data']] == list(range(5)), f"prompt_tokens={rb['usage']['prompt_tokens']}")
worst = 1.0
for i, t in enumerate(texts):
    c, r1 = call('/v1/embeddings', {'input': t}); worst = min(worst, cos(emb(rb, i), emb(r1)))
# a sequence >= 96 tokens that runs alone uses the dense-attention kernels, inside a packed batch the lane kernels: int8 rounding differs ~1e-4
check('batched vs alone (cosine; dense-vs-lane kernel noise only)', worst >= 0.9995, f"worst={worst:.6f}")
perm = [texts[i] for i in (3, 0, 4, 2, 1)]
c, rp = call('/v1/embeddings', {'input': perm})
same = min(cos(emb(rb, orig), emb(rp, k)) for k, orig in enumerate((3, 0, 4, 2, 1)))
check('a text does not depend on its batch neighbours / order', same >= 0.999999, f"worst={same:.7f}")
c, rdup = call('/v1/embeddings', {'input': [texts[0]] * 3})
check('duplicates inside one batch are identical', cos(emb(rdup, 0), emb(rdup, 2)) >= 0.999999)
c, r64 = call('/v1/embeddings', {'input': T, 'encoding_format': 'base64'})
v64 = list(struct.unpack('<768f', base64.b64decode(emb(r64)))); check('encoding_format=base64 (what the OpenAI SDK sends by default)', c == 200 and cos(v64, emb(r)) > 0.99999)
c, rd = call('/v1/embeddings', {'input': T, 'dimensions': 256})
v = emb(rd); check('dimensions=256 (Matryoshka) truncates + renormalises', len(v) == 256 and abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-3 and cos(v, emb(r)[:256]) > 0.9999)
long_text = 'word ' * 1500
c, rl = call('/v1/embeddings', {'input': long_text}); check('1500-word text (> 1024 tokens, GPU dense path)', c == 200, f"tokens={rl.get('usage', {}).get('prompt_tokens')}")

if mm and not a.no_cli:
    print('multimodal (vs CLI)')
    jpg = os.path.join(MEDIA, 'cats.jpg'); wav = os.path.join(MEDIA, 'jfk.wav'); mp4 = os.path.join(MEDIA, 'clip_cats.mp4')
    cases = [
        ('image', [{'type': 'image_url', 'image_url': {'url': data_url(jpg, 'image/jpeg')}}], [f'image:{jpg}']),
        ('audio', [{'type': 'input_audio', 'input_audio': {'data': base64.b64encode(open(wav, 'rb').read()).decode(), 'format': 'wav'}}], [f'audio:{wav}']),
        ('text + image', [{'type': 'text', 'text': 'cats on a couch'}, {'type': 'image_url', 'image_url': data_url(jpg, 'image/jpeg')}], ['text:cats on a couch', f'image:{jpg}']),
        ('video', [{'type': 'video_url', 'video_url': {'url': data_url(mp4, 'video/mp4')}}], [f'video:{mp4}']),
    ]
    for name, parts, cli_items in cases:
        t0 = time.time(); c, rr = call('/v1/embeddings', {'input': [parts]}); dt = time.time() - t0
        ok = c == 200 and cos(emb(rr), cli(cli_items)) >= 0.9999
        check(name, ok, f"{dt:.1f}s tokens={rr.get('usage', {}).get('prompt_tokens') if c == 200 else rr}  cos={cos(emb(rr), cli(cli_items)):.6f}" if c == 200 else str(rr))
    # mixed list: text strings and a multimodal item in one request
    c, rm = call('/v1/embeddings', {'input': ['hello world', [{'type': 'image_url', 'image_url': {'url': data_url(jpg, 'image/jpeg')}}]]})
    check('mixed list (string + image) -> 2 vectors', c == 200 and len(rm['data']) == 2)

print('errors and auth')
c, e = call('/v1/embeddings', b'{not json'); check('invalid JSON -> 400', c == 400 and 'error' in e)
c, e = call('/v1/embeddings', {}); check('missing input -> 400', c == 400)
c, e = call('/v1/embeddings', {'input': []}); check('empty input array -> 400', c == 400)
c, e = call('/v1/embeddings', {'input': [[1, 2, 3]]}); check('token-id input -> 400', c == 400)
c, e = call('/v1/embeddings', {'input': [[{'type': 'image_url', 'image_url': {'url': 'http://example.com/x.png'}}]]}); check('remote image URL refused -> 400', c == 400, e['error']['message'][:60])
c, e = call('/v1/embeddings', {'input': [[{'type': 'image_url', 'image_url': {'url': 'file:///etc/passwd'}}]]}); check('file:// refused -> 400', c == 400)
c, e = call('/v1/embeddings', {'input': [[{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]]}); check('undecodable image -> 400 (not a crash)', c == 400, str(e.get('error', {}).get('message'))[:50])
c, e = call('/v1/embeddings', {'input': 'x', 'dimensions': 9999}); check('dimensions out of range -> 400', c == 400)
c, e = call('/nope'); check('unknown path -> 404', c == 404)
c, e = call('/v1/embeddings', method='GET'); check('GET /v1/embeddings -> 405', c == 405)
if a.api_key:
    c, e = call('/v1/embeddings', {'input': 'x'}, key='wrong'); check('wrong API key -> 401', c == 401)
c, h2 = call('/health', key=''); check('server still healthy after the abuse', c == 200 and h2['status'] == 'ok', f"requests={h2['requests']} errors={h2['errors']}")

print('\nALL PASS' if not fails else '\nFAILED: ' + ', '.join(fails))
sys.exit(1 if fails else 0)
