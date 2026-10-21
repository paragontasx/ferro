#!/usr/bin/env python3
"""Load test for `ferro-embed serve`: N keep-alive clients, each sending `--reqs` text embedding requests back to back.

  python3 scripts/bench_serve.py --url http://127.0.0.1:8088 --conc 1,4,16,32 --words 120
Reports requests/s, tokens/s (server-side prompt_tokens) and latency p50 / p95 per concurrency level.
"""
import argparse, http.client, json, random, statistics, threading, time, urllib.parse

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8088'); ap.add_argument('--api-key', default='')
ap.add_argument('--conc', default='1,4,16,32'); ap.add_argument('--reqs', type=int, default=12, help='requests per client')
ap.add_argument('--words', type=int, default=120, help='words per text (~1.3 tokens each)')
ap.add_argument('--inputs', type=int, default=1, help='texts per request')
a = ap.parse_args()
u = urllib.parse.urlparse(a.url)
WORDS = "the cat sat on a warm sunny windowsill while rain fell softly over quiet streets and distant trains hummed through valleys".split()

def client(seed, n, lat, toks, errs):
    rnd = random.Random(seed); c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    hdr = {'Content-Type': 'application/json'}
    if a.api_key: hdr['Authorization'] = 'Bearer ' + a.api_key
    for _ in range(n):
        body = json.dumps({'input': [' '.join(rnd.choice(WORDS) for _ in range(a.words)) for _ in range(a.inputs)] if a.inputs > 1 else ' '.join(rnd.choice(WORDS) for _ in range(a.words))})
        t0 = time.perf_counter(); c.request('POST', '/v1/embeddings', body, hdr); r = c.getresponse(); d = r.read(); dt = time.perf_counter() - t0
        if r.status != 200: errs.append(r.status); continue
        lat.append(dt); toks.append(json.loads(d)['usage']['prompt_tokens'])
    c.close()

print(f"{a.words} words/text (~{int(a.words*1.3)} tok), {a.inputs} text(s)/request, {a.reqs} requests per client")
print(f"{'clients':>8} {'req/s':>8} {'tok/s':>8} {'p50 ms':>8} {'p95 ms':>8} {'errors':>7}")
for conc in [int(x) for x in a.conc.split(',')]:
    client(0, 2, [], [], [])                                                   # warm-up
    lat, toks, errs = [], [], []
    ths = [threading.Thread(target=client, args=(i, a.reqs, lat, toks, errs)) for i in range(conc)]
    t0 = time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.perf_counter() - t0
    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] * 1e3 if lat else float('nan')
    print(f"{conc:8d} {len(lat)/wall:8.1f} {sum(toks)/wall:8.0f} {p(0.5):8.0f} {p(0.95):8.0f} {len(errs):7d}")
