<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <img src="assets/logo.png" alt="ferro - geometric inference engine" width="420">
  </picture>
</p>

# Ferro

**A tiny inference engine for Intel low-power SoCs (N100-class).** One C++ binary, no Python, no
framework — it loads GGUFs directly and runs them **on the integrated GPU by default** (OpenCL, int8
DP4A), with automatic CPU fallback. Three model kinds from the same binary, detected from GGUF metadata:

| Model | Arch (GGUF) | Command | What it returns |
|---|---|---|---|
| **EmbeddingGemma-2** | `gemma-embedding2` | `embed` / `bench` | sentence embeddings (+ image / audio / video via `--mmproj`) |
| **LiquidAI d1-omni-600M** | `lfm2` + `lfm2.decision.*` | `decide` / `bench` | typed answers (yes/no, choice, score) over a state, one forward pass, no generated tokens |
| **LiquidAI LFM2 / LFM2.5** (1.2B-Thinking, 350M, ...) | `lfm2` (causal) | `generate` / `serve --lm` | text generation: chat + completions, streaming, OpenAI-compatible |

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![language: C++](https://img.shields.io/github/languages/top/paragontasx/ferro)
![iGPU by default](https://img.shields.io/badge/iGPU%20by%20default-OpenCL%20%C2%B7%20int8%20DP4A-green)
![OpenAI--compatible](https://img.shields.io/badge/OpenAI--compatible-server-informational)
![no dependencies](https://img.shields.io/badge/runtime%20deps-none%20(libOpenCL%20dlopen'd)-lightgrey)

**Why:** on a ~6 W N100 box, ferro beats llama.cpp's own iGPU backend by ~2x (measured 2.1-2.3x on
embeddings and decisions), and 1.6-1.7x on agent-loop TTFT through its prefix cache — while keeping an
exactness contract llama.cpp does not offer: a stream decodes identically alone or among any number of
streams, speculative decoding is provably a no-op, and every number on this page is reproducible from
[`bench/RESULTS.md`](bench/RESULTS.md) with the scripts in `scripts/`.

### Quick start (2 minutes)

```sh
make -j4                          # needs OpenCL headers (CL_INC=...); Mesa rusticl is enough
make test                         # kernel + decision + serving tests
./ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --d1 models/d1-omni-600M-Q8_0.gguf \
                    --lm-slots 4 --lm-ctx 4096 --kv-dtype f32 --prefix-cache-mb 256 --port 8090
curl localhost:8090/v1/chat/completions -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"stream":true}'
# then open http://localhost:8090/chat for the built-in test page
```

Models are plain GGUFs (`--model kind` is auto-detected); get e.g. d1 from
`https://huggingface.co/LiquidAI/d1-omni-600M-GGUF`.

### Contents

[What's new](#whats-new-2026-10-22-tag-validated-2026-10-22-2) ·
[Results](#results-intel-n100-same-model-files--inputs-as-llamacpp-interleaved-runs) ·
[d1 decision model](#using-d1-omni-decision-model) ·
[EmbeddingGemma-2 multimodal](#using-embeddinggemma-2-multimodal-text--image--audio--video---one-768-d-embedding) ·
[LFM2.5 chat](#using-lfm25-language-models-chat--completions) ·
[HTTP server](#serving-over-http-ferro-embed-serve) ·
[Exactness & ops](#exactness-scheduling-and-operations-lm-serving) ·
[How it's fast](#how-the-igpu-path-is-fast) ·
[Build & run](#build--run) ·
[CPU path](#cpu-path) ·
[Code layout](#layout) ·
[Env knobs](#env-knobs)

## What's new (2026-10-22, tag `validated-2026-10-22-2`)

- **Split-K decode attention** (default): context is no longer a tax. A 1.9k-token prompt decodes
  27-45% faster - 350M at ~1.9k context: 31.7 -> **40.2 tok/s** (f32 KV), 26.8 -> **38.8** (int8).
  Per attention layer at 1921 keys: 2.82 -> 1.20 ms (f32), 3.76 -> 1.30 ms (int8); a row's output stays
  bit-identical alone, in any batch order, with any launch stride (speculative windows stay exact).
- **fp16 KV cache** (`--kv-dtype f16`, opt-in): half the storage of f32 with fp32 arithmetic
  (4 x 4096: 192 MB vs 108 int8 / 384 f32). After split-K, f32 is both the fast *and* the accurate
  default - int8 KV is no longer faster anywhere measured.
- **End-of-prompt prefix snapshots ("tails")**: an agent turn now restores the *previous prompt
  exactly* and prefills only the new reply + tool output - **1.6-1.7x lower TTFT per turn**. Tails
  are their own cache class: one per conversation, quarter-budget cap, skipped rather than evicting
  a shared entry, never written to disk.
- **CPU pool parking**: idle workers futex-park after a 200 us spin - an idle server now draws
  **0% of a core** (it used to burn one fully, 62% in-kernel, inside the shared CPU+iGPU power budget).
- Repetition penalty gained llama.cpp-convention `repeat_last_n` windows and `--penalty-scope generated`
  for agents that must copy ids/paths/JSON keys out of their prompt.
- Built-in **`GET /chat`** test page (single file, no external requests): streaming chat + per-reply
  stats + a d1 "Decide" tab with image/audio upload.

Validation for the whole branch: llama.cpp-oracle parity top-1 **12/12 on f32, int8 and f16** KV,
batch invariance **ALL PASS** at 6 and 8 concurrent streams (temperature 0 and 0.8), speculative-window
suites all pass, `make test` green. Per-window logs: `bench/RESULTS.md`.

## Results (Intel N100, same model files + inputs as llama.cpp, interleaved runs)

**EmbeddingGemma-2** (real embedding path):

| Engine | 16 seqs / 188 tok | 64 seqs / 752 tok |
|---|---|---|
| **ferro-embed (iGPU, default)** | **219 ms · 839 tok/s** | **796 ms · 938 tok/s** |
| llama.cpp Vulkan (same iGPU)¹ | 461 ms · 408 tok/s | 1757 ms · 428 tok/s |
| llama.cpp CPU (`llama_encode`, 2 t)¹ | 586 ms · 320 tok/s | 2214 ms · 340 tok/s |
| ferro-embed `--cpu` | ~900–1000 ms · ~175–206 tok/s | ~4000 ms · ~174 tok/s |

¹ llama.cpp rows measured on 2026-10-20 (not re-run after ferro's 2026-10-21 GEMM fix, which took ferro from
627 to ~840–940 tok/s: about 2.1× Vulkan, 2.7× CPU against those numbers). `--hetero` (CPU ‖ iGPU) is not
re-measured after the fix.

**d1-omni-600M Q8_0** (16 support tickets × 3 questions = 48 decision sequences, 2,989 tokens;
llama.cpp = `llama-server /v1/systemone`):

| Engine | all 16 at once | one request at a time |
|---|---|---|
| **ferro-embed (iGPU, default)** | **5.77 s · 518 tok/s** | **5.99 s · 499 tok/s (0.37 s/request)** |
| llama.cpp Vulkan (same iGPU, `-np 8`) | 10.98 s · 272 tok/s | 13.74 s · 217 tok/s |
| ferro-embed `--cpu` | 41.3 s · 72 tok/s | 42.1 s · 71 tok/s |
| llama.cpp CPU (2 t, `-np 8`) | 62.9 s · 48 tok/s | 60.1 s · 50 tok/s |

d1 iGPU: **1.9× (all at once) / 2.3× (one at a time) llama.cpp's own iGPU backend, 10–11× llama.cpp CPU**
(interleaved scoreboard 2026-10-21, `bench/d1/scoreboard_2026-10-21.log`); ferro's CPU path is also
1.4–1.5× llama.cpp CPU here. Answers match llama.cpp: identical prompt tokens on 15/15 requests, same top
answer 23/23, and every answer is bit-identical whether a request runs alone or batched.
Details: `bench/RESULTS.md`.

## Using d1-omni (decision model)
Request format = llama.cpp's `/v1/systemone` (Decision Index schema); output = the same response JSON.
```sh
cat > req.json <<'EOF'
{"state": "I was charged twice this month, please refund one of them.",
 "questions": {
   "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
   "team":    {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults"}},
   "urgency": {"type": "score",  "instructions": "How urgent is this?",
               "criteria": ["Can wait", "Today", "Blocking the customer now"]}}}
EOF
./ferro-embed decide --model models/d1-omni-600M-Q8_0.gguf --request req.json      # or --request - (stdin)
# {"model":"d1-omni-600M","answers":{"refund":{"type":"noul","noul":0.998...},"team":{"type":"choice",
#   "choice":"billing","probabilities":{...},"confidence":0.98...},"urgency":{"type":"score","score":1.80...,...}},
#   "usage":{"input_tokens":...,"output_tokens":0}}
```
A JSON **array** of requests is answered in one batched pass (output = array). States may be text, any
JSON value, or `null`; `noul` criteria accept `true/false` or `yes/no`; up to 255 options. Answers are
calibrated with the per-type temperatures stored in the GGUF. Get the model:
`curl -L -o models/d1-omni-600M-Q8_0.gguf https://huggingface.co/LiquidAI/d1-omni-600M-GGUF/resolve/main/d1-omni-600M-Q8_0.gguf`

**Media (image / audio):** `files` (or `images`) take data URLs; chat-message `image_url` / `input_audio`
parts in `state` are also accepted. Either images or **one** audio clip (WAV / MP3 / FLAC, any sample rate,
cut to 30 s), never both; remote URLs and `videos` are rejected exactly as llama.cpp does (extract frames
with ffmpeg and send them as images). The mmproj (`models/mmproj-d1-omni-600M-Q8_0.gguf`, fetched by
`scripts/fetch_assets.sh`) is found next to the model. Encoders: SigLIP vision tower with LFM2-VL tiling
(whole encoder on the iGPU) and a FastConformer audio encoder (iGPU GEMMs + AVX2 conv / rel-pos attention).
Media embeddings are computed once per request and shared by all of its questions. Like llama.cpp, media
answers are uncalibrated (temperature 1). `ferro-embed mm-embed --mmproj M --media F --out f.bin` dumps the
embeddings (same format as `bench/mm/mm_ref`, the libmtmd oracle). Parity vs llama-server `--mmproj`
(`scripts/d1_parity_mm.py`, 12 requests): token counts 12/12, top answers 18/18, worst probability gap 0.019.
Encoder embeddings match libmtmd to mean cosine >= 0.9992 (images; llama.cpp's own flash-attention vs
non-flash outputs differ by about the same, worst token 0.97) and >= 0.9996 (audio).

Tested with the Q8_0 GGUF; the loader also accepts F16/BF16/F32 tensors (quantized to Q8 at load for the
int8 path), but that is untested.

## Using EmbeddingGemma-2 multimodal (text + image + audio + video -> one 768-d embedding)
EmbeddingGemma-2 is natively multimodal: the text model is `embeddinggemma-2-*.gguf`, the vision (gemma4v ViT) and audio
(gemma4a conformer) encoders ship in `mmproj-embeddinggemma-2-*.gguf` (fetched by `scripts/fetch_assets.sh`). Items are
interleaved in the order given; video is ffmpeg frames (4 fps by default) with `[XmY.YYs]` timestamps, exactly as llama.cpp does.
```sh
./ferro-embed embed-media --model models/embeddinggemma-2-Q8_0.gguf --mmproj models/mmproj-embeddinggemma-2-Q8_0.gguf \
    --item text:"two cats on a sofa" --item image:cats.jpg --item audio:clip.wav --item video:clip.mp4 --out emb.bin   # f32[768], L2-normalised
```
Both encoders and the text model run on the iGPU (the audio conformer blocks - GEMMs, chunked local attention, conv module, norms - stay on the device; only the SSCP conv front-end and the tail run on the CPU); the CPU path is a working fallback (`FERRO_GPU=0`, `FERRO_MM_GPU=0`).
Audio is cut into 30 s pieces inside one `<|audio> .. <audio|>` span (as mtmd does). Measured on the N100 (iGPU, quiet machine,
`embed-media` time = encoders + text model, excludes process start and weight load):

| Input | Tokens | iGPU | CPU fallback (encoder) |
|---|---|---|---|
| text | 12 | 0.08 s | |
| image 100x80 | 84 | 1.5 s | 9 s |
| image 640x480 | 134 | 2.8 s | 17 s |
| image 1920x1080 | 924 | ~74 s | does not finish |
| audio 11 s | 279 | 1.4 s | 6.9 s |
| audio 44 s (two 30 s chunks) | 1104 | 6.5 s | 46 s |
| video 2 s, 8 frames | 669 | 12 s | |

llama.cpp's CPU encoder needs 183 s for the 1080p photo (ferro 67 s encode); its Vulkan path was not benchmarked for these encoders.
**Parity** (`scripts/eg2_parity_mm.py`: final embedding cosine vs llama-server `/v1/embeddings`, gate 0.999, 12 cases, all pass): text .99995,
images .99936-.99985, audio .99992 (mp3 .99992, flac .99995, 44 s .99992), text+image .9993, text+image+audio .9998, video .99965;
token counts identical to the server. Per-chunk encoder cosine vs libmtmd (`bench/mm/mm_ref`): audio mean .99987 on the CPU path /
.99985 on the iGPU (conformer blocks fully on-device), vision mean .9962 (.9972 on the CPU path): int8 activations compound smoothly over the 16 ViT layers
(layer-wise relative error 0.24% -> 6.4%, no step anywhere); the pooled embedding averages it out.
Limits: video frames are upscaled to the model's 70-token minimum image (a 320x240 frame costs 1.3 s), so video is ~1.5 s/frame;
large images are bound by ViT attention (88% of the 1080p time); text/audio sequences up to ~2900 tokens use the iGPU,
beyond that the layer loop falls back to the CPU. Text-model attention for single long sequences uses head-dim generic dense
kernels (5x faster than the lane-per-key kernels at 950 tokens).

## Using LFM2.5 language models (chat / completions)
`LiquidAI/LFM2.5-1.2B-Thinking` (and the other causal `lfm2` GGUFs: Instruct, Base, ...) run natively: 10 gated short-conv blocks + 6 GQA
attention blocks, SwiGLU, tied LM head, byte-level BPE, the model's own chat template (ported natively, byte-identical to llama.cpp's jinja
rendering incl. tools and the past-turn `<think>` stripping) and a sampler (temperature / top-k / top-p / min-p / repetition penalty / seed;
defaults are LiquidAI's recommended 0.05 / 50 / 1.05). Q8_0, F16, BF16 and F32 GGUFs load (non-Q8 weights are quantised to Q8_0 on load).
```sh
./ferro-embed generate --lm models/LFM2.5-1.2B-Thinking-Q8_0.gguf --prompt "Explain why the sky is blue in two sentences." --n-predict 400
./ferro-embed generate --lm models/LFM2.5-1.2B-Thinking-Q8_0.gguf --messages chat.json --temp 0.1   # chat.json: [{"role":..,"content":..}] or {"messages":..,"tools":..}
```
Everything runs on the iGPU: Q8 weights stay device-resident, the KV cache (attention layers) and the 2-token conv state live in per-sequence
device slots, prefill uses a causal tiled attention kernel, decode batches one token of every active sequence into a single pass
(split-K GEMV for the 1-4 token GEMMs, one-query-per-row attention over the cache). Measured on the N100 (Q8_0, same file as llama.cpp),
ferro on its default GPU stack — Intel NEO/compute-runtime, `FERRO_GPU_DRIVER=neo` (Mesa rusticl remains a one-variable fallback):

| LFM2.5-1.2B-Thinking Q8_0 | ferro iGPU (2026-10-25) | llama.cpp CPU, 2 threads (2026-10-20) |
|---|---|---|
| prefill, 784 tokens | **222 tok/s** | 11.2 tok/s (**19.8×**) |
| prefill, 3,626 tokens | **143 tok/s** | not measured |
| decode, 1 stream, short ctx | **21 tok/s** | 6.1 tok/s (**3.4×**) |
| decode, 1 stream, @ 3.6k ctx | **15.5 tok/s** | not measured |
| decode, aggregate over 2 / 4 streams (server) | 22 / 35 tok/s (2026-10-20) | |

**LFM2.5-350M** (`LiquidAI/LFM2.5-350M-GGUF`, Q8_0, same code path, no model-specific changes) measured the same way:

| LFM2.5-350M Q8_0 | ferro iGPU (2026-10-25) | llama.cpp CPU, 2 threads (2026-10-20) |
|---|---|---|
| prefill, 784 tokens | **748 tok/s** | 66.7 tok/s (**11.2×**) |
| prefill, 3,626 tokens | **488 tok/s** | not measured |
| decode, 1 stream, short ctx | **57 tok/s** | 31.8 tok/s (**1.8×**) |
| decode, 1 stream, @ 3.6k ctx | **36.5 tok/s** | not measured |
| decode, aggregate over 2 / 4 / 8 streams (server) | 58 / 98 / 132 tok/s (2026-10-20) | |

The llama.cpp CPU column was measured on 2026-10-20 and is kept as-is — llama.cpp has not been re-run since, so those are the same
numbers a reader can reproduce with the pinned build; the ferro column was re-measured on 2026-10-25 on the same box, single tenant.
The jump over the previous ferro column (1.2B prefill 153→222 tok/s, decode 15-19→21; 350M decode 36-37→57, prefill 490→748) comes from
switching the GPU driver stack from Mesa rusticl to Intel's NEO/compute-runtime and from two kernel fixes in the prefill attention path:
a `KT5` key-chunk retune and a `#pragma unroll 8` on the QK inner loop (full unroll had pushed the query vector into 64 GRF/lane, which
sat exactly at the 128-register limit and spilled under Intel's compiler). Per-driver and per-kernel numbers: `bench/RESULTS.md`,
2026-10-24/25 entries.

**Long-context decode, old vs new binary** (2026-10-22, same file: 64 tokens greedy, `FERRO_LM_STEPTIME`,
median of the 64 steps, nothing else running; "old" predates split-K attention and the pool fix; the
table above is short-context and predates the change too):

| ~1.9k-token prompt | f32 KV | int8 KV | f16 KV (new) |
|---|---|---|---|
| old binary | 31.7 tok/s | 26.8 tok/s | — |
| **new (default)** | **40.2** | **38.8** | 35.1 |

Split-K pays only with context (+27% / +45% at 1.9k tokens, nothing at 200) — the old kernel was
latency-bound, which is also why int8 KV used to beat f32 at decode and now does not. Post-change
parity against the llama.cpp oracle: top-1 12/12 on all three dtypes (per-prompt TV ≤ 0.06: 12/12
f32, 10/12 int8, 11/12 f16 — outliers are the known quantisation-sensitive prompts), and
`test_batch_invariance.py` ALL PASS at 6 and 8 streams (temperature 0 and 0.8).

Parity vs llama.cpp Q8_0 on the same weights: top-1 12/12, mean TV 0.017 (every prompt < 0.06). This model is far more sensitive to Q8 than the 1.2B (llama.cpp's own Q8 vs F16 is TV 0.040; ferro vs F16 0.043; both flip the same near-tie on an out-of-distribution `<think>` prompt), so compare against the same-weights Q8 oracle for it. Decode streams ~510 MB/token (int8 + fp32 scales) = 19.7 ms at the 25.9 GB/s ceiling against a ~26 ms step; the rest is ~70 small-kernel launches at ~40 us each (after the default conv/qkv fusions), so single-stream decode sat ~1.2x llama.cpp CPU before the 2026-10-22 attention fix (at ~1.9k tokens it now runs ~40 vs 32 tok/s; llama.cpp CPU not re-measured at long context); prefill and batched serving are where the iGPU wins (details: `bench/RESULTS.md`, "Decode kernel fusion study").
The chat template is read from each GGUF: the 350M template walks content arrays itself (text parts concatenated, system arrays too) while the Thinking template does not (llama.cpp's server joins parts with `\n` there); `serve` logs the detected variant and `/apply-template` is byte-identical to llama.cpp's jinja for both.

Decode streams 1.17 GB of weights per token at ~22 GB/s; a pure-streaming read kernel reaches 25.9 GB/s on this iGPU, so Q8 decode is ~85% of the
hardware limit (only a smaller weight format can go meaningfully faster). **Accuracy** (`scripts/lm_parity.py`, 12 prompts incl. chat-template,
code and a 900-token one; next-token distribution vs llama.cpp **F16** as the oracle): top-1 agreement 12/12, mean total-variation distance **0.0198** -
the same as llama.cpp's own Q8_0 vs its F16 (0.0200), i.e. at the quantisation noise floor. At the hidden-state level (3 / 8 layers, per-token relative L2
vs F16) ferro's fp32-activation CPU reference is *closer* to F16 than llama.cpp's int8-activation Q8 path (1.1 / 1.4% vs 1.9 / 2.6%); the iGPU path quantises
activations to int8 like llama.cpp and lands on the same TV. Debug knobs: `FERRO_LM_GPU=0` (CPU reference, ~0.5 tok/s decode), `FERRO_LM_PROF=1` (per-op device times),
`FERRO_LM_STEPTIME=1`, `FERRO_LM_SELFCHECK=1` (chunked prefill / token-by-token decode vs one prefill), `FERRO_LM_KEEP_HOST=1`, `FERRO_GPU_SMALL=0` / `FERRO_GPU_GEMV=0`.
Fusion / timing-experiment knobs: `FERRO_LM_FUSE=0` (unfused launches), `=2` (opt-in prologue-fused GEMVs, break-even on this GPU), `FERRO_PRO_MASK`, `FERRO_LM_ABL=op,..` / `FERRO_LM_DUP=op,..` (skip / double ops to measure their cost), `FERRO_LM_NOEOS=1`.

## Serving over HTTP (`ferro-embed serve`)
One resident process, OpenAI-compatible, embeddings and/or language model and/or the d1 decision model on the same iGPU (one worker thread owns the device):
```sh
./ferro-embed serve --model models/embeddinggemma-2-Q8_0.gguf --mmproj models/mmproj-embeddinggemma-2-Q8_0.gguf \
                    --lm models/LFM2.5-1.2B-Thinking-Q8_0.gguf --d1 models/d1-omni-600M-Q8_0.gguf --port 8088 [--lm-slots 4] [--lm-ctx 4096] [--kv-dtype f32|int8|f16] [--spec-draft 6] [--snapshot-dir D] [--api-key 'label: key']... [--host 0.0.0.0]
./ferro-embed plan <same flags>      # what this configuration needs vs the container limit, before anything is loaded
curl localhost:8088/v1/chat/completions -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"stream":true}'
curl localhost:8088/v1/embeddings -d '{"input":"hello"}'
```
| Endpoint | |
|---|---|
| `POST /v1/embeddings` | text / `image_url` / `input_audio` / `video_url` parts (data URLs), `dimensions`, `encoding_format: base64`; concurrent text requests share packed forward passes |
| `POST /v1/chat/completions`, `POST /v1/completions` | `messages` (+`tools` are rendered into the prompt), `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` (+ `repeat_last_n`, `penalize_prompt`: below), `seed`, `stop`, `stream` (SSE) + `stream_options.include_usage`; `usage` and llama.cpp-style `timings` in every response |
| `POST /v1/systemone` | d1-omni typed decisions (choice / score / noul) over a text, image or audio state - llama.cpp's contract, answers identical to `ferro-embed decide`; queued requests share one forward pass; `crop` / `crops` regions (below) |
| `POST /apply-template` | rendered chat prompt (same contract as llama.cpp) |
| `POST /tokenize`, `POST /detokenize` | token ids for a `prompt` / `messages` (chat template applied) and back; vLLM's fields |
| `GET /metrics` | Prometheus text (counters, live tokens/s, TTFT histogram, memory); needs a key when keys are set |
| `GET /v1/models`, `GET /health` | capabilities / context / slots; queue, active slots, throughput counters |
| `GET /chat` | built-in single-file test page (no external requests): streaming chat with reasoning display and per-reply stats (cached tokens, TTFT, tok/s, warnings), Stop, settings; a "Decide" tab posts to `/v1/systemone` with image/audio upload |

`<think>..</think>` output is returned as `reasoning_content` (`--reasoning-format none` keeps it inline). Generation is continuously batched over `--lm-slots`
device slots (each holds `--lm-ctx` tokens of KV cache, ~25 MB per 1k tokens); embedding requests are served between decode steps (a short text embedding
measured 0.12-1.8 s while 4 streams were generating - it waits for the decode step or prefill in progress); a closed client connection frees its slot within a step; 503 when the queue is full; Bearer `--api-key`
(or `FERRO_API_KEY`). Not implemented: tool-call *parsing* of model output (tools are only rendered into the prompt), `n > 1`, logprobs, image input to the LM.

**Repetition penalty window.** The default is LiquidAI's chat recommendation, `repetition_penalty` 1.05 over the whole context (prompt included, as Hugging Face). An agent that must copy ids, paths or JSON keys out of its prompt can keep the penalty off the prompt with `--penalty-scope generated` (server default) or `"penalize_prompt": false` (per request); `--repeat-last-n N` / `repeat_last_n` (alias `penalty_last_n`) limits the window to the last N tokens (-1 = whole context, 0 = off, llama.cpp's convention) and `--repeat-penalty P` sets the server default. Speculative windows sample every row with the same window, so drafting stays exact.

**Prompt-prefix cache (`--prefix-cache-mb`, default 128, 0 = off).** Digest-style workloads send the same long system prompt every time. The server snapshots the LM state (K/V rows + conv state) at the end of a leading system
message (and at any prefix a request shares with a recent one - plain `/v1/completions` prompts too), keeps the snapshots on the device in an LRU cache, and later requests restore the longest matching snapshot into their slot and prefill only the rest.
It is exact (causal state only, restore == prefill within the usual chunking noise; greedy answers identical in the tests) and visible: `usage.prompt_tokens_details.cached_tokens` (also `timings.cache_n`, `cached=N` in the log, `/health` `lm.prefix_cache`).
Measured on a 1400-token system prompt + 8 different ~30-token digests (24 generated tokens each): per request 4.3 s -> 1.1 s on the 350M (3.9x warm, 2.8x over all 8 incl. the cold first) and 12.4 s -> 2.3 s on the 1.2B (5.5x, 3.5x over 8);
the restored part costs a device-side copy (writing a 732-token snapshot measured 10-17 ms) instead of ~2.9 s of prefill. Costs: ~25 KB of device memory per cached token (a 1400-token prompt = 33 MB), and the first request of each new prefix pays ~4% for writing the snapshot (workloads that never
repeat a prefix: turn it off or lower the budget). Per request `"cache_prompt": false` neither reads nor writes the cache (llama.cpp's field name). Boundaries are exact token prefixes, so the cached part must be byte-identical:
same system text, same tools, same template variant. Chats with several messages or tools also leave a
snapshot at the end of the prompt — **only** once the conversation has actually continued (some
`assistant` or `tool` message exists), so one-shot system+user calls pay nothing; `--no-prefix-tail`
turns tails off entirely. Tails are their own LRU class: one entry per conversation lineage (a newer
tail replaces the one it extends), capped at a quarter of the cache budget, *skipped* rather than
evicting any non-tail entry — which is what keeps the shared system-prompt entry alive through an agent
loop — and never written to disk. Measured: on a 5-turn conversation (347-token system prompt)
`cached_tokens` equals the previous prompt's length from turn 3 on (370/393, 393/416, 416/439); on a
6-turn synthetic agent conversation the old build restored the prompt *minus its tail* (a turn behind)
while the new one restores it exactly, prefilling only the new reply + tool output — **1.6-1.7× lower
TTFT per turn (316-358 ms → 177-217 ms)**. `test_serve_prefix --url` passes with tails on: 20/20 at a
128 MB cache, and the `--small-cache` run against a real 8 MB instance (entries 1, 5.2 MB, 4 evictions,
budget respected). One property to watch on a tight cache: when non-tail entries fill the budget a tail
is skipped rather than evicting them, so a full cache can legitimately hold no tail — that is the cap
doing its job; raise `--prefix-cache-mb`, it is not a bug.

**Decisions over HTTP (`--d1`).** `POST /v1/systemone` takes exactly the request documented in "Using d1-omni" (single object or array) and returns the same JSON; text decisions of ~3 questions take ~0.25 s, and they stay at
~0.14 s while three chat streams decode (the worker interleaves them). **Regions:** a frame plus a detector box needs no client-side crop / JPEG re-encode. Any image entry (`files` / `images` item or `image_url` part) can be an object:
```json
{"state": "Camera 3 motion event", "questions": {"person": {"type": "noul", "instructions": "Is there a person?"}},
 "files": [{"url": "data:image/jpeg;base64,...1080p frame...",
            "crop": {"x": 900, "y": 300, "w": 400, "h": 400, "pad": 0.1}}]}
```
`crop` = `{x,y,w,h}` or `{x1,y1,x2,y2}`, `"unit": "px"` (default) or `"norm"` (0..1), `pad` = fraction of the box added on every side; boxes are clipped to the frame. `crops: [..]` sends several boxes of one frame as separate images,
`include_full: true` adds the whole frame as well (crops count towards the 8-image limit; audio rejects them). The crop is cut from the decoded pixels before the encoder, so it is identical to sending that region as its own PNG
(tested). It is also the fast path: a 400x400 crop of a 1080p frame answers in 1.3 s, the whole frame in 21.8 s (7000 vision tokens) - the server-side crop costs ~45 ms (JPEG decode) over a pre-cropped upload.
**Memory** (iGPU buffers are system RAM and not swappable - size `--lm-slots` x `--lm-ctx` to your limit; measured via cgroup, device buffers show as shmem):

| Process | anon | device (shmem) |
|---|---|---|
| `--lm` only, 4 x 4096 | ~0.6 GB | 1.8 GB |
| `--lm` only, 1 slot | ~0.8 GB | 1.5 GB |
| `--model` + `--mmproj` | ~1.6 GB | 0.8 GB |
| EG2 text only | ~0.9 GB | 0.2 GB |
| `--d1` (+ its mmproj) | ~0.9 GB | 0.7 GB |

Combinations (anon + shmem, incl. ~0.35 GB of other processes in the container): d1 1.9 GB, chat 350M 1.7 GB, EG2+mmproj 2.3 GB, d1 + chat 2.1 GB, EG2mm + d1 2.1 GB, **all four 3.4 GB**: inside a 4 GiB
container that leaves too little headroom for multimodal embedding bursts (the all-four server was OOM-killed on its first video/image embeddings) - the startup log now says so. Pair two or three models per container, or give it more RAM.

The server prints the KV size and warns when it approaches a container limit. Tests: `scripts/test_serve.py` (embeddings), `scripts/test_serve_lm.py`
(chat: usage accounting, seeds, stream == non-stream, stop strings, UTF-8, errors, cancellation, concurrency, template parity with `--llama URL`),
`scripts/test_serve_prefix.py` (prefix cache: cold vs warm, exactness vs `cache_prompt:false`, tools, learned prefixes, streaming usage, concurrency, LRU eviction), `scripts/test_serve_d1.py` (decisions: CLI parity, media, crop == pre-cropped image, error cases, concurrency / pass sharing), `scripts/bench_lm_serve.py` (concurrency scaling), `scripts/bench_serve.py` (embeddings).

Newer suites (each starts the servers it needs or takes `--url`): `scripts/test_batch_invariance.py` (solo == among N streams), `scripts/test_serve_sched.py` (chunked prefill),
`scripts/test_serve_think.py` (thinking budget; needs the 1.2B-Thinking model), `scripts/test_serve_spec.py` (speculative decoding exactness), `scripts/test_memory.py`, `scripts/test_snapshots.py`,
`scripts/test_serve_ops.py`; benchmarks `scripts/bench_spec.py`, `scripts/bench_lm_serve.py`.

## Exactness, scheduling and operations (LM serving)
Everything here is measured on the N100 iGPU with the files in `models/` (scripts named in each item are in `scripts/`).

**Exactness contract.** A stream decodes to the same tokens alone or among any number of other streams, a prompt filled in slices or restored from a
cache equals the same prompt filled at once, and speculative decoding changes nothing: the output is what one-token decoding gives.
Decode rows always use the split-K GEMV (8-row passes beyond 8 rows) and prefill rows always use the tile GEMM, so a row's arithmetic does not depend on how
many other rows share the launch; a 1-token prefill tail takes the prefill path. Before this, 2 of 4 greedy prompts differed between a solo run and a run among
5-8 streams (a different GEMM kernel, a different float summation order); now 4/4 are identical at 2..12 streams, greedy and seeded
(`scripts/test_batch_invariance.py`), and the engine self-check (`FERRO_LM_SELFCHECK=1`) shows split prefill and every cache restore bit-identical
(`max|diff| 0`, also at 641 tokens). Cost: 1-4 streams unchanged, 5 streams -5% (95 -> 91 tok/s aggregate), 8 streams -1.5%. `FERRO_ROW_POLICY=0` restores the old mix.

**Chunked prefill (`--prefill-slice-ms`, default 250; `--decode-share`, default 40%).** A long prompt used to stop everything: a d1 decision took 3855 ms
instead of 113 ms while a 1615-token prompt filled. The prompt is now filled in slices sized from the measured prefill rate (a whole prompt at once when nothing
else is waiting), and after each slice the decode streams get their share of the time; d1 and embedding work run between slices. During that prefill: d1
2187 -> 170-196 ms (113 alone), longest stall of a decoding stream 2278 -> 360 ms, a short chat 2088 -> 1041 ms; the long prompt itself takes ~38% longer under
contention (6.2 s vs 4.5 s). Same output as unsliced (`scripts/test_serve_sched.py`).

**Speculative decoding by prompt lookup (`--spec-draft N`, default 6; per request `"draft": false`).** Replies often copy the prompt (ids, names, quoted text) or
repeat themselves. The server looks for the last 2-4 generated tokens earlier in the prompt + reply and proposes the tokens that followed, then verifies them
in one forward as a window of rows that use exactly the arithmetic of serial decode (own conv kernel, conv-state rollback; window == serial decode `max|diff| 0`).
Tokens are sampled row by row with one sampler draw per produced token, and a draft is kept only while it equals the sampled token, so greedy and seeded replies
are identical with and without it, also under concurrency and when a stop string or `max_tokens` falls inside a window (`scripts/test_serve_spec.py`).
Greedy tokens/s, 350M (interleaved rounds, `scripts/bench_spec.py`): copy a list verbatim 31.6 -> 52.3 (**1.65x**, 67% of drafted tokens accepted), extract ids 1.25x,
rewrite a log 1.12x, write code 1.08x, free-form text 1.03x (no drafts attempted). The 1.2B-Thinking measured 0.99-1.08x, i.e. within noise: it spends its tokens on
reasoning prose, which does not copy. A window costs (350M, plain step 26.6 ms): 2 rows 27.1 ms, 4 rows 32.9, 5-8 rows 43-47 (1.2B: 58 / 72 / 112-129 ms), so streams start
at 3 drafted tokens and use up to `--spec-draft` only once >85% of their drafts are accepted; a stream whose drafts keep failing sits out a few steps, and a stream that
has not shown drafting works needs a 3-token match. Only with the default fused kernels (`FERRO_LM_FUSE=1`) and at most 8 active streams.

**Thinking budget (`--thinking-budget N`, request `thinking_budget`).** After N reasoning tokens a `</think>` is forced and the model answers (unbounded: 217 tokens;
budget 24: reasoning cut at 24). A reply that stops at `max_tokens` inside `<think>` has empty `content`, so it now says so: a log line, and a `warnings` array in the
response (and on the last streamed chunk). `ignore_eos: true` generates past the end token (fixed-length benchmark runs).

**KV cache (`--kv-dtype f32|int8|f16`, default f32).** K/V rows as int8 with one fp32 scale per 32 values (llama.cpp's q8_0 KV scheme), quantized in the fused
norm+rope+cache-write kernel and dequantized in the attention kernels; prefix snapshots carry the scales. 350M at 4 x 4096: KV 384 -> 108 MB (resident memory -276 MB
measured); f16 sits between (192 MB). Accuracy vs the same-weights llama.cpp oracle: top-1 12/12 either way, mean TV 0.0172 (f32) -> 0.0276 (int8), the loss concentrated on short prompts;
f16 lands between as a 10-bit mantissa predicts. Speed: before split-K decode attention, int8 cost 10-15% at a 2k context (350M 23.1 -> 19.6 tok/s) - the dequant plus the latency-bound
per-head kernel. After the 2026-10-22 fix the gap is ~4% (same-binary A/B at 1.9k tokens: f32 40.2 vs int8 38.8 vs f16 35.1 tok/s) and **f32 is both the fast and the accurate default**:
switch to int8/f16 only when you need the memory, not for speed.

**Memory planning and admission.** `ferro-embed plan <serve flags>` prints what a configuration needs from GGUF headers and file sizes alone (runtime ~700 MB + 1.3 x every
GGUF + KV + prefix cache; calibrated: 350M + d1 + EG2 + mmproj estimated 3585 MB, measured 3373 MB) against the container limit, with concrete advice when it does not fit
(`--lm-slots 3 fits`, `--kv-dtype int8 saves 1104 MB`, ...); exit code 3 = does not fit. `serve` runs the same check *before loading anything* and refuses an over-committed
configuration instead of being OOM-killed (`--no-memory-check` overrides). Requests with media are admitted only if their estimated transient memory fits the headroom right now
(reserved while in flight): 413 with guidance when nothing else is running ("send a crop / downscale / fewer frames"), 503 when waiting would help. Measured transients: d1 image
+42 / 74 / 166 / 293 MB at 0.07 / 0.23 / 0.92 / 2.07 MP (capped by the vision-token budget); EG2 image +375 MB at 1080p; audio and video use conservative payload-size estimates (not
measured). Freed heap is returned to the OS when the worker goes idle. `/health` `memory` shows limit / used / headroom / reserved / rejected. `scripts/test_memory.py`.

**Persistent prefix snapshots (`--snapshot-dir D`, `--snapshot-mb`, default 2048).** New prefix-cache entries are also written to `D` in the background (CRC, atomic rename, oldest-first
disk budget), and a device-cache miss falls back to disk, so a restart no longer throws the system-prompt prefill away: the first request after a restart took 835 ms instead of 5927 ms
(1856 of 1881 tokens restored; a 1881-token snapshot is 43.6 MB with an f32 KV cache), with an answer identical to the uncached one. A snapshot's identity is the model file, KV dtype and kernel build:
files from anything else are never loaded (and never deleted), corrupt files are detected and dropped, over-budget entries are not stored. `scripts/test_snapshots.py`.

**Operations.** `GET /metrics` (Prometheus text: request / token counters, live decode and prefill tokens per second, TTFT histogram, queue / slot gauges, prefix-cache and snapshot stats,
speculative drafted / accepted, memory gauges); `/health` adds `live` rates and the speculative counters. `POST /tokenize` and `/detokenize` (vLLM's fields: `prompt` or `messages`,
`add_special_tokens`, `return_token_strs`; they use the exact tokenization generation uses, so a client can count and plan cached prefixes). Several labelled API keys: `--api-key 'label: key'`
(repeatable), `--api-key-file F` (one `label: key` or bare key per line, `#` comments; **re-read on SIGHUP**), `Authorization: Bearer` or `x-api-key`; the log names the key used; `/metrics` needs a key unless
`--metrics-open`, `/health` stays open. `scripts/test_serve_ops.py`. **Idle cost:** pool workers futex-park after a 200 µs spin (`FERRO_POOL_SPIN_US`), so an
idle server draws 0% of a core; before this a live server kept one core at 100% for its whole life (62% in-kernel), inside the shared CPU+iGPU power budget.

## How the iGPU path is fast
1. **int8 DP4A GEMMs** (`cl_khr_integer_dot_product`): Q8_0 weights resident on the GPU in a
   coalesced `[block][k][rows]` layout; activations quantized per 32-block (llama.cpp's scheme);
   activation tile in shared local memory; one accumulator chain per token; **32-bit typed weight indexing**
   (Xe-LP emulates 64-bit integer multiplies; they were costing a third of GEMM throughput). **Swizzled work-group
   order** for weight matrices larger than the LLC (each weight slice streamed from DRAM ~once).
2. **Whole model on-device**: one upload, one tiny readback per batch (embeddings, or one logit per
   decision option).
3. **Fused producers**: residual + norm + int8 quantization in one pass; attention, SwiGLU / GELU,
   the LFM2 short-conv gate and the decision head's LayerNorm/bias/ReLU emit the next GEMM's int8
   input directly.
4. **Attention**: online softmax; for 64-dim heads (d1) a query-tiled kernel (32 queries per work-group,
   K/V staged once in SLM, per-key rescale -> results independent of batch packing). Long batches are
   split into bounded launches so the i915 watchdog never resets the GPU context.
5. d1: the decision head runs only the asked question type (llama.cpp evaluates all three).

## Build & run
```sh
make -j4            # needs OpenCL headers: CL_INC=../ferro-gpu/third_party/OpenCL-Headers (default)
./ferro-embed embed --model models/embeddinggemma-2-Q8_0.gguf --in sentences.txt --out emb.json
./ferro-embed bench --model models/embeddinggemma-2-Q8_0.gguf --tokens bench/tokens_16.txt --reps 8
./ferro-embed decide --model models/d1-omni-600M-Q8_0.gguf --request bench/d1/card.json
./ferro-embed bench  --model models/d1-omni-600M-Q8_0.gguf --request bench/d1/tickets16.json --reps 4
make test           # kernel tests (incl. iGPU DP4A GEMM) + decision tests (JSON, prompt tokens vs llama.cpp)
python3 scripts/d1_parity.py models/d1-omni-600M-Q8_0.gguf   # vs a running llama-server + batched==alone
```
Flags: `--cpu` (force CPU) · `--gpu` (force/verbose iGPU) · `--hetero` (embeddings: CPU‖iGPU split,
+8–15% throughput, +~1 GB RAM) · `--threads N`.
Container: pass `--device /dev/dri`. Mesa rusticl is enough (no Intel compute-runtime needed).

## CPU path
AVX2 fp32: load-time Q8_0→f32 planes, 4-row × 16-token register-tiled GEMM, 2 spin-pinned threads
(power-capped N100: 2 cores at 2.9 GHz beat 4 at 1.3 GHz), vectorized erf-GELU / SiLU, FTZ/DAZ.
Embeddings: slower than llama.cpp's CPU path (AVX-VNNI int8; a VNNI GEMM is the open CPU item).
d1: faster than llama.cpp CPU (1.5×), but ~7× slower than the iGPU — use the default.

## Layout
- `src/gpu_cl.{h,cpp}` — dlopen'd OpenCL backend: DP4A GEMM + all fused layer kernels (both models)
- `src/model_eg2.cpp` — EmbeddingGemma-2 forward (`layers_gpu` on-device loop; CPU loop; hetero split)
- `src/model_lfm2.cpp` — LFM2 decision model: loader, CPU forward, on-device forward (`forward_gpu`)
- `src/model_lfm2_lm.*` — LFM2/LFM2.5 generative engine (causal CPU reference + iGPU prefill/decode, device KV/conv slots) · `src/chat_template.*` · `src/sampler.*` · `src/lm_cli.*` (`generate`) · `src/server.*` (HTTP)
- `src/decision.cpp` — systemone protocol: request parsing, prompt template + token budgets, answers
- `src/tokenizer_bpe.cpp` — byte-level BPE (LFM2/llama3 pre-tokenizer); `third_party/llama_unicode/`
  (vendored from llama.cpp, MIT) · `src/json.cpp` — order-preserving JSON incl. jinja `tojson`
- `src/kernels*.{h,cpp}` — CPU kernels (AVX2/FMA + scalar refs) · `src/gguf.*` · `src/tokenizer.*`
- `bench/ferro_ref.cpp`, `bench/ferro_ref_vk.cpp` — llama.cpp embedding harnesses (CPU / Vulkan)
- `scripts/final_h2h.sh` (embeddings) · `scripts/d1_bench.py` (d1) — interleaved scoreboards
- `src/media*.{h,cpp}` (stb_image / miniaudio decode) · `src/mm_preproc.*` (tiling, Pillow resize, log-mel; ported from mtmd) · `src/mm_encoder.*` (SigLIP + FastConformer)
- `src/mm_gemma4*.{h,cpp}` · `src/mm_clin.h` · `src/embed_mm.*` — EmbeddingGemma-2 vision (gemma4v) / audio (gemma4a) encoders and the multimodal sequence builder; `bench/mm/{mel_ref,ggml_ops_ref}.cpp` + `pin_*.py` — upstream oracles and per-op layout checks; `scripts/eg2_parity_mm.py`
- `scripts/d1_parity.py` — d1 answers vs llama-server + batched==alone contract · `scripts/d1_parity_mm.py` — media · `bench/d1/` requests
- iGPU R&D, kernel shootouts and findings: `/opt/data/ferro-gpu/`

## Env knobs
`FERRO_GPU=0|1` · `FERRO_HETERO=1` · `FERRO_GPU_KBENCH=1` (per-kernel device times, both models) ·
`FERRO_GPU_SWIZZLE=0|1` · `FERRO_GPU_ATTN_NO5=1` / `FERRO_GPU_UNFUSED=1` / `FERRO_GPU_V2=1` (older kernel
paths, for A/B) · `FERRO_DECISION_BATCH=N` (tokens per d1 forward, default 4096) ·
`FERRO_MM_GPU=0` / `FERRO_MM_DENSE=0` (media encoder fallbacks) · `FERRO_DECISION_DUMP=1` (rendered d1 prompts + token ids) · `FERRO_DEBUG_LAYERS=1` · `FERRO_TIME=1` ·
`FERRO_MM_AUDIO_GPU=0` (audio GEMMs on CPU) · `FERRO_MM_AUDIO_DEV=0` (audio blocks back to host-driven GEMMs) · `FERRO_GPU_DENSE_TEXT=0` / `FERRO_GPU_DENSE_MIN=N` (dense long-sequence text attention) · `FERRO_G4_SYNC=1` / `FERRO_G4_DUMP=dir` / `G4A_TIME=1` / `EMB_TIME=1` (multimodal debugging) ·
`FERRO_LM_*` (see the LM section) · `FERRO_ROW_POLICY=0` (old decode/prefill GEMM mix) · `FERRO_KV=f32|int8` · `FERRO_MEM_LIMIT_MB=N` (override the detected container limit; tests) · `FERRO_LM_WINBENCH=1` (with `FERRO_LM_SELFCHECK=1`: cost of a speculative window by row count) · `FERRO_LOW_MEM=1` · `FERRO_DBGDIR=/path` (CPU stage dumps for golden diffing) · `FERRO_POOL_SPIN_US=N` (CPU pool: how long a worker spins after a task before it parks on a futex, default 200 µs; 0 = park at once) · `FERRO_LM_ATTN_SPLIT=0` (LM decode attention: the per-head kernels instead of split-K)
