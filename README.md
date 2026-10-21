# Ferro

Lightweight inference engine for Intel low-power SoCs (N100-class) that runs two models from one
binary, **on the integrated GPU by default** (OpenCL, int8 DP4A) with automatic CPU fallback:

| Model | Arch (GGUF) | Command | What it returns |
|---|---|---|---|
| **EmbeddingGemma-2** | `gemma-embedding2` | `embed` / `bench` | sentence embeddings |
| **LiquidAI d1-omni-600M** | `lfm2` + `lfm2.decision.*` | `decide` / `bench` | typed answers (yes/no, choice, score) over a state, one forward pass, no generated tokens |
| **LiquidAI LFM2 / LFM2.5** (LFM2.5-1.2B-Thinking, LFM2.5-350M, ...) | `lfm2` (causal) | `generate` / `serve --lm` | text generation: chat + completions, streaming, OpenAI-compatible |

The model kind is detected from the GGUF metadata. libOpenCL is `dlopen`'d, so the binary also runs on
GPU-less hosts. Native GGUF loading and native tokenizers (Gemma SPM-BPE; LFM2 byte-level BPE).

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
(split-K GEMV for the 1-4 token GEMMs, one-query-per-row attention over the cache). Measured on the N100 (Q8_0, same file as llama.cpp):

| LFM2.5-1.2B-Thinking Q8_0 | ferro iGPU | llama.cpp CPU (2 threads) |
|---|---|---|
| prefill, 762 tokens | **153 tok/s** | 11.2 tok/s |
| decode, 1 stream | **15-19 tok/s** | 6.1 tok/s |
| decode, aggregate over 2 / 4 streams (server) | **22 / 35 tok/s** | |

**LFM2.5-350M** (`LiquidAI/LFM2.5-350M-GGUF`, Q8_0, same code path, no model-specific changes) measured the same way:

| LFM2.5-350M Q8_0 | ferro iGPU | llama.cpp CPU (2 threads) |
|---|---|---|
| prefill, 192 / 762 / 1522 tokens | 192 / **490** / 414 tok/s | 66.7 tok/s (762 tokens) |
| decode, 1 stream | **36-37 tok/s** | 31.8 tok/s |
| decode, aggregate over 2 / 4 / 8 streams (server) | **58 / 98 / 132 tok/s** | |

Parity vs llama.cpp Q8_0 on the same weights: top-1 12/12, mean TV 0.017 (every prompt < 0.06). This model is far more sensitive to Q8 than the 1.2B (llama.cpp's own Q8 vs F16 is TV 0.040; ferro vs F16 0.043; both flip the same near-tie on an out-of-distribution `<think>` prompt), so compare against the same-weights Q8 oracle for it. Decode here is latency-bound (0.38 GB of weights = ~15 ms at 25.9 GB/s, the step takes ~27 ms: ~190 small kernel launches/token), so it is only ~1.1x llama.cpp CPU single-stream; prefill and batched serving are where the iGPU wins.
The chat template is read from each GGUF: the 350M template walks content arrays itself (text parts concatenated, system arrays too) while the Thinking template does not (llama.cpp's server joins parts with `\n` there); `serve` logs the detected variant and `/apply-template` is byte-identical to llama.cpp's jinja for both.

Decode streams 1.17 GB of weights per token at ~22 GB/s; a pure-streaming read kernel reaches 25.9 GB/s on this iGPU, so Q8 decode is ~85% of the
hardware limit (only a smaller weight format can go meaningfully faster). **Accuracy** (`scripts/lm_parity.py`, 12 prompts incl. chat-template,
code and a 900-token one; next-token distribution vs llama.cpp **F16** as the oracle): top-1 agreement 12/12, mean total-variation distance **0.0198** -
the same as llama.cpp's own Q8_0 vs its F16 (0.0200), i.e. at the quantisation noise floor. At the hidden-state level (3 / 8 layers, per-token relative L2
vs F16) ferro's fp32-activation CPU reference is *closer* to F16 than llama.cpp's int8-activation Q8 path (1.1 / 1.4% vs 1.9 / 2.6%); the iGPU path quantises
activations to int8 like llama.cpp and lands on the same TV. Debug knobs: `FERRO_LM_GPU=0` (CPU reference, ~0.5 tok/s decode), `FERRO_LM_PROF=1` (per-op device times),
`FERRO_LM_STEPTIME=1`, `FERRO_LM_SELFCHECK=1` (chunked prefill / token-by-token decode vs one prefill), `FERRO_LM_KEEP_HOST=1`, `FERRO_GPU_SMALL=0` / `FERRO_GPU_GEMV=0`.

## Serving over HTTP (`ferro-embed serve`)
One resident process, OpenAI-compatible, embeddings and/or language model on the same iGPU (one worker thread owns the device):
```sh
./ferro-embed serve --model models/embeddinggemma-2-Q8_0.gguf --mmproj models/mmproj-embeddinggemma-2-Q8_0.gguf \
                    --lm models/LFM2.5-1.2B-Thinking-Q8_0.gguf --port 8088 [--lm-slots 4] [--lm-ctx 4096] [--api-key K] [--host 0.0.0.0]
curl localhost:8088/v1/chat/completions -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"stream":true}'
curl localhost:8088/v1/embeddings -d '{"input":"hello"}'
```
| Endpoint | |
|---|---|
| `POST /v1/embeddings` | text / `image_url` / `input_audio` / `video_url` parts (data URLs), `dimensions`, `encoding_format: base64`; concurrent text requests share packed forward passes |
| `POST /v1/chat/completions`, `POST /v1/completions` | `messages` (+`tools` are rendered into the prompt), `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `stop`, `stream` (SSE) + `stream_options.include_usage`; `usage` and llama.cpp-style `timings` in every response |
| `POST /apply-template` | rendered chat prompt (same contract as llama.cpp) |
| `GET /v1/models`, `GET /health` | capabilities / context / slots; queue, active slots, throughput counters |

`<think>..</think>` output is returned as `reasoning_content` (`--reasoning-format none` keeps it inline). Generation is continuously batched over `--lm-slots`
device slots (each holds `--lm-ctx` tokens of KV cache, ~25 MB per 1k tokens); embedding requests are served between decode steps (a short text embedding
measured 0.12-1.8 s while 4 streams were generating - it waits for the decode step or prefill in progress); a closed client connection frees its slot within a step; 503 when the queue is full; Bearer `--api-key`
(or `FERRO_API_KEY`). Not implemented: tool-call *parsing* of model output (tools are only rendered into the prompt), `n > 1`, logprobs, image input to the LM, prefix caching.
**Memory** (iGPU buffers are system RAM and not swappable - size `--lm-slots` x `--lm-ctx` to your limit; measured via cgroup, device buffers show as shmem):

| Process | anon | device (shmem) |
|---|---|---|
| `--lm` only, 4 x 4096 | ~0.6 GB | 1.8 GB |
| `--lm` only, 1 slot | ~0.8 GB | 1.5 GB |
| `--model` + `--mmproj` | ~1.6 GB | 0.8 GB |
| EG2 text only | ~0.9 GB | 0.2 GB |

The server prints the KV size and warns when it approaches a container limit. Tests: `scripts/test_serve.py` (embeddings), `scripts/test_serve_lm.py`
(chat: usage accounting, seeds, stream == non-stream, stop strings, UTF-8, errors, cancellation, concurrency, template parity with `--llama URL`),
`scripts/bench_lm_serve.py` (concurrency scaling), `scripts/bench_serve.py` (embeddings).

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
`FERRO_LM_*` (see the LM section) · `FERRO_LOW_MEM=1` · `FERRO_DBGDIR=/path` (CPU stage dumps for golden diffing)
