# ferro-embed

Lightweight inference engine for Intel low-power SoCs (N100-class) that runs two models from one
binary, **on the integrated GPU by default** (OpenCL, int8 DP4A) with automatic CPU fallback:

| Model | Arch (GGUF) | Command | What it returns |
|---|---|---|---|
| **EmbeddingGemma-2** | `gemma-embedding2` | `embed` / `bench` | sentence embeddings |
| **LiquidAI d1-omni-600M** | `lfm2` + `lfm2.decision.*` | `decide` / `bench` | typed answers (yes/no, choice, score) over a state, one forward pass, no generated tokens |

The model kind is detected from the GGUF metadata. libOpenCL is `dlopen`'d, so the binary also runs on
GPU-less hosts. Native GGUF loading and native tokenizers (Gemma SPM-BPE; LFM2 byte-level BPE).

## Results (Intel N100, same model files + inputs as llama.cpp, interleaved runs)

**EmbeddingGemma-2** (real embedding path):

| Engine | 16 seqs / 188 tok | 64 seqs / 752 tok |
|---|---|---|
| **ferro-embed (iGPU, default)** | **297 ms · 627 tok/s** | **1089 ms · 687 tok/s** |
| ferro-embed `--hetero` (CPU ‖ iGPU) | 247–284 ms · up to 683 tok/s | ~1000 ms · up to 690 tok/s |
| llama.cpp Vulkan (same iGPU) | 461 ms · 408 tok/s | 1757 ms · 428 tok/s |
| llama.cpp CPU (`llama_encode`, 2 t) | 586 ms · 320 tok/s | 2214 ms · 340 tok/s |
| ferro-embed `--cpu` | ~900–1000 ms · ~175–206 tok/s | ~4000 ms · ~174 tok/s |

**d1-omni-600M Q8_0** (16 support tickets × 3 questions = 48 decision sequences, 2,989 tokens;
llama.cpp = `llama-server /v1/systemone`):

| Engine | all 16 at once | one request at a time |
|---|---|---|
| **ferro-embed (iGPU, default)** | **8.46 s · 353 tok/s** | **8.95 s · 334 tok/s (0.56 s/request)** |
| llama.cpp Vulkan (same iGPU, `-np 8`) | 11.08 s · 270 tok/s | 13.74 s · 218 tok/s |
| ferro-embed `--cpu` | 40.7 s · 73 tok/s | 42.4 s · 71 tok/s |
| llama.cpp CPU (2 t, `-np 8`) | 61.2 s · 49 tok/s | 61.4 s · 49 tok/s |

d1 iGPU: **1.31–1.54× llama.cpp's own iGPU backend, ~7× llama.cpp CPU**; ferro's CPU path is also
1.45–1.5× llama.cpp CPU here. Answers match llama.cpp: identical prompt tokens on 15/15 requests, same top
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

**Scope:** text states. Image/audio inputs (the model's `mmproj` SigLIP2 / FastConformer encoders) are
not implemented; such requests are rejected with a clear error. Tested with the Q8_0 GGUF; the loader
also accepts F16/BF16/F32 tensors (quantized to Q8 at load for the int8 path), but that is untested.

## How the iGPU path is fast
1. **int8 DP4A GEMMs** (`cl_khr_integer_dot_product`): Q8_0 weights resident on the GPU in a
   coalesced `[block][k][rows]` layout; activations quantized per 32-block (llama.cpp's scheme);
   activation tile in shared local memory; 2 independent accumulator chains. **Swizzled work-group
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
d1: faster than llama.cpp CPU (1.5×), but ~5× slower than the iGPU — use the default.

## Layout
- `src/gpu_cl.{h,cpp}` — dlopen'd OpenCL backend: DP4A GEMM + all fused layer kernels (both models)
- `src/model_eg2.cpp` — EmbeddingGemma-2 forward (`layers_gpu` on-device loop; CPU loop; hetero split)
- `src/model_lfm2.cpp` — LFM2 decision model: loader, CPU forward, on-device forward (`forward_gpu`)
- `src/decision.cpp` — systemone protocol: request parsing, prompt template + token budgets, answers
- `src/tokenizer_bpe.cpp` — byte-level BPE (LFM2/llama3 pre-tokenizer); `third_party/llama_unicode/`
  (vendored from llama.cpp, MIT) · `src/json.cpp` — order-preserving JSON incl. jinja `tojson`
- `src/kernels*.{h,cpp}` — CPU kernels (AVX2/FMA + scalar refs) · `src/gguf.*` · `src/tokenizer.*`
- `bench/ferro_ref.cpp`, `bench/ferro_ref_vk.cpp` — llama.cpp embedding harnesses (CPU / Vulkan)
- `scripts/final_h2h.sh` (embeddings) · `scripts/d1_bench.py` (d1) — interleaved scoreboards
- `scripts/d1_parity.py` — d1 answers vs llama-server + batched==alone contract · `bench/d1/` requests
- iGPU R&D, kernel shootouts and findings: `/opt/data/ferro-gpu/`

## Env knobs
`FERRO_GPU=0|1` · `FERRO_HETERO=1` · `FERRO_GPU_KBENCH=1` (per-kernel device times, both models) ·
`FERRO_GPU_SWIZZLE=0|1` · `FERRO_GPU_ATTN_NO5=1` / `FERRO_GPU_UNFUSED=1` / `FERRO_GPU_V2=1` (older kernel
paths, for A/B) · `FERRO_DECISION_BATCH=N` (tokens per d1 forward, default 4096) ·
`FERRO_DECISION_DUMP=1` (rendered d1 prompts + token ids) · `FERRO_DEBUG_LAYERS=1` · `FERRO_TIME=1` ·
`FERRO_LOW_MEM=1` · `FERRO_DBGDIR=/path` (CPU stage dumps for golden diffing)
