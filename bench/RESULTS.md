# Benchmark results — ferro-embed on Intel N100 (2026-10-20, iGPU edition)

Hardware: Intel N100 (4× Gracemont, AVX2 + AVX-VNNI, no AVX-512) + Alder Lake-N UHD iGPU
(24 EU Xe-LP @ 750 MHz) passed through as `/dev/dri/renderD128`; Mesa 25.0.7 (rusticl OpenCL,
ANV Vulkan). Model: EmbeddingGemma-2 Q8_0 GGUF. All runs pre-tokenized, full pooled embeddings,
interleaved with 8 s cooldowns (`scripts/final_h2h.sh`).

## Final scoreboard — identical workload through each engine's real embedding path

**16 sequences / 188 tokens**

| Engine | Backend | ms / batch | tok/s | vs best llama.cpp |
|---|---|---|---|---|
| **ferro-embed `--hetero`** | CPU ‖ iGPU split | **247 best / 275–284 avg** | **662–683** | **1.62×** |
| **ferro-embed (default)** | **iGPU, int8 DP4A** | **297.5** | **625–627** | **1.55×** |
| llama.cpp | Vulkan, same iGPU (`-ngl 99`) | 460–463 | 406–409 | 1.00× |
| llama.cpp | CPU `llama_encode`, 2 threads (best) | 584–589 | 319–322 | 0.79× |
| ferro-embed `--cpu` | CPU AVX2 fp32, 2 threads | 898–1004 | 173–206 | 0.45× |

**64 sequences / 752 tokens**

| Engine | Backend | ms / batch | tok/s |
|---|---|---|---|
| **ferro-embed (default)** | **iGPU** | **1089** | **687** |
| ferro-embed `--hetero` | CPU ‖ iGPU | 984–1000 best (avg 1090–1223) | 615–690 |
| llama.cpp | Vulkan iGPU | 1757 | 428 |
| llama.cpp | CPU 2 t | 2214 | 340 |
| ferro-embed `--cpu` | CPU | ~4000–4160 | ~174 |

ferro's iGPU path is **1.55–1.61× llama.cpp's own iGPU (Vulkan) backend** and **~2× llama.cpp CPU**
on the same chip, same model file, same tokens.

## ⚠ Correction to the earlier CPU-only claim
The previous RESULTS.md claimed ferro's CPU path was "≈1.5× llama.cpp". That baseline was
`llama-bench -p 188` (~130 tok/s), which is **not representative** of llama.cpp's embedding
throughput. llama.cpp's actual embedding path (`llama_encode`, batched sequences, 2 threads,
`bench/ferro_ref_vk.cpp`) runs **~320–340 tok/s** — about **1.9× faster than ferro's CPU path**.
Likely cause: ggml uses AVX-VNNI int8 dot products for Q8_0×Q8_0; ferro's CPU GEMM is fp32 FMA
and does not yet use the VNNI the chip has (`avx_vnni` in /proc/cpuinfo). Use the iGPU path
(default) or `--hetero`; a VNNI CPU GEMM is the open item for CPU-only hosts.

## Accuracy (cosine vs llama.cpp-CPU embeddings, same token ids)

| Path | cos min | cos mean |
|---|---|---|
| ferro CPU (fp32) | 0.999591 | 0.999885 |
| **ferro iGPU (int8 activations, = llama.cpp's Q8_0 activation scheme)** | **0.999516** | 0.999864 |
| ferro `--hetero` | 0.999516 | 0.999867 |
| llama.cpp Vulkan | 0.999249 | — |

Deterministic across runs. `make test` includes the iGPU DP4A GEMM vs CPU GEMM (rel L2 5.3e-3).

## iGPU path: what produced the speed
See `/opt/data/ferro-gpu/docs/FINDINGS.md` for the full journey. Summary (ms / forward, 188 tok):
CPU 898 → per-GEMM offload 578 → on-device layer loop 457 → fused int8 producers 437 →
coalesced attention 372 → PLM GEMM on GPU 339 → subgroup/online-softmax v3 309 →
DP4A TT=10 + split accumulators **297**.
GEMMs (int8 DP4A, ~220–240 GF-equiv) are ~75% of the remaining time; the rest is attention
(~35 ms), norms+quant (~23 ms), gelu (~15 ms).

## LiteRT-LM
`litert-lm` v0.18 wheel aborts at engine init in this container (SIGABRT / tcsetattr, no metrics)
for `kontextdev/embeddinggemma-300m-litertlm`; not benchmarkable here.

## Reproduce
```sh
make -j4 && make test
./scripts/final_h2h.sh                     # all engines/backends, interleaved
./ferro-embed bench --model models/embeddinggemma-2-Q8_0.gguf --tokens bench/tokens_16.txt --reps 8
FERRO_GPU_KBENCH=1 ./ferro-embed bench ...  # per-kernel iGPU device times
```
llama.cpp Vulkan baseline: built in `baselines/llama.cpp/build-vk` (Vulkan-Headers, SPIRV-Headers
and glslc from `/opt/data/ferro-gpu/third_party`); harness `bench/ferro_ref_vk.cpp`
(`REF_NGL=99` GPU / `0` CPU, `REF_THREADS`).

---

# LiquidAI d1-omni-600M (decision model) — 2026-10-20

Model: `LiquidAI/d1-omni-600M-GGUF` Q8_0 (sha256 cd94463f…), arch `lfm2` + 2-layer decision head
(16 LFM2 trunk layers: 10 short-conv + 6 GQA attention, d=1024, ffn 4608). ~0.9 GFLOP per token
(≈3.6× EmbeddingGemma). Text states only (no mmproj image/audio encoders).

## Scoreboard — `scripts/d1_bench.py bench/d1/tickets16.json 2`
16 support tickets × 3 questions (noul + 5-way choice + 4-level score) = 48 decision sequences,
2,989 tokens. llama.cpp = `llama-server -b 4096 -ub 4096 -t 2` on `/v1/systemone` (same GGUF), CPU build
and Vulkan build (`-ngl 99`, `GGML_VK_VISIBLE_DEVICES=0`). Interleaved, 10 s cooldowns, best of 2 rounds
(rounds agree within ~2%).

| Engine | Throughput (all 16 at once) | Latency (one request at a time) |
|---|---|---|
| **ferro-embed iGPU (default)** | **8,460 ms · 353 tok/s** | **8,949 ms · 334 tok/s** |
| llama.cpp Vulkan, same iGPU (`-np 8` / `-np 1`) | 11,076 ms · 270 tok/s | 13,742 ms · 218 tok/s |
| ferro-embed `--cpu` (2 t) | 40,708 ms · 73 tok/s | 42,380 ms · 71 tok/s |
| llama.cpp CPU (2 t, `-np 8` / `-np 1`) | 61,190 ms · 49 tok/s | 61,374 ms · 49 tok/s |

ferro iGPU = **1.31× (throughput) / 1.54× (latency) llama.cpp-Vulkan** and **7.2× / 6.9× llama.cpp CPU**.
ferro CPU = 1.45–1.5× llama.cpp CPU (ferro runs only the asked question type through the decision head;
llama.cpp evaluates all three).

## Correctness — `scripts/d1_parity.py` (15 adversarial requests vs llama-server CPU)
- Prompt tokens identical 15/15 (template, budgets, escaping, `<think>`-style control tokens, unicode,
  JSON states with Python-repr floats, null state, 12-option choice, 10-level score, 2.2k-token state).
- Top answer agrees **23/23**; worst |Δprob| 0.056 (int8 activation rounding differs between engines).
- **batched == alone contract: bit-identical** (every request answered alone equals its answer inside the
  15-request batch).
- `make test`: decision tests (JSON/jinja tojson, prompt token golden = llama.cpp ids, calibration).

## iGPU kernel profile (T=2,989, `FERRO_GPU_KBENCH=1`), 8.5 s/forward device time
gate|up GEMM 3.6 s · down 1.9 s · conv.in 0.8 s · attention (8 layers) 0.45 s · silu 0.4 s · rest ~1.3 s.
Optimizations that mattered: swizzled GEMM work-group order (+30%: the 9.4 MB gate|up matrix was
re-streamed from DRAM per token tile), query-tiled attention for 64-dim heads (2.7×), head only for the
asked type.

## Bugs found on the way (all fixed, see skill n100-cpu-inference-engine)
- `gemm_q8_fma` with resid == Y doubled the GEMM output (in-place residual) — now snapshotted.
- reused attention kernel assumed ≥32 lanes (head_dim 256); 64-dim heads left half the scores unset.
- i915 watchdog silently reset the GPU context on multi-second attention submissions (2.2k-token states):
  answers collapsed to p=0.5 with no error. Fixed by an efficient kernel + bounded launches.
- chunk-wise online softmax made answers depend on batch packing (up to 4% on near-ties) — per-key rescale.

## Not done / limits
- Image and audio inputs (SigLIP2 vision tower, FastConformer audio encoder in the mmproj GGUF).
- Long states are attention-bound on the iGPU (2×2.25k-token sequences: ~40 s per request).
- CPU path ~70 tok/s: usable, but the iGPU is ~5× faster.
