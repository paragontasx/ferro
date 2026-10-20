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
(≈3.6× EmbeddingGemma). Image/audio inputs: see the media section below.

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
- Not covered: F16/BF16 weights, video (frames must be extracted by the caller).
- Long states are attention-bound on the iGPU (2×2.25k-token sequences: ~40 s per request).
- CPU path ~70 tok/s: usable, but the iGPU is ~5× faster.


## d1 media encoders (2026-10-21, N100, nothing else running; after the 32-bit GEMM fix)
Embeddings vs llama.cpp libmtmd (`bench/mm/mm_ref` + `cmp.py`); end-to-end answers vs llama-server
`--mmproj` (`scripts/d1_parity_mm.py`): 12/12 token counts, 18/18 top answers, worst |dprob| 0.019.

| Input | Tokens | Encode time (iGPU) | Mean cosine vs libmtmd |
|---|---|---|---|
| 640x480 photo | 234 | 1.2 s | 0.9993 |
| 100x80 image | 72 | 0.28 s | 0.9973 |
| 1920x1080 (8 tiles + thumbnail) | 2,300 | 11.5 s | 0.9987 |
| 11 s audio | 138 | 0.70 s | 0.99996 |
| 30 s audio (cap) | 375 | 2.07 s | 0.99996 |

Per-token worst cosine is 0.97-0.99 for images; llama.cpp's own flash vs non-flash attention differ by the
same amount, so mean cosine is the meaningful gate. llama.cpp's media-encode speed was not benchmarked.
What moved the numbers (photo 7.2 s -> 1.5 s; 1080p 94 s CPU -> 14 s; 30 s audio 3.8 s -> 2.0 s): whole vision
encoder on the iGPU; dense float4-register-tiled attention (no materialized softmax; a dynamically indexed
private accumulator array had been spilling); AVX2 audio conv stack (982 -> 224 ms) and rel-pos attention
(1.2 s -> 0.47 s); a thread-pool int overflow that crashed big images.

## iGPU limits measured (pure-register microkernels, `peak.py`)
DP4A peak 435 GMAC/s, fp32 peak 109 GFMA/s (218 GFLOP/s; not 288). The Q8 GEMM runs at 134 GMAC/s (31% of
DP4A peak) *independent of weight size* (64 KB..9 MB): it is issue-bound, not bandwidth-bound. Ablation: weight
global loads cost ~70% of attainable throughput (ALU-generated weights: 231 GMAC/s), activation staging 4%,
epilogue ~19%. Tried and rejected (all slower or identical): software-pipelined weight prefetch (register
spills, 28-89 GMAC/s), wider/contiguous per-lane weight loads (55-102), uniform global activation loads (47),
subgroup-broadcast activations (77), unroll hints and `prefetch()` (identical), TT 8/12/14 (12+ spills; TT=10
stays). Adopted: single DP4A chain per token (+2.9% end-to-end, 3/3 interleaved rounds).
Open idea (unproven): SLM-staged weights with split loads interleaved between token iterations.

## 2026-10-21 update: 32-bit GEMM indexing (+45-50% on both models)
The Q8 DP4A GEMM computed weight addresses with 64-bit multiplies (`(size_t)b * 8 * rows + ...`), which Xe-LP
emulates. Typed `uint4*` indexing with 32-bit offsets took the micro-benchmark from 134 to 210 GMAC/s (48% of the
435 GMAC/s DP4A peak) with no other change (this was the "weight loads cost 70%" seen in the ablations above:
the loads were fine, the address math was not). Also: `native_exp2` in SwiGLU and flash attention (attention
-11%, parity improved). TT=10 stays optimal (TT>=11 spills); KB=32 would add ~3% but does not divide 4608.

Scoreboard `scripts/d1_bench.py bench/d1/tickets16.json 2` (log: `bench/d1/scoreboard_2026-10-21.log`; best of 2
interleaved rounds, 10 s cooldowns):

| Engine | all 16 at once | one request at a time |
|---|---|---|
| **ferro iGPU** | **5,774 ms · 518 tok/s** | **5,985 ms · 499 tok/s** |
| llama.cpp Vulkan (same iGPU) | 10,977 ms · 272 tok/s | 13,744 ms · 217 tok/s |
| ferro CPU (2 t) | 41,327 ms · 72 tok/s | 42,094 ms · 71 tok/s |
| llama.cpp CPU (2 t) | 62,882 ms · 48 tok/s | 60,084 ms · 50 tok/s |

ferro iGPU vs llama.cpp Vulkan: 1.90x / 2.30x; vs llama.cpp CPU: 10.9x / 10.0x. ferro CPU vs llama.cpp CPU:
1.52x / 1.43x. Parity after the change: 15/15 token counts, 23/23 top answers, worst |dprob| 0.039, batched==alone
0.00. EmbeddingGemma: 16 seqs 219 ms (839 tok/s), 64 seqs 796 ms (938 tok/s), cosine vs reference >= 0.9995.
EmbeddingGemma llama.cpp rows were not re-run (still the 2026-10-20 numbers).
d1 kernel profile now (T=2,989, 5.6 s device): ffn gate|up 2.19 s, ffn down 1.20 s, conv.in 0.46 s, attention
0.47 s, SwiGLU 0.25 s; GEMMs are ~75% of the forward.
