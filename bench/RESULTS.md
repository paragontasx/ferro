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


## 2026-10-21: EmbeddingGemma-2 multimodal on the iGPU (vision, audio, video)

`ferro-embed embed-media` (text / image / audio / video items interleaved into one 768-d embedding). Oracle: llama-server
`/v1/embeddings` with `--mmproj` (`scripts/eg2_parity_mm.py`, cached references, built-in oracle-sanity guard). Gate: cosine of the
final L2-normalised embedding >= 0.999. Encoder ground truth: `bench/mm/mm_ref` (libmtmd), audio mel: `bench/mm/mel_ref` (upstream
`mtmd_audio_preprocessor_gemma4a`, compiled unmodified; our mel is bit-exact).

### Final-embedding parity (12 cases, all pass, final build)
| case | tokens | cosine | ferro time |
|---|---|---|---|
| text | 12 | 0.999952 | 0.07 s |
| image 100x80 | 84 | 0.999362 | 1.5 s |
| image 640x480 | 134 | 0.999652 | 2.8 s |
| image 1920x1080 | 924 | 0.999848 | 70 s |
| text + image | 139 | 0.999300 | 2.9 s |
| audio 11 s (wav / mp3 / flac 44.1 kHz) | 279 | 0.999924 / 0.999915 / 0.999947 | 1.4 s |
| text + audio | 284 | 0.999943 | 1.4 s |
| audio 44 s (two 30 s chunks, one span) | 1104 | 0.999920 | 6.5 s |
| video 2 s (8 frames @ 4 fps) | 669 | 0.999646 | 12.3 s |
| text + image + audio | 412 | 0.999831 | 4.3 s |
Token counts equal the server's in every case (1104 for 44 s: ONE `<|audio>..<audio|>` span for all 30 s chunks, as mtmd).

### What moved the numbers
- **Vision tower on the iGPU** (was CPU-only: 51% attention_fma + 38% GEMM): 640x480 photo 16.4 s -> 2.5 s. Fused `[q|k|v]` and
  `[gate|up]` weights, `row_fused` for the sandwich norms, one new kernel (`qkv_post_g4`: 2-D RoPE + q/k/v norms), dense attention.
  Per-chunk mean cosine vs libmtmd .9962 (CPU path .9972): int8 activations compound over 16 layers (relL2 vs CPU 0.24% at layer 0
  -> 6.4% at layer 15, smooth, no step) - the pooled embedding averages it out (.9997).
- **Audio** (new): mel frontend bit-exact; encoder = SSCP (ggml_conv_2d rounds BOTH conv operands to fp16 - reproduced), 12 conformer
  blocks with chunked local attention (C=12, P=12), blocked relative shift, softcap 50, clamped linears, causal depthwise conv.
  `per_dim_scale` is applied raw (softplus: cos .90). 11 s clip: CPU 4.5 s -> iGPU GEMMs 1.4 s. Per-chunk mean cosine .99987 (CPU) /
  .99976 (iGPU). Remaining audio cost (11 s, warm): FFN 735 ms, attention 253, conv module 173, sscp 173 (attention/conv on the CPU).
- **Text model attention for long single sequences** (the lane-per-key kernels are O(T) serial per query; at 660 tokens attention was
  90% of the forward): head-dim generic dense kernels (`dattn_qk_h` / `dattn_stats_w` / `dattn_pv_h`, band mask for the SWA layers,
  swa_window is stored as window/2+1 = 513) used when ONE segment covers all tokens. iGPU forward, old -> new: 126 tok 363 -> 217 ms,
  423 tok 2830 -> 804, 542 tok 4468 -> 1131, 672 tok 6729 -> 1545, 951 tok 13239 -> 2608 (5.1x). Sequences above the old 1024-token
  limit stay on the iGPU (1150 tok 3.7 s, 1587 tok 6.4 s, 2104 tok 10.4 s; cos vs llama.cpp >= .9998). Headline 16x188 bench unchanged.
- **Query-tiled dense attention**: a 1080p photo is 8280 patches (3.3 GB of scores); queries are tiled into a bounded 64 MB scratch.
  1080p: 67 s encode vs llama.cpp CPU 183 s; attention is 88% of it (qk 28.9 s, pv 28.1 s, stats 4.7 s, ~29 GMAC/s each = 27% of the 109 GFMA/s fp32 peak; an earlier version of this line said 55% because it double-counted MACs as FLOPs).

### Limits / not done
- 1080p photos are slow (~70 s): ViT attention runs at ~27% of fp32 peak (29 GMAC/s), so kernel tuning has real headroom.
- Video costs ~1.5 s/frame: small frames are upscaled to the 70-token minimum image (a 320x240 frame becomes 480x384 = 720 patches).
  Batching frames into one forward (segment-masked attention) would amortise the weight traffic; not done.
- The audio SSCP conv front-end (147 ms per 11 s: im2col 39, conv gemm 33, transpose 24, layernorm 38, input proj 9) and the tail still run on the CPU; most of it is memory-layout overhead, not math.
- Text/audio sequences beyond ~2900 tokens fall back to the CPU layer loop (score scratch cap 128 MB).
- Llama.cpp's Vulkan path was not benchmarked for the EmbeddingGemma-2 encoders.


## 2026-10-21 (later): audio conformer on the iGPU, attention tuning study, video batching ruled out

### Audio conformer blocks fully on the iGPU
Five kernels (`a_norm_quant`, `a_act_quant`, `a_resid`, `a_conv_mid`, `a_attn`) mirror the validated CPU reference; the residual stream
stays on the device (per call: X up, X down). q/k/v share their clamps in all 12 blocks, so one fused qkv GEMM + one clamp inside the
attention kernel serves them; Pk = k_rel(pos_emb) is precomputed per layer. 11 s clip, audio encode 1456 -> 796 ms; the device part is
615 ms, of which the GEMMs are ~470 ms (79 GMAC at ~170 GMAC/s) and all new kernels together ~150 ms (attention 67, act_quant 52, resid 36,
norm_quant 29, conv_mid 26). Per-chunk mean cosine vs libmtmd .99985 (host-driven version .99976). Device vs CPU path at 4..205
tokens (incl. lengths that are not a multiple of the 12-token block): mean cos >= .9998.
`FERRO_MM_AUDIO_DEV=0` returns to the host-driven GEMMs; any device failure falls back to them automatically.

### SWA band skipping (text model)
`dattn_qk_h` returns early for 64x64 tiles entirely outside the sliding window, `dattn_pv_h` only walks the keys in the band.
1154 tokens 3.7 -> 3.46 s, 1591 tokens 6.4 -> 5.1 s; cos vs the CPU path >= .9999 (gain grows with length).

### Video: frame batching does not help (measured, not built)
GEMM cost per token is 0.82 ms (T = 720, one frame) vs 0.80 ms (T = 8280, the 1080p photo): batching the 8 frames into one forward pass
saves ~3% of the GEMM time. Attention stays per frame either way. The 1.3 s per frame is 0.6 s GEMM (int8 DP4A, ~190 GMAC/s) + 0.57 s
attention + 0.13 s elementwise; none of it is launch or weight-streaming overhead.

### Attention kernel study (pyopencl ablation, T = 1104, nq = 4, hd = 256, QK: 44 ms = 28 GMAC/s = 26% of the 109 GFMA/s peak)
Decomposition by ablation: FMA loop ~14 ms (82% of peak - the inner loop is fine), S store 6, barriers 7, global-load latency 15.
Variants, all bit-identical or within 2e-5 of the baseline: register prefetch of the next chunk 49.7 ms (more private memory: spills);
128x64 tile (8 rows per lane) 49.6; coalesced loads + vstore4 + k-vector inner loop 46.5; smaller work-groups (16x8: 65 ms, 16x4: 130 ms);
SLM chunk 32: 43.0; forcing SIMD16 via `intel_reqd_sub_group_size`: same; SIMD8: 50.4; no SLM at all with Q/K stored transposed and direct
global loads: 117 ms (10 GMAC/s). Occupancy probes (SLM 16/32/48 KB per work-group) change nothing. Conclusion: the 64x64 SLM-staged kernel is
at the practical plateau for this GPU/driver; a flash-style fusion (no S round trip) is the only remaining structural lever and was not attempted.
A single work-group needs 310 us, at saturation ~30 us per group: about 10 groups are in flight.

## 2026-10-22: LiquidAI LFM2.5-1.2B-Thinking (generative LFM2) on the iGPU + HTTP serving

Model: `LFM2.5-1.2B-Thinking-Q8_0.gguf` (arch `lfm2`, causal; 16 blocks = 10 short-conv + 6 GQA attention (32 q / 8 kv heads, hd 64), d 2048, ffn 8192, vocab 65536, tied head).
llama.cpp reference: same Q8_0 file (CPU, 2 threads) and the F16 GGUF as the high-precision oracle. All numbers N100, nothing else running.

### Speed
| | ferro iGPU | llama.cpp CPU |
|---|---|---|
| prefill 6 / 192 / 762 / 1522 tokens | 11 (warm-up) / 158 / 153 / 134 tok/s | 11.2 tok/s (767 tokens) |
| decode, 1 stream | 15-19 tok/s (`generate` 15-17; HTTP stream 19.1 in the last run, 13.5-15 in earlier ones) | 6.1 tok/s |
| decode aggregate, server, 1 / 2 / 4 concurrent streams (final build) | 18.3 / 22.2 / 34.5 tok/s (per stream 19.1 / 11.7 / 9.7) | |

Decode progression: 9.2-9.6 tok/s with the batch-tiled GEMM (a 10-token tile at 1 token wastes ~90% of the DP4A work; per-op synced profile: GEMM = 70% of the step) ->
**14.3-15** with 1/2/4-token tile variants of the same GEMM kernel (`FERRO_GPU_SMALL=0` to disable). A split-K GEMV
(`q8gemv`: 16 row-groups x 8 k-splits per work-group, SLM reduce; more threads for the 2048-row matrices) gave **no gain** (14.86 -> 14.99 tok/s, logits cos 1.0): by then
the step is bandwidth-bound. Enqueue time is 4 ms of a 53 ms step, so it is not launch-bound either. A pure streaming-read OpenCL kernel on this iGPU tops out at
**25.9 GB/s**; decode moves 1.17 GB/token (1.03 GB layers + 0.13 GB tied head) in 53 ms = 22 GB/s = 85% of that. Only a smaller weight format (Q4) can go meaningfully faster.
Remaining host cost per token ~10 ms (top-k selection over 65536 logits, 256 KB readback, embedding gather).
Startup: 20-37 s -> 16 s for LM + EG2 (parallel 64-row-tile weight repack: LM load 10.4 -> 5.8 s; the 1/2/4-token GEMM variants are built at load, not on first use - the
first request used to pay a ~1 s JIT stall).

### Accuracy (what looked like a bug and was not)
`scripts/lm_parity.py`, 12 prompts, next-token distribution, reference = llama.cpp **F16**:
| candidate | top-1 agree | mean TV (top-20, renormalised) |
|---|---|---|
| llama.cpp Q8_0 (noise floor) | 12/12 | 0.0200 (max 0.065) |
| **ferro iGPU (int8 activations)** | **12/12** | **0.0198** |
Hidden-state check with truncated GGUFs (first N blocks, `gguf_trunc.py`), per-token relative L2 of the final normed hidden state, mean over 6 tokens:
| blocks | llama.cpp Q8 vs F16 | ferro CPU (fp32 act.) vs F16 | ferro CPU vs llama.cpp Q8 |
|---|---|---|---|
| 3 | 1.88% | **1.06%** | 1.61% |
| 8 | 2.62% | **1.36%** | 2.11% |
So ferro is closer to the F16 oracle than llama.cpp's own int8-activation path; the "2-4% deviation from llama.cpp" seen first was llama.cpp's noise. Things that were ruled out by
experiment (each variant scored 3-30x worse on the same metric, so the shipped wiring is the right one): reversed conv taps (113% error), swapping only the two history taps (70%),
RoPE before the q/k norm (30%), interleaved RoPE pairs (11%); F32 KV + no flash-attention in llama.cpp changes nothing (attention numerics are not the source); int8 activations double ferro's
own error (2.4% -> 5.2% vs F16) but the pipeline is still at llama.cpp's level.
Lessons: (1) llama-server adds BOS itself - a `<|startoftext|>` in the prompt gives it a doubled BOS and a different input (that alone produced a spurious 6x worse TV);
(2) always calibrate against the reference's own quantisation noise before calling a deviation a bug; (3) bisect by layer count with truncated GGUFs, not by guessing.
Chat template: byte-identical to llama.cpp's jinja (`/apply-template`) for system / multi-turn / `<think>` stripping / tools; one real difference found and fixed - multi-part text content is joined with "\n".

### Server (`scripts/test_serve_lm.py`, `scripts/test_serve.py`: all pass)
Continuous batching over device slots: each slot = per-attention-layer KV rows + 2-token conv state; prefill is chunked (`FERRO_LM_TMAX`, default 1024 rows) and uses the causal tiled kernel,
decode runs all active sequences in one pass. Prefill of a new request is serialised in front of decode steps (TTFT 0.19 s alone, 0.40 s with 2 and 0.89 s with 4 simultaneous new requests, final build; it was 1.2-1.6 s before the decode GEMM variants were pre-built at load).
Embedding jobs run between decode steps: 0.12 s when idle-adjacent, up to ~1.8 s when they land behind a prefill.

### Memory (this container is capped at 4 GiB; cgroup `memory.max`)
iGPU buffers are system RAM and count as shmem. Measured: `--lm` 4 slots x 4096: anon 0.6 GB + device 1.8 GB; 1 slot: 1.5 GB device; `--model --mmproj`: anon 1.6 GB + device 0.8 GB;
EG2 text only: anon 0.9 GB + device 0.2 GB. The first combined server (4 x 8192 KV, host weight copies kept) was **OOM-killed 3 times** (cgroup `oom_kill` counter) by the test
suite spawning a second model process. Fixes: free the host copy of the device-resident LM weights after upload (-1.2 GB, no CPU fallback for that instance, `FERRO_LM_KEEP_HOST=1` keeps it),
default `--lm-ctx 4096`, the server prints the KV size and warns near the container limit; CLI-spawning tests use `--no-cli` against a loaded server.

### Regression (after all of the above)
`make test` (kernels incl. iGPU GEMM, decision, media) ALL PASS; EG2 multimodal parity 12/12, worst cosine 0.99930 (gate 0.999); d1 media parity 12/12 token counts equal, 18/18 top answers;
EG2 headline 218.6 ms (A/B with the small-tile GEMM off: 218.5 ms); d1 `bench` 510-524 tok/s with and without it.

### Not done / limits
Q4 weight path (decode is bandwidth-bound: ~2x); prefix caching for multi-turn chat (conv state is only valid at the end of a sequence - needs a snapshot at the end of the prompt);
tool-call parsing of model output; `n > 1`, logprobs; sampling is on the host (~10 ms/token); 1522-token prefill drops to 134 tok/s (attention); the CPU reference decodes at ~0.5 tok/s.

### LFM2.5-350M (same engine, no model-specific code) - 2026-10-22
`LFM2.5-350M-Q8_0.gguf` (d 1024, 16 heads x 64, 8 kv, ffn 6656, 10 conv + 6 attention), llama.cpp reference = same file on CPU (2 threads) and the F16 GGUF.
| | ferro iGPU | llama.cpp CPU |
|---|---|---|
| prefill 192 / 762 / 1522 tokens | 192 / 490 / 414 tok/s | 66.7 tok/s (762) |
| decode, 1 stream (CLI 36.6-37.0, HTTP 35.7) | 36-37 tok/s | 31.8 tok/s |
| aggregate decode, 1 / 2 / 4 / 8 streams (8 slots, `bench_lm_serve.py`) | 35.8 / 58.3 / 97.8 / 132.1 tok/s (TTFT med 0.09 / 0.15 / 0.22 / 0.35 s) | |
Parity (`lm_parity.py`, 12 prompts): vs llama.cpp **Q8_0** (same weights) top-1 12/12, mean TV 0.0172, max 0.0445; vs F16: llama.cpp Q8 0.0398 (11/12 top-1), ferro 0.0426 (11/12) - the 350M is ~2x more
quantisation-sensitive than the 1.2B (0.020) and prompt #3 (an out-of-distribution `<think>` ending for a non-thinking model) flips the same near-tie (token 9266 vs 730) in llama.cpp's Q8 and in ferro.
So for small models validate against the same-weights quantised oracle AND report the F16 noise floor.
Decode profile (`FERRO_LM_STEPTIME`): enqueue 4-6 ms + device 21-23 ms per token; ideal at the measured 25.9 GB/s streaming ceiling is ~15 ms for 0.38 GB, so ~7-10 ms is per-kernel launch/dependency latency over ~190 launches
(sync profile: GEMM 48%, resid_norm 15%, silu_mul 8%, conv+conv_state 10%). Fusing conv+state, qkv_post+kv_write and the residual/norm ops is the available win (~1.3x on this model); the 1.2B is bandwidth-bound and gains more from a Q4 weight path.
**Chat template differs per model**: the 350M GGUF carries a template that handles typed content arrays itself (no "\n" join, system arrays concatenated, images -> `<image>`); the Thinking one does not. A port hard-coded
to the Thinking template rendered multi-part content wrongly for the 350M (caught by the byte-level `/apply-template` comparison). The variant is now detected from `tokenizer.chat_template`; both verified byte-identical to llama.cpp's jinja.
Server memory warning is now computed from the real file sizes (it hard-coded the 1.2B footprint).
