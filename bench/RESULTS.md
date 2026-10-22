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
Decode profile (`FERRO_LM_STEPTIME`): enqueue 2-4 ms (overlapped) + device ~22-25 ms per token. (An earlier version of this note called the gap to a "15 ms ideal" pure launch
latency and promised ~1.3x from fusion; that was wrong - see the fusion study below: the device streams ~510 MB/token, not 0.38 GB, because the fp32 block scales ride along.)
**Chat template differs per model**: the 350M GGUF carries a template that handles typed content arrays itself (no "\n" join, system arrays concatenated, images -> `<image>`); the Thinking one does not. A port hard-coded
to the Thinking template rendered multi-part content wrongly for the 350M (caught by the byte-level `/apply-template` comparison). The variant is now detected from `tokenizer.chat_template`; both verified byte-identical to llama.cpp's jinja.
Server memory warning is now computed from the real file sizes (it hard-coded the 1.2B footprint).

### Decode kernel fusion study (LFM2.5-350M / 1.2B, 2026-10-22) - what it bought and what it did not
Method: interleaved A/B rounds (config order rotated each round so thermal/DVFS drift hits every config equally), median of per-step `enqueue + device wait` over 55 decode steps.
The iGPU drifts +-1.5 ms between runs, so every claim below is a median of >= 4-6 interleaved rounds. `FERRO_LM_ABL=op,..` skips ops, `FERRO_LM_DUP=op,..` launches them twice,
`FERRO_LM_NOEOS=1` keeps generating past EOS (timing experiments; outputs are garbage), `FERRO_LM_FUSE=0|1|2` selects unfused / default / +prologue GEMVs.

1. **Where the 350M step goes (27 ms).** Skipping all ~87 small kernels (norms, silu, conv, qkv/rope, kv write, attention): 19-21 ms left. The GEMMs alone are at the bandwidth ceiling: 453 M weights x 1.125 B
   (int8 + fp32 scale per 32) = 510 MB per token / 25.9 GB/s = 19.7 ms. So the GEMMs cannot be sped up (only a smaller weight format would), and the small-kernel set is the only fusible part.
2. **Marginal cost of a launch, measured in situ** (launch each op twice, 4 interleaved rounds): +86 launches = +3.65 ms, i.e. **~42 us per launch**. Per-category skipping looked like "free" (+-0.3 ms) because skipping a
   producer also feeds stale data to the GEMMs and the noise is 1-2 ms; duplicate-launch is the reliable probe. So the whole small-kernel set is worth ~3.7 ms (13% of the step) and a perfect fusion could never give 1.3x.
3. **Fusions shipped (default, `FERRO_LM_FUSE` unset/1)**: decode causal-conv gate + conv-state update in one kernel (`lm_conv_dec_q`, 10 launches/token), q/k RMSNorm + rope + K/V cache write in one kernel
   (`lm_qkv_post_kv`, 6 launches/token saved, also used by prefill). Result: 350M single-stream decode 29.0 -> 26.1 ms/step (**1.11x**, 34.5 -> 38.3 tok/s; unfused runs 27.4-30.1 vs fused 23.4-27.0).
   Decode-vs-prefill self-check cosine 0.99961 (unfused 0.99957); chunked prefill bit-identical; `lm_parity.py` identical fused vs unfused (350M: 12/12, TV 0.0172; 1.2B: 12/12, TV 0.0168); `make test`, `test_serve_lm.py` ALL PASS.
4. **1.2B and multi-stream: no measurable gain.** The saving is a fixed ~2.5-3 ms/token, i.e. ~4% of the 1.2B's 59 ms step (below the noise floor; its unfused runs span 59-76 ms) and nothing at all once several streams share a step.
   Server A/B (`bench_lm_serve.py`), fused vs `FERRO_LM_FUSE=0`: 1.2B 1/2/4 streams 16.8 / 31.0 / 42.5 vs 16.3 / 30.6 / 44.5 tok/s; 350M 1/2/4/8 streams 34.9 / 62.1 / 101.7 / 134.2 vs 33.2 / 65.5 / 101.1 / 130.2 tok/s.
5. **Prologue-fused GEMVs (opt-in `FERRO_LM_FUSE=2`): break-even, not shipped as default.** `q8gemv_p` computes `rmsnorm(X+D)*w` or `silu(gate)*up` inside the consumer GEMV (activations built once per work-group in local
   memory, bit-identical quantisation) and removes 49 more launches/token (all resid_norm, silu_mul, the lm_head row copies). With the prologue skipped (timing-only) the step is 24.4 ms (-4 ms) - the launch saving is real - but every
   work-group repeats the prologue: first version +13 ms (uncoalesced), coalesced version +3.8 ms, fatter work-groups (64 rows) bring it to break-even: 350M 26.1 (default) vs 27.1 ms (FUSE=2); 1.2B 59.2 vs 58.5 ms. Per-site
   (`FERRO_PRO_MASK`) no site wins alone (-2..-5%); prefetching the next weight block to hide the latency made it worse (29.8 ms: register pressure slows the weight loop). Repeated-prologue cost scales with the number of
   work-groups (`lm_head` has 256-1024), which is why the idea does not pay on a 24-EU GPU where only ~21 work-groups are resident at once.
**Conclusion:** single-stream decode now runs at ~75% of the weight-streaming limit on both models (350M: 19.7 ms ideal vs 26.1 ms; 1.2B: 1.17 GB at 25.9 GB/s = 45 ms ideal vs 59 ms). The remaining ~25% is the ~70 small launches (~3 ms)
plus GEMV tail effects; the lever left is fewer weight bytes (Q4, or fp16 scales: the fp32 scales are 11% of the traffic), not more fusion.

### d1 over HTTP: `POST /v1/systemone` + server-side crops (2026-10-22)
`decide` was CLI-only; `serve --d1 d1-omni.gguf [--d1-mmproj ...]` now answers llama.cpp's `/v1/systemone` contract from the resident worker (the pipeline moved out of `main.cpp` into `DecisionService`:
prepare -> infer -> format; the CLI uses it too and its output is byte-identical to before on text / batch / PNG / JPEG / audio requests).
Latency, server-side (`prep` = parse + decode + crop + preprocess, `run` = encoders + one packed forward): text, 3 questions / 118 tokens **240 ms**, 1 question **110 ms**; 400x400 crop of a 1080p frame (593 tokens) **1.25 s**
(prep 10-50 ms, nearly all the rest is the vision tower); whole 1080p frame (6986 tokens) **21.8 s**. A frame + `crop` field costs 1293 ms vs 1250 ms for the same region uploaded pre-cropped (the 43 ms is the 1080p JPEG decode) and
1287 ms for a client that decodes, crops and re-encodes itself - i.e. the field saves the client work and the 17x between crop and whole frame is why detector boxes should always be sent as regions.
d1 is compute-bound at ~500 tok/s: 18 concurrent requests merged into one pass took 8.7 s, the same as running them back to back (batching changes the order of completion, not throughput), so the pass budget defaults to 2048 tokens
(`ServerOptions::d1_batch_tokens`) to stop short requests waiting behind a large batch. With 3 chat streams decoding, a 1-question decision takes 143 ms instead of 110 ms.
Memory (cgroup anon+shmem, includes ~0.35 GB of other processes): d1+mmproj 1.9 GB, 350M chat 1.7, EG2+mmproj 2.3, d1+chat 2.1, EG2mm+d1 2.1, all four 3.4 GB; the all-four server was OOM-killed (4 GiB cap) on its first
multimodal embedding request, so the startup check now sums every loaded GGUF (700 MB + 1.3 x files + KV) and prints a headroom note above 80% of the container limit.
Tests: `scripts/test_serve_d1.py` (50+ checks: server == one-shot CLI, image / audio / image_url parts, crop {x,y,w,h} == {x1,y1,x2,y2} == normalised == a pre-cropped PNG, pad, clipping, crops lists, include_full, 12 error cases, 18-way concurrency).

### LM prompt-prefix cache (2026-10-22)
`LmEngine::snapshot / restore` copy the K/V rows [0,P) of every attention layer and the conv state (device to device, ~23.5 KB per token on LFM2.5: 6 attention layers x 8 kv heads x 64 x 2 x 4 B, plus 80 KB of conv state);
the server keeps them in an LRU cache (`--prefix-cache-mb`, default 128) and `gen_start` restores the longest cached prefix of the prompt (always leaving >= 1 token to prefill for the logits). Cut points: the end of a leading system
message (verified to be an exact token prefix of the full prompt) and the longest common prefix with the last 8 prompts (learns prefixes of plain completions: miss, miss + learn, hit).
Exactness (`FERRO_LM_SELFCHECK`, 350M and 1.2B): restore into a different slot + prefill the rest vs one prefill: cos 0.9998-1.0 (bit-identical when the suffix chunk shape matches; the rest is the known chunking noise), a decode step after the
restore == the donor's decode (max|diff| 0), the snapshot survives the donor advancing and can be restored again. Server: greedy answers with / without the cache identical on 6 digests and under 6-way concurrency; tools in the system block are part of the key.
Speed (system prompt 1396 tokens + a ~30-token digest, 24 generated tokens, `cache_prompt` true vs false on the same server): 350M 4265 -> 1083 ms per request after the first (3.9x; 8 requests 34.1 s -> 12.0 s = 2.8x),
1.2B 12447 -> 2268 ms (5.5x; 99.7 s -> 28.6 s = 3.5x). Prefill of a 714-token cached prefix: 1603 ms cold -> 129 ms warm (12x; the warm 129 ms covers the restore, the 32 uncached tokens and the first sampled token; writing a 732-token snapshot measured 10-17 ms).
Cost of caching on a cold request: +4.5% (350M 1519 -> 1590 ms, 1.2B 5494 -> 5748 ms for a 741-token prompt); the snapshot is 10-17 ms of that, the rest is the extra prefill chunk boundary. Memory: 16.8 MB for 714 tokens;
the budget is enforced by LRU eviction (8 MB test: 9 evictions, never above budget) and is included in the startup memory estimate.
Not done: a snapshot at the end of every prompt (would give multi-turn chats a hit one turn earlier at the price of churn), prefix caching for d1's question prompts (compute-bound at ~500 tok/s with ~40-token questions, so little to save).

## Exactness, scheduling, speculation, memory (LM serving study)
N100 iGPU, 350M / 1.2B-Thinking Q8_0, interleaved runs, greedy unless noted. Details and flags: README "Exactness, scheduling and operations".

**Batch / chunk invariance** (`scripts/test_batch_invariance.py`, engine `FERRO_LM_SELFCHECK=1`): a stream alone vs among N concurrent streams, greedy and T=0.8 seeded.
Before: 2 of 4 prompts differed at >= 5 streams. After: identical at 2, 3, 4, 5, 6, 8, 12 streams; split prefill (1+640, 2+639, 3+638, 320+321) and cache restores `max|diff| 0`.
Aggregate decode tok/s (350M, policy off -> on): 1 stream 33-35 -> 34; 2: 56-57 -> 61-67; 4: 95-99 -> 98-100; 5: 95-97 -> 91; 8: 129-130 -> 127.5.

**Chunked prefill** (1615-token prompt in flight, `scripts/test_serve_sched.py`): d1 decision 2187 ms -> 170-196 ms (alone 108-113), longest decode stall 2278 -> 360 ms (72 tokens kept flowing),
short chat 2088 -> 1041 ms; the prompt itself 4.5 s idle, 6.2-6.8 s under contention.

**Speculative decoding** (`scripts/bench_spec.py`, 350M, drafts off -> on, tokens/s): copy a list 31.6 -> 52.3 (1.65x, 67% accepted); extract ids 26.6 -> 33.2 (1.25x, 48%); rewrite a log 28.6 -> 31.9 (1.12x, 59%);
write code 38.6 -> 41.6 (1.08x, 52%); free-form 37.0 -> 37.9 (1.03x, no drafts). 1.2B-Thinking: copy 1.01x, extract 0.99x, rewrite 1.02x, free 1.08x, code 1.07x (within noise; reasoning prose does not copy).
Before the "3-token match until proven" rule, free-form text lost 4% (0.96x) to coincidental 2-token matches. Verification window cost (`FERRO_LM_WINBENCH=1`, median of 15, incl. logits read-back),
350M: 1 row 24.8 ms, 2: 27.1, 3: 27.7, 4: 32.9, 5: 42.8, 6: 46.5, 7: 45.4, 8: 47.3 (plain decode step 26.6); 1.2B: 56.9, 58.5, 72.4, 71.5, 111.9, 114.7, 124.2, 129.0 (plain 64.2). The 8-row GEMV variant is the
expensive part, hence short default windows. Exactness: verified window rows == serial decode rows `max|diff| 0` (f32 and int8 KV, one and two sequences), commit of 1/2/3/5 accepted rows then decode == serial `max|diff| 0`.

**int8 KV cache**: 350M, 4 x 4096: KV 384 -> 108 MB (resident -276 MB). Parity vs llama.cpp (same Q8_0 weights): top-1 12/12 both, mean TV 0.0172 -> 0.0276 (prompts 6/8/9 worst: 0.045 -> 0.096, 0.031 -> 0.073, 0.031 -> 0.058).
Decode at a 1921-token context: 350M 23.1 -> 19.6 tok/s, 1.2B 12.0 -> 10.9 tok/s (prefill unchanged: 380 -> 383, 125 -> 128).

**Memory admission** (transient above idle, sampled every 10 ms; later requests understate it because freed arenas stay resident): d1 image 256x256 +42 MB, 640x360 +74, 1280x720 +166, 1920x1080 +293 (21.8 s, ~7000 vision tokens);
EG2 image 1280x720 +261, 1920x1080 +375 (69.5 s). Plan estimate vs measured: 350M + d1 + EG2 + mmproj 3585 vs 3373 MB.

**Persistent snapshots**: 1881-token system prompt on the 350M: first request after a restart 5927 ms (cold) -> 835 ms (1856 tokens restored from disk), answer identical to the uncached one; snapshot 43.6 MB (f32 KV).

**Measurement note.** Parity here is judged against llama.cpp running the *same Q8_0 weights* (`lm_parity.py --llama` pointed at a server on the Q8_0 file); an F16 oracle measures weight-quantization noise on top
(350M: mean TV 0.0426 against F16 vs 0.0172 against Q8_0, identical code) and flips the argmax on a near-tie prompt - not a regression.

## 2026-10-22: branch `perf-fixes` - pool parking, split-K decode attention, end-of-prompt snapshots, penalty window (kernel-level results)

Built and tested in a worktree while the production server kept serving; the tests load no model, so end-to-end numbers are still pending (list at the end).

**CPU pool parking.** Idle pool workers spun on `_mm_pause` and then `sched_yield()` forever: the live `serve` (2 threads) kept its one worker at 100% of a core
for its whole life (3.9 h of CPU in 3.8 h of uptime, 62% of it in the kernel), inside the shared CPU+iGPU power budget. Workers now park on a futex after a 200 us
spin (`FERRO_POOL_SPIN_US`). `test_kernels`: 3 idle workers use 0.6 ms of CPU in 300 ms (spinning pool: ~900 ms); 300 dispatches (100 of them after the workers had parked), results exact.
**Every iGPU figure above this section was measured with that worker spinning** - re-measure the baselines before comparing against them.

**Split-K decode attention** (default; `FERRO_LM_ATTN_SPLIT=0` = the per-head kernels). One attention layer at the 350M decode shape (16 q / 8 kv heads, hd 64),
device time per launch, best of 3 interleaved rounds of 30 launches:

| keys x streams | f32 KV: per-head -> split-K | int8 KV: per-head -> split-K |
|---|---|---|
| 1921 x 1 | 2.82 -> 1.20 ms | 3.76 -> 1.30 ms |
| 1921 x 4 | 8.01 -> 4.43 ms | 11.46 -> 4.44 ms |
| 200 x 1 | 0.35 -> 0.29 ms | 0.46 -> 0.34 ms |

The per-head times x 6 attention layers (17 ms f32, 23 ms int8 at 1921 keys) account for the context-dependent decode cost measured end to end earlier
(+17 / +25 ms per token at a 1.9k context): the old kernel was latency-bound, which is also why the int8 cache was slower than f32. Expected at a 1.9k context,
one stream: ~10 ms (f32) / ~15 ms (int8) less per token (int8 19.6 -> ~28 tok/s); per layer int8 is now 8% slower than f32 instead of 33% (equal at 4 streams) - a projection until measured end to end.
Accuracy: both kernels within 0.50 int8 output steps of a double reference (GQA 2 and 4, f32 and int8 KV, 1..1921 keys); a row's split-K output is bit-identical
alone, in reversed batch order and with a different slice stride. Headroom left: 1.2 ms vs a ~0.3 ms KV-streaming floor per layer (the per-slice V loop is serial per lane).

**End-of-prompt snapshots** (`--no-prefix-tail`) and **penalty window / scope** (`repeat_last_n`, `penalize_prompt`, `--penalty-scope generated`): code plus the
sampler unit test only.

**Pending (needs the server stopped):** `lm_parity.py` (350M, int8 and f32 KV), `test_batch_invariance.py`, `test_serve_lm.py`, `test_serve_prefix.py`,
`d1_parity.py` and the EG2 cosine check (the pool and `gpu_cl.cpp` are shared), `FERRO_LM_STEPTIME` at 0.2k / 2k context, and the hydra suite (TTFT, cached tokens
per request, draft acceptance, pass rate) with and without `--penalty-scope generated`.

**fp16 KV cache** (`--kv-dtype f16`, opt-in, GPU only - init refuses it without the fp16 kernels and split-K decode attention). Half storage, fp32
arithmetic; every stored value is clamped into the fp16 range first (K is RMS-normalised, V is not: max |V| 6.1 on random data, 0 clamped).
Kernel-level tests (`test_kernels`, no model): the f32 write kernel matches a double reference to 9.1e-7; the f16 store is the rounded fp32 value
(worst 1.2 fp16 ulp, i.e. rounding alone); split-K decode on f16 rows is 0.50 int8 output steps from a double reference at 1921 keys, the same as the
f32 and int8 paths. Device time per layer at the 350M shape, 1921 keys, 1 stream: f32 1.21-1.29 ms vs f16 1.18-1.34 ms across runs - no win at this
context (latency-bound, as expected); 4 streams not measured. Memory 4x4096: 192 MB vs 108 int8 / 384 f32.

After split-K the KV ranking inverted for single-stream decode: f32 is now both faster (1.20 vs 1.30 ms per layer) and better (mean TV 0.0172 vs
0.0276). The live serve still runs `--kv-dtype int8`, so switching it to `--kv-dtype f32` is the free quality win today (+276 MB of the ~1 GB
headroom, and every persisted snapshot goes cold because the dtype is part of the snapshot identity). f16 is the option for when that memory is needed.

**End-to-end decode, old vs new binary** (`generate`, 64 tokens, temperature 0, `FERRO_KV` per row, `FERRO_LM_STEPTIME=1`, median device
wait+read of the 64 steps, nothing else running; the old binary is `ferro-embed.bak-20261022-1732`):

| prompt | old f32 | new f32 | old int8 | new int8 | new f16 |
|---|---|---|---|---|---|
| ~200 tokens | 45.7 tok/s (21.9 ms) | 47.8 (20.9) | 45.2 (22.1) | 46.3 (21.6) | 46.9 (21.3) |
| ~1900 tokens | 31.7 tok/s (31.5 ms) | **40.2 (24.9)** | 26.8 (37.3) | **38.8 (25.8)** | 35.1 (28.5) |

Split-K pays only with context, as the per-layer numbers predicted: +27% (f32) and +45% (int8) at 1.9k tokens, nothing at 200. The int8
gain (11.5 ms per token) is the predicted size; f32 gains 6.6 ms. int8 is no longer faster than f32 anywhere measured (1.9k: 24.9 vs 25.8 ms)
and f16 buys memory, not speed (28.5 ms - the earlier "about equal to f32" guess was optimistic).

**Deployed** 2026-10-22: `scripts/serve_prod.sh` = `--kv-dtype f32 --prefix-cache-mb 256` (`plan` 2641 MB, runtime 3368 MB used / 728 MB
headroom, no new OOM kills). Live after the restart: idle CPU **0% of a core** (was 100% of one), warm prefix 749-token prompt 3294 -> 475 ms
with 727 tokens cached, 4 concurrent identical prompts bit-identical to the single run.

**Still not run** (no oracle in this container - `which llama-server` is empty, so llama.cpp has to be built first): LM parity, d1 parity,
EG2 cosine, `test_batch_invariance.py`, `test_serve_spec.py`, the speculative-window self-check under real batching, and the hydra A/B with
and without `--penalty-scope generated`.

**2026-10-22 validation window** (llama.cpp oracle built locally, CPU, `baselines/llama.cpp/build-cpu/bin/llama-server` on :8095, same Q8_0 file;
`scripts/lm_parity.py` per dtype via `FERRO_KV`):

| dtype | top-1 | per-prompt TV | TV <= 0.06 gate |
|---|---|---|---|
| f32 | 12/12 | 0.0066 - 0.0118 | 12/12 |
| int8 | 12/12 | 0.0045 - 0.0065 | 10/12 |
| f16 | 12/12 | 0.0013 - 0.0092 | 11/12 |

The int8/f16 gate outliers are the known quantisation-sensitive prompts (worst 0.096 at int8 in the earlier matrix): split-K, the pool and the dtype
plumbing cost nothing measurable. f16 lands between f32 and int8 on this metric, as a 10-bit mantissa predicts, and in practice nearer f32.

`test_batch_invariance.py` (throwaway 8-slot instance on :8096): **ALL PASS** - temperature 0 and 0.8, 4/4 prompts identical alone vs 6 and 8
concurrent streams. That is the end-to-end check of split-K together with speculative windows, which the kernel unit test could only approximate.

**#3 verified** - six-turn synthetic agent conversation, each turn appending to the previous prompt, temperature 0, int8 KV, old binary vs new:

| turn (prompt tokens) | old: cached / wall | new: cached / wall |
|---|---|---|
| 1 (590) | 0 / 1417 ms | 0 / 1413 ms |
| 2 (620) | 567 / 339 ms | 590 / 195 ms |
| 3 (650) | 590 / 343 ms | 620 / 177 ms |
| 4 (680) | 620 / 316 ms | 650 / 206 ms |
| 5 (710) | 650 / 353 ms | 680 / 217 ms |
| 6 (740) | 680 / 358 ms | 710 / 217 ms |

The old build cached the previous prompt minus its tail (a turn behind); the new one restores the previous prompt exactly, so each turn prefills only
its new reply plus tool output: 1.6-1.7x lower TTFT per turn after the first, ~30 fewer prompt tokens per turn in steady state.

Still open: `test_serve_spec.py`, `test_serve_lm.py`, `test_serve_prefix.py` (they take `--url`; a positional URL makes them exit), d1/EG2 parity,
f16 at 4 streams, one hydra round, and the copy-heavy penalty A/B. #6 and #7 are unimplemented: the measured decode floor leaves #6 about 1 ms per
token (~4%) behind a per-slot exactness flag across every GEMM variant, and #7's shared-token fraction has not been dumped yet.

### window2 (18:08-18:11) - the serving suites, both KV dtypes, penalty A/B

- `test_serve_spec.py --url`: **ALL PASS** on f32 **and** int8 (max_tokens and a stop string cutting inside a verification window land on the same token).
- `test_serve_lm.py --url`: **ALL PASS** (6 concurrent requests over 4 slots, server idle and healthy afterwards).
- `test_serve_prefix.py --url`: **FAILED** two checks. With the cache at 121.8 MB of 128 (7 entries, 19 evictions) the end-of-prompt snapshots
  (~28 MB each at ~1.1k tokens) dominate the LRU, so the suite's entry-set assertions and its `--small-cache` "most recent prefix is still cached"
  case (`cached_tokens: 0`) both break. Answers stay exact - this is cache bookkeeping and hit-rate, not correctness - but the documented
  prefix-cache contract regressed, so **production now runs with `--no-prefix-tail`** until the store policy is fixed.
- Penalty A/B (8 quoted identifiers, temperature 0): **8/8 identical copies both ways** - no signal at this length, so the default stays
  `repeated_penalty 1.05 over the whole context`. Repeat with longer prompts (300+ tokens of surrounding text) before drawing a conclusion.

**Fix planned for the tail store:** give tail entries their own small budget (or evict tail entries first), never let one evict the shared
system-prompt entry, and skip the tail when the device cache is within one tail of its budget. Re-run `test_serve_prefix` with tails on before
re-enabling it in `serve_prod.sh`.
### Tail store: fixed in code, not yet validated (commits d046d4a, 5f75fd5)

Two changes. **Trigger:** a tail is taken only when the conversation has already continued (some message has role `assistant` or `tool`). The old
test (`messages >= 2 || tools`) was true for every chat request, so one-shot system+user calls and every hydra request (it always sends tools)
left a ~28 MB entry behind. **Class:** tails now have their own budget - storing one first drops older tails that the new prompt extends, so a
lineage holds one entry instead of one per turn; tail bytes are capped at `max(--prefix-cache-mb / 4, px->bytes)`; and a tail is *skipped* rather
than evicting any non-tail entry, which is the property that keeps the shared system-prompt entry alive through an agent loop. Tails are also no
longer written to the 2 GB disk store.

Both commits build (`make -j1`, 13 s each). **Neither has been through `test_serve_prefix`**, so production still runs `--no-prefix-tail` and the
`validated-2026-10-22` tag still marks the last fully validated tree (`7c9d977`), two commits behind HEAD. To re-enable: one window that runs
`test_serve_prefix --url` with tails on at 128 MB and with `--small-cache`, then remove the flag from `scripts/serve_prod.sh` and re-tag.