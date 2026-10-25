# Block-granular prompt cache on local NVMe (experiment)

Status: **experiment branch**. Behind `--block-cache N` (0 = off, the default), so nothing changes unless asked for.

## Why

The existing on-disk prefix cache (`prefix_store`) writes one file per completed prefix, keyed by its exact token
vector, and finds the longest match by scanning every entry and comparing token vectors. Two consequences:

- **Storage is O(prefix) per turn.** A 12-turn conversation that grows to 3,300 tokens stores ~541 MB at fp32 KV
  (24.0 KB/token), because every turn rewrites a prefix that is 99% identical to the previous one.
- **Lookup is O(entries x tokens).** Fine for a few hundred entries; not a cache.

A 128 MB budget therefore holds about **a quarter of one conversation** for the 1.2B model.

## What this does

A prefix is cut into fixed blocks of `N` tokens. Each block is stored **once**, content-addressed by a hash chain:
block *i*'s key is `mix(hash(block i-1) ^ tokens(block i) ^ model identity)`. Two conversations that share a system
prompt therefore share the physical blocks, and a lookup costs `O(prefix / N)` hashes with no token comparison.

```
<snapshotdir>/blocks/<2 hex>/<16 hex chain hash>.blk
  header  magic | version | state_id | chain | parent | n_tokens | geometry | sizes | crc32 | created
  body    tokens(N) | K/V rows of this block, per attention layer [+ int8 scales] | conv state at end of block
```

Body layout is a prefix blob restricted to the block's own rows, so blocks of one chain concatenate per layer into
exactly what `LmEngine::import_prefix` already takes. No change to the restore path.

### The tip: one exact end per conversation

Blocks alone strand up to `blk-1` tokens. A conversation that stops at token 2,253 only has whole blocks up to 2,176, so the
next turn re-prefills 77 tokens that never became a block. The tip is that stretch: one variable-length block, parented to
the last block that actually exists, replaced on every turn (one per conversation).

It is captured at the end of **generation**, not at the end of the prompt, so it covers prompt + reply. The next turn of the
same conversation therefore restores to exactly where the model stopped and re-prefills only what the client added.

Measured on a 2,270-token continuation turn after a restart (350M, greedy, same answer both ways):

| | cached | re-prefilled | prefill |
|---|---|---|---|
| blocks only | 2,176 | 94 tokens | 442 ms |
| with tip | 2,253 | **17 tokens** | **147 ms** |

Cost: one tip per conversation, 1.9 MB here, overwritten each turn — versus 96 MB for a whole-prefix tail at this length.

### Why blocks are only taken at block boundaries

The conv state is `bx(t-2), bx(t-1)` — a rolling 2-timestep value that is a function of the *whole* preceding prefix
(it feeds the next layer's input, so it cannot be recomputed from KV alone). A chain may therefore only be resumed
where a block recorded it. That is why prefill slices are aligned to block boundaries during capture: the slice end
is the only position whose conv state is a cacheable value.

This is the structural difference from DeepSeek's design. Their SWA state lives in a per-sequence slot and, when it
is missing, they replay a bounded 128-token window and accept an **approximate** KV (their words). Here the state is
captured exactly at every boundary, so a restored chain is **bit-identical** to a cold prefill — measured, not assumed.

Cost: one block is captured with `LmEngine::export_block`, which reads `O(N)` rows straight from the live slot
(`gpu::read_at`) instead of snapshotting the whole prefix. Capturing at every boundary during a sliced prefill is
therefore `O(prefix)`, not `O(prefix^2)`.

## Measured (N100 iGPU, 350M Q8_0, fp32 KV)

`scripts/test_blocks.py` — 9 checks, ALL PASS:

| Check | Result |
|---|---|
| blocks captured while prefilling | 4 blocks / 12.3 MB for a 626-token prompt |
| restarted server restores a chain | `disk_hits=1`, 512 tokens reused |
| **restore is exact** | identical tokens, cold vs restored |
| sibling conversation reuses shared blocks | 607 of 626 tokens cached, **0 new blocks written** |
| sibling served from blocks == cold prefill | identical tokens |
| storage, 6 growing turns | **17.8 MB vs 98.4 MB** whole-prefix = **5.53x** (blocks + tip) |
| restart then continue the same conversation | 17 tokens re-prefilled instead of 94 (442 -> 147 ms) |
| tip-served continuation == cold prefill | identical tokens |

Per-block cost: 3.08 MB at N=128 (24.6 KB/token), i.e. blocks of one chain share nothing wasted.

### Cost: prefill (it is now free)

Prefill runs at full width: no slice is forced to end on a block boundary. Instead, while the single prefill runs, each
conv layer also writes `b*x` of the two rows before every boundary into a capture buffer, and the blocks are exported
from K/V afterwards in one pass. Interleaved A/B, min of 4 runs, 1,400-token cold prefill:

| block size | prefill | vs unsliced |
|---|---|---|
| off | 5,118 ms | — |
| 128 | 5,151 ms | **+0.6%** |
| 512 | 5,124 ms | **+0.1%** |

This needs no kernel change: `b*x` is what `lm_conv_state` already stores, and it can be pointed at any two rows of the
layer input, so the capture is one extra launch per conv layer.

The captured state is exact, not approximate — `scripts/test_blocks.py` proves a restored chain produces the same tokens
as a cold prefill. Cost is a per-slot buffer of `n_conv x (max_ctx/blk) x 2d x 4` bytes, allocated on first use:
4.4 MB per slot for the 350M, 15.6 MB for the 1.2B at 4k context.

Before this, prefill had to stop at every boundary to read the conv state, which cost ~50 ms per extra `step()` call:
+10.7% at 128-token blocks, +3.1% at 512. Two other suspects were measured and ruled out along the way: batching or
removing the per-layer i915 `finish()` changed nothing, and skipping the block capture changed nothing.

Warm-path check: after a restart, a 1,938-token prompt with a 768-token chain restored prefilled 1,170 tokens in
3,475 ms (337 tok/s), where a cold prefill of the full prompt is ~5,750 ms.

## Long context: 15,013 tokens on the 350M (ctx 16384, fp32 KV)

| | cold | warm, after a restart |
|---|---|---|
| prefill | 84,601 ms (177 tok/s) | **721 ms** |
| tokens re-prefilled | 15,013 | **37** |
| answer | identical | identical |

Chain: 112 blocks, 345 MB on disk (3.08 MB per block, 24.6 KB/token), plus a tip. **117x faster to first token.**

Two things this test surfaced that short prompts hide:

- **The device prefix cache budget can be smaller than one prefix.** A 15k-token prefix is ~360 MB against a 128 MB
  default `--prefix-cache-mb`, and the restore used to be dropped on that check, silently discarding a chain that had
  just been read from disk. A prefix that does not fit the cache is now used once and not cached.
- **The tail is not always one block.** When prefill and generation together leave a longer stretch uncaptured, the tip
  absorbs all of it (observed at 1,829 tokens in one run). That is correct but costs disk: prefer a block size that
  divides the usual conversation shape, or accept a bigger tip.

For long contexts, set `--prefix-cache-mb` above the prefix size to keep a restored prefix reusable; otherwise each
restore is a single use.

## Decision: the tip stays best-effort

Making the tip deterministic needs ids rather than text: either a client that sends token ids, or a server that owns the
conversation (`previous_response_id` style, replaying its own stored history). Neither changes the cache, and neither makes
prefill faster — they only make the existing saving deterministic. That saving is 332 ms per chat turn on the 350M and
950 ms on the 1.2B, but only 0.2% of a turn at 15k context, where the blocks do all the work. It is bounded on a miss and
never wrong, so best-effort is the right default for an OpenAI-compatible endpoint. If stateful conversations are ever
added they would make the tip deterministic as a side effect, and would earn their keep by not re-sending history every
turn — not by fixing the tip.

## Eviction: leaves first, orphans reclaimed, TTL per class

A chained block is unreachable once any link above it is missing, so evicting a mid-chain block does not free one block,
it strands its whole subtree. LRU over everything therefore behaves badly under pressure: the oldest blocks are the
ancestors of everything, so the first eviction of a squeezed store takes the cache with it. Measured before the change -
one eviction, nine orphans, a nine-block chain reduced to two blocks that no longer restored anything.

The policy now:

- **only leaves are evictable**, least recently used first, so a shared prefix is never orphaned while its descendants
  still hold data;
- **the budget may overshoot by the single block just written** rather than evict an ancestor (the block being saved is
  excluded, and it is often the only leaf); the next save evicts the previous leaf;
- **orphans are reclaimed immediately**, so the budget always holds reachable cache instead of dead weight;
- **TTL per class**: `--tip-ttl-h` (the tail of one conversation, superseded every turn) and `--block-ttl-h` (durable
  shared prefixes), both `0` = keep forever. Both run off *last use*, not creation, so a block that keeps being touched
  never expires.

`scripts/test_evict.py` (10 checks) drives it: a 30 MB budget against a 1.2k-token conversation plus a branch of it
evicts 3 leaves with 0 orphans, and after a restart - cold RAM cache, so this is the disk policy - the shared ancestors
still restore **1152 of 1180** tokens. `/health` reports `evicted`, `expired`, `orphans`, `dangling` (must stay 0) and
`links`, which is what located the `keep`-exclusion bug.

## Known limitations

- **The tip is best-effort: it fires only when the echoed turn re-tokenises to the same ids.** A KV cache is keyed on
  token ids, not text, so `" hello"` vs `"hello"` is a different bookmark, and a chat template that carries a Thinking
  model's reasoning in a separate field will not reproduce the generated ids on the way back. Measured: it fires for the
  350M through the chat path (645 of 667 tokens reused) and for the 1.2B through raw completions (492 of 508), and it
  misses for the 1.2B through chat and for the 350M through completions — i.e. it depends on the template and the
  tokenizer seam, not on the model. A miss is safe and bounded: the block chain still restores the head (512 tokens of
  667 in the chat case) and the answer stays bit-identical; the cost is up to `blk` tokens of re-prefill that turn.
  `scripts/test_tip_roundtrip.py` asserts the contract (always exact, head always restored) and reports a seam drift as
  INFO, because text round-trips cannot be id-exact by construction. `test_blocks.py` on the 1.2B will report the tip
  check as a failure for the same reason; pass `FERRO_TEST_NO_THINK=1` there.
- Tested at 128-token blocks on the 350M (fp32 and int8) and the 1.2B (fp32); the 1.2B at 16k was not run (prefill is
  4x slower per token, ~4 minutes per pass).

## Prior art: what is and is not novel (checked 2026-10)

The mechanism set now ships in production engines. Mapping, so a reviewer can check every row:

| here | equivalent elsewhere |
|---|---|
| blocks keyed by tokens + parent hash (chained) | [**vLLM APC**](https://docs.vllm.ai/en/stable/design/prefix_caching/) ("we hash each kv-cache block by the tokens in the block and the tokens in the prefix"); [**TensorRT-LLM**](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html) blocks in a radix search tree; [**d-inference**](https://github.com/Layr-Labs/d-inference/blob/master/docs/reference/ssd-kv-cache.md) `<blockHashHex>.darkbloom-kv` files |
| prefix sharing + LRU over those blocks | [**SGLang RadixAttention**](https://arxiv.org/html/2312.07104v2); TensorRT-LLM's radix tree |
| disk/remote tier behind the same prefix identity | [**LMCache**](https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html) (GPU/CPU/disk/remote); [**SGLang HiCache**](https://docs.sglang.io/docs/advanced_features/hicache_design) L1 GPU, L2 host, L3 `file`/`mooncake`/`hf3fs`/`nixl`/`aibrix`/`eic`/SeaweedFS |
| **conv/recurrent state at a boundary -> exact resume** | [**SGLang Unified Radix Cache**](https://www.sglang.io/blog/unified-radix-cache): MAMBA/SSM checkpoints as components, copy-on-write, "the deepest boundary accepted by all components", int8 checkpoints, default since v0.5.16 |
| **tip (prompt+output, exact stop point)** | SGLang stores the full token path, "input *and* output", per completed request |
| whole-prefix exact reuse, no dedupe or tiering | **llama.cpp** slot `cache_prompt` + `--prompt-cache`; **Ollama** RAM-only ([open request](https://github.com/ollama/ollama/issues/17247)) |
| the approximate alternative | **DeepSeek** SWA state slot + bounded replay (their own word: approximate) |

**So the design is not novel, and this doc should not claim it is.** It is the same mechanism class as
SGLang's Unified Radix Cache + HiCache, reimplemented for hardware and a model family those engines do
not serve:

- **target**: a 6 W N100 iGPU with int8 KV in unified memory - no datacenter accelerator, no RDMA, one disk;
- **evidence**: a published bit-identical contract (restore == cold prefill, tokens compared) rather than
  correctness by construction, including at 15k context;
- **two different choices, not new mechanisms**: the tip is one mutable per-conversation block (1.9 MB,
  overwritten each turn) instead of a shared reference-counted tree node, and capture is zero-slice
  (point `lm_conv_state` at the two boundary rows) instead of a separate checkpoint pass.

Where nothing equivalent exists: **hybrid conv/SSM models on iGPU/CPU-only consumer boxes**. The engines
those boxes actually run top out at RAM-only reuse (Ollama) or one whole-prefix file per slot
(llama.cpp), and llama.cpp's exact restore for hybrid/recurrent and SWA models is an open bug (`/slots`
restore yields no reuse, [#28194](https://github.com/ggml-org/llama.cpp/issues/28194)) - the exact failure mode this design exists to avoid. At this prefill
rate the technique is also worth more than on a 4090, where the same 15k prefill costs a few hundred
milliseconds: here it is 84.6 s -> 721 ms.

## Replicating the DeepSeek cache layer: one box, or inference + cache box

DeepSeek's efficiency is four legs and the cache cluster is one of them: prefill/decode disaggregation
(EP32 prefill, EP144 decode, dual-microbatch overlap), the EP comms stack (DeepEP/EPLB), **3FS as the KV
store** ("a cost-effective alternative to DRAM-based caching"), and sparse attention (DSA). The
cache-specific parts that transfer:

- KV pages at **64-token granularity** (FlashMLA), the same unit the API cache uses;
- entries are **prefix units** persisted at three triggers - request boundaries **including the end of
  model output**, detected common prefixes, and fixed token intervals - and a request hits only on a
  **full unit match** ([DeepSeek context caching](https://api-docs.deepseek.com/guides/kv_cache/));
- the store is a **disaggregated NVMe fleet**: [3FS](https://github.com/deepseek-ai/3FS), 6.6 TiB/s aggregate,
  **40+ GiB/s per client**, CRAQ chain replication, metadata in FoundationDB (system overview:
  [open-infra-index](https://github.com/deepseek-ai/open-infra-index));
- for SWA/sparse generations they keep one state slot per sequence with **bounded replay** and accept
  approximate KV.

Sizing is where this box differs by orders of magnitude. Feed rate needed = prefill rate x KV bytes/token:

| | this N100 (350M, 15k ctx) | DeepSeek per node |
|---|---|---|
| prefill | ~177 tok/s | ~73.7k tok/s |
| KV per token | 24.6 KB fp32 / ~12.3 KB int8 | MLA latent, 64-token pages |
| **feed rate needed** | **~4.4 / ~2.2 MB/s per stream** | **40+ GiB/s per client** |

Gigabit therefore carries ~28 concurrent **fp32** restores here (~56 at int8), so the fabric is not the
design problem - latency and prefetch are.

### A. One box

Present already: chained content-addressed blocks on NVMe, exact conv-state checkpoints, the tip, the
async writer, the LRU budget, `/health` + `/metrics`. To match the idea:

1. **Async prefetch** - match the chain before the slot is free and overlap the read with the queue wait,
   so disk latency leaves TTFT. ~1 day.
2. **Direct-to-device restore** - apply blocks at their slot offsets instead of assembling a host blob
   (drops an `O(prefix)` copy: 345 MB at 15k). Needs `restore_range` on the engine. ~2 days.
3. **Prefix-unit detection** - when several requests share a prefix, materialise it as a durable unit so
   the second and later conversations are cheap too, not just the same one. ~1 day.
4. **Leaf-preferring eviction + TTL by traffic class** - evicting a mid-chain block truncates every
   descendant. ~0.5 day.
5. **Tier metrics** - hit rate per tier, bytes served, prefetch wait: the analogue of
   `prompt_cache_hit_tokens`. ~0.5 day.

Footprint: a 15k-token chain is 345 MB fp32 / 172 MB int8, so 1 TB of NVMe holds ~2,900 such chains.
Capacity is not the constraint on one box.

Two notes carried over from the earlier roadmap draft, because they are design constraints rather than
tasks:

- **A hot RAM tier is deliberately absent.** On this box the iGPU and the CPU share physical memory, so a
  "device cache" and a "RAM cache" are the same memory; a separate RAM tier is just a bigger
  `--prefix-cache-mb`. At 24.6 KB/token an exact 4k prefix costs 96 MB, so that tier holds about one
  entry either way. The tip is what makes a tier that small sufficient.
- **Choose N per workload, and N=64 is the better default for this box.** Tail size is prompt length mod N, so the block
  size sets how many tokens a warm turn re-prefills below the last whole boundary. Measured on a 14,705-token prompt,
  same conditions, 350M Q8_0:

  | | N=64 | N=128 |
  |---|---|---|
  | cold prefill | 82.1 s | 82.1 s |
  | tail prefill (warm) | **707 / 723 ms** | 1315 / 1311 ms |
  | warm TTFT | **1421 / 1984 ms** | 2000 / 2425 ms |
  | cached (tail) | 14,656 (49) | 14,592 (113) |
  | blocks, store | 229, 362.9 MB | 114, 353.9 MB |

  Halving N halves the average tail at no measurable cost in cold prefill (+0.0% here, against the +0.6% that N=128
  measured versus unsliced) and 2.5% more storage, for the extra per-block conv states. The earlier note that N=512 is
  cheapest was about slice overhead and is the wrong lever for the tail: N=512 doubles it.

### B. Inference box + cache box

The block format is already self-describing (magic, version, state_id, chain, parent, n_tokens,
geometry, crc32), so the split is an interface extraction, not a redesign:

```
BlockBackend
  bool has(uint64_t chain);
  bool get(const std::vector<uint64_t>& chain, Sink* out);   // ranged, streamed, CRC per block
  void put(uint64_t chain, const uint8_t* kv, size_t, const uint8_t* cst, size_t);
  void touch(uint64_t chain);
    LocalDir      (today: a directory of files)
    RemoteBlocks  (new: one batched multi-GET per chain)
```

Rules that make it safe:

- **index local, data remote** - the chain index (112 entries at 15k) stays on the inference box, so
  matching never pays a round trip; only a miss path talks to the cache box.
- **one request per chain, not per block** - 112 blocks x RTT would dominate; send the hash list, receive
  a concatenated CRC-checked stream.
- **never on the correctness path** - miss, CRC failure or timeout means recompute from zero. Same
  best-effort contract the hosted API advertises.
- **write-back after the response** - a slow cache box must never delay a token.
- **identity in the key** - `state_id` (model + kernels + KV dtype) prevents cross-build poisoning; it is
  the `modelHash` check d-inference performs on every load.
- **decide honestly** - for one box, local NVMe (1-2 GB/s) beats gigabit: a 345 MB chain is ~0.3 s local
  against ~2.8 s at 125 MB/s, still 30x better than the 84.6 s cold prefill but a downgrade. Split only
  when one cache must serve several inference boxes, the inference box has no NVMe headroom, or the cache
  must outlive a reinstall.

Acceptance gate for the split already exists: `scripts/test_blocks.py` (exactness, sharing, storage) must
pass unchanged against `RemoteBlocks`.

### Not copying

3FS itself - CRAQ chain replication, a FoundationDB metadata service, USRBIO, RDMA. The consistency
requirement here is "immutable block + atomic rename", which POSIX provides; the metadata is an index
rebuilt from the directory. Their scale is why they need a storage *system*; this needs a directory and,
at most, a small object server.

## Running it

```
./ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --snapshot-dir /path --block-cache 128
python3 scripts/test_blocks.py           # needs models/ and a built binary
```
`--block-cache 0` (default) keeps the whole-prefix snapshot path exactly as before.
`/health` exposes `block_cache: {block, entries, mb, saved, disk_hits}`; `/metrics` exposes four counters/gauges.
