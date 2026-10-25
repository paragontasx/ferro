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

### Cost: prefill slicing

A 1,400-token cold prefill, interleaved A/B, min of 4 runs:

| block size | slices | prefill | vs unsliced |
|---|---|---|---|
| off | 1 | 5,144 ms | — |
| 512 | 3 | 5,304 ms | +3.1% |
| 128 | 11 | 5,694 ms | +10.7% |

About **50 ms per extra `step()` call**, independent of context length — so the cost is the number of slices, not the
number of blocks. Two candidate causes were measured and **ruled out**:

- *per-layer sync* (the i915 watchdog `gpu::finish()`): batching it (every 4 layers) or removing it entirely changed
  nothing (5,058 / 5,201 ms vs 5,228 ms).
- *block capture*: skipping `export_block` changed nothing (5,700 ms with capture vs 5,704 ms without).

That leaves the fixed cost of starting a step — argument marshalling and ~400 kernel launches per slice on this driver.
The fixes are therefore fewer slices, in order of effort:

1. **Larger blocks.** `--block-cache 512` costs +3.1% instead of +10.7%. With the tip in place the tail stays exact,
   so the only thing a larger block trades away is how finely two conversations can share a diverging prefix.
2. **Capture boundary states inside one slice.** `op_lm_conv_state` already writes `b*x` of chosen rows into a state
   slot; a variant that also writes the two rows before every block boundary into a side buffer would let prefill run
   unsliced and blocks be exported in a post-pass, removing the cost entirely. This is the real fix; it touches a kernel.
3. Fewer launches per layer (fusion) — helps all prefill, not just the cache.

It also applies only to a *cold* request (`cache_prompt: false` skips alignment and capture) and only to the first turn
of a conversation. On the 1.2B the relative penalty is about half, because prefill is 5-7x slower per token there.

Warm-path check: after a restart, a 1,938-token prompt with a 768-token chain restored prefilled 1,170 tokens in
3,475 ms (337 tok/s), where a cold prefill of the full prompt is ~5,750 ms.

## Where this could go next

1. **Async prefetch**: the chain is matched before the slot is free, so the read could overlap the queue.
2. **A hot tier is deliberately absent.** On this box (iGPU, unified memory) "device cache" and "RAM cache" are the same
   physical memory, so a separate RAM tier is just a bigger `--prefix-cache-mb`; at 24 KB/token an exact 4k prefix costs
   96 MB, so the hot tier holds ~1 entry either way. The tip is what makes that small tier sufficient.
2. **Direct-to-device block writes**: apply blocks into the slot at their offsets instead of assembling a full host
   buffer (removes an `O(prefix)` copy; needs a `restore_range` on the engine).
3. **TTL by traffic class** and an eviction policy that prefers leaves: evicting a mid-chain block truncates every
   descendant. Correct (lookup stops at the first miss) but wasteful.
4. **A remote backend**: the same block format behind a `get/put/has` interface, so one cache node can serve several
   engines. Break-even is ~3.4 MB/s of link bandwidth to beat this box's prefill rate.
5. Choose N per workload: small N shares more finely but pays more slice overhead.

## Running it

```
./ferro-embed serve --lm models/LFM2.5-350M-Q8_0.gguf --snapshot-dir /path --block-cache 128
python3 scripts/test_blocks.py           # needs models/ and a built binary
```
`--block-cache 0` (default) keeps the whole-prefix snapshot path exactly as before.
`/health` exposes `block_cache: {block, entries, mb, saved, disk_hits}`; `/metrics` exposes four counters/gauges.
