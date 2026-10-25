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
