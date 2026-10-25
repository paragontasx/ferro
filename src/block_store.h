// block_store.h — block-granular prompt cache on local disk.
//
// A prompt prefix is cut into fixed blocks of `blk` tokens. Each block is stored once, content-addressed by a hash chain:
// block i's hash depends on block i-1's hash, so two conversations that share a system prompt share the physical blocks,
// and a lookup costs O(prefix/blk) hashes instead of a comparison against every stored prefix.
//
// A block holds the K/V rows of its own token range (per attention layer, with int8 scales) plus the conv state as it stood
// at the end of the block. That conv state is why blocks are only taken at block boundaries: it is a rolling 2-timestep
// value that cannot be reconstructed from the KV alone, so a chain may only be resumed where a block recorded it.
//
// Layout of a block's K/V payload is a prefix blob restricted to its own rows, so blocks of one chain concatenate (per
// layer) into exactly what LmEngine::import_prefix expects. `blob_bytes()` is the assembled size for a chain of k blocks.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ferro {

// Per-model block geometry: what the payload of one block looks like. Filled from the engine, validated on every load.
struct BlockGeom {
    uint32_t n_layers = 0, n_conv = 0;          // attention layers (have K/V), conv layers (have state)
    uint32_t row_bytes = 0;                     // K/V bytes per token: 2 * n_kv * head_dim * kvesz
    uint32_t scale_row_bytes = 0;               // int8 KV: fp32 scales per token, else 0
    uint32_t conv_layer_bytes = 0;              // 2 * n_embd * 4
};

class BlockStore {
public:
    struct Entry {
        std::vector<int> toks;                  // the stored prefix (a whole number of blocks)
        std::vector<uint64_t> chain;            // one hash per block, deepest last
        std::string path;
        int64_t mtime = 0;
        uint64_t bytes = 0;
    };
    ~BlockStore();
    // dir is created if needed; id = identity of (model, kernels, kv dtype); blk = tokens per block; budget_bytes bounds
    // the directory (least recently used blocks go first), 0 = unlimited
    bool open(const std::string& dir, uint64_t id, int blk, uint64_t budget_bytes, const BlockGeom& geom, std::string* err);
    int block() const { return blk_; }
    const BlockGeom& geom() const { return geom_; }
    size_t blob_bytes(int n_tokens) const;                          // assembled prefix blob for n_tokens (multiple of blk)
    // deepest stored prefix of `prompt` that is strictly shorter than `prompt` (false if none)
    bool longest(const std::vector<int>& prompt, Entry* out) const;
    bool load(const Entry& e, std::vector<uint8_t>* blob) const;     // assemble the chain; false if a block is gone/corrupt
    // store the block ending at `end` (a multiple of blk) of `prompt`; kv/cst come from LmEngine::export_block
    void save_async(const std::vector<int>& prompt, int end, std::vector<uint8_t>&& kv, std::vector<uint8_t>&& cst);
    void touch(const Entry& e);
    bool has_chain(uint64_t chain) const;
    size_t entries() const;
    uint64_t bytes() const;
    uint64_t saved() const { return n_saved_; }
    size_t loads() const { return n_loads_; }
    void flush();
    void close();

private:
    struct Rec { std::string path; uint64_t bytes = 0; int64_t mtime = 0; uint64_t used = 0; int32_t n_tokens = 0; };
    struct Job { std::vector<int> toks; int end = 0; uint64_t chain = 0, parent = 0; std::vector<uint8_t> kv, cst; };
    uint64_t chain_next(uint64_t parent, const int* toks, int n) const;
    std::string path_for(uint64_t chain) const;
    void writer_loop();
    void enforce_budget_locked(const std::string& keep);
    bool read_block(const std::string& path, uint64_t chain, std::vector<uint8_t>* kv, std::vector<uint8_t>* cst, std::vector<int>* toks) const;
    bool write_block(const Job& j, std::string* err);

    std::string dir_;
    uint64_t id_ = 0, budget_ = 0, root_ = 0, n_saved_ = 0, tick_ = 0;
    int blk_ = 0;
    BlockGeom geom_;
    mutable size_t n_loads_ = 0;
    mutable std::mutex m_;
    std::unordered_map<uint64_t, Rec> idx_;
    std::deque<Job> jobs_;
    std::condition_variable cv_, idle_cv_;
    bool stop_ = false, busy_ = false;
    std::thread th_;
};

}  // namespace ferro
