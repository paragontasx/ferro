// block_store.h — block-granular prompt cache on local disk, with an exact per-conversation tip.
//
// A prompt prefix is cut into fixed blocks of `blk` tokens. Each block is stored once, content-addressed by a hash chain:
// block i's key depends on block i-1's key, so two conversations that share a system prompt share the physical blocks and
// a hit can only be served for a prefix that really matches.
//
// A block holds the K/V rows of its own token range (per attention layer, with int8 scales) and the conv state as it stood
// at the end of that range. The conv state is a rolling 2-timestep value that cannot be reconstructed from K/V, so a chain
// may only be resumed where a block recorded it — which is why blocks are only taken at block boundaries.
//
// The tip closes the gap that leaves: the tokens between the last whole block and the exact end of a conversation (prompt
// plus reply) are stored as one variable-length block, replaced on every turn. Without it a resumed conversation re-prefills
// up to blk-1 tokens; with it, a continuation turn re-prefills only what the client actually added.
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
        std::vector<int> toks;                  // the hit prefix, including the tip
        std::vector<uint64_t> chain;            // one key per whole block, deepest last
        int tip_len = 0;                        // tokens covered by the tip past the last whole block (0 = none)
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
    size_t blob_bytes(int n_tokens) const;                          // assembled prefix blob for n_tokens
    // deepest stored prefix of `prompt` strictly shorter than `prompt`; extends past the last whole block when a tip matches
    bool longest(const std::vector<int>& prompt, Entry* out) const;
    bool load(const Entry& e, std::vector<uint8_t>* blob) const;     // assemble chain + tip; false if a piece is gone/corrupt
    // store the block ending at `end` (a multiple of blk) of `prompt`; kv/cst come from LmEngine::export_block
    void save_async(const std::vector<int>& prompt, int end, std::vector<uint8_t>&& kv, std::vector<uint8_t>&& cst);
    // store the exact end of `prompt` from `start` (the last whole block that actually exists) to `end`
    void save_tip_async(const std::vector<int>& prompt, int start, int end, std::vector<uint8_t>&& kv, std::vector<uint8_t>&& cst);
    // how many whole blocks of `prompt` are stored (multiple of blk); the depth a tip must hang off
    int block_depth(const std::vector<int>& prompt) const;
    void touch(const Entry& e);
    bool has_chain(uint64_t chain) const;
    size_t entries() const;
    size_t tips() const;
    uint64_t bytes() const;
    uint64_t blocks_bytes() const;
    uint64_t saved() const { return n_saved_; }
    size_t loads() const { return n_loads_; }
    // Eviction policy: leaves go first, so a shared prefix is never orphaned while its own descendants still have data to
    // serve; anything left hanging off an evicted block is reclaimed immediately instead of occupying the budget as dead
    // weight. TTLs are per class (0 = never): a tip is the tail of one conversation and is superseded every turn, so it is
    // the class worth expiring first.
    void set_ttl(int64_t block_ttl_s, int64_t tip_ttl_s) { ttl_s_ = block_ttl_s; tip_ttl_s_ = tip_ttl_s; }
    uint64_t evicted() const { std::lock_guard<std::mutex> g(m_); return n_evicted_; }
    uint64_t expired() const { std::lock_guard<std::mutex> g(m_); return n_expired_; }
    uint64_t orphans() const { std::lock_guard<std::mutex> g(m_); return n_orphans_; }
    size_t dangling() const;
    std::string link_dump() const;                  // debug: "chain<-parent, ..." for the first few blocks                        // blocks whose parent is gone: must always be 0 after enforcement
    void flush();
    void close();

private:
    struct Rec { std::string path; uint64_t bytes = 0; int64_t mtime = 0; uint64_t used = 0; int32_t n_tokens = 0;
                 uint64_t parent = 0;               // chain this block hangs off (root_ when it is first)
                 int64_t last_used = 0; };          // wall clock of the last touch: the TTL's clock;
    struct Tip : Rec { uint64_t hash = 0; };
    struct Job { std::vector<int> toks; int start = 0, end = 0; uint64_t chain = 0, parent = 0; bool tip = false;
                 std::vector<uint8_t> kv, cst; };
    uint64_t chain_next(uint64_t parent, const int* toks, int n) const;
    uint64_t chain_at(const std::vector<int>& prompt, int end) const;   // key of the last whole block before `end`
    std::string path_for(uint64_t hash) const;
    void writer_loop();
    void enforce_budget_locked(const std::string& keep);
    void sweep_orphans_locked(uint64_t* total);
    void evict_locked(std::unordered_map<uint64_t, Rec>::iterator it, uint64_t* total);
    bool read_block(const std::string& path, uint64_t chain, std::vector<uint8_t>* kv, std::vector<uint8_t>* cst, std::vector<int>* toks) const;
    bool write_block(const Job& j, std::string* err);

    std::string dir_;
    uint64_t id_ = 0, budget_ = 0, root_ = 0, n_saved_ = 0, tick_ = 0;
    int blk_ = 0;
    BlockGeom geom_;
    mutable size_t n_loads_ = 0;
    mutable std::mutex m_;
    std::unordered_map<uint64_t, Rec> idx_;          // whole blocks, by chain
    std::unordered_map<uint64_t, Tip> tips_;         // parent chain -> the exact end of that conversation
    uint64_t n_evicted_ = 0, n_expired_ = 0, n_orphans_ = 0;
    int64_t ttl_s_ = 0, tip_ttl_s_ = 0;
    std::deque<Job> jobs_;
    std::condition_variable cv_, idle_cv_;
    bool stop_ = false, busy_ = false;
    std::thread th_;
};

}  // namespace ferro
