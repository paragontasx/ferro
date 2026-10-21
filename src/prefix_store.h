// prefix_store.h — on-disk prompt-prefix snapshots for the LM (survive restarts, spill target for the device cache).
// One file per snapshot: header (magic, version, identity, token count, payload size, CRC) + tokens + the engine's state blob.
// The identity hashes everything the state depends on (model file, KV dtype, kernel build): a snapshot from anything else is never loaded.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ferro {

class PrefixStore {
public:
    struct Entry { std::string path; std::vector<int> toks; int64_t mtime = 0; uint64_t bytes = 0; };
    ~PrefixStore();
    // dir is created if needed; id = identity of (model, kernels, kv dtype); budget_bytes bounds the directory (oldest files go first), 0 = unlimited
    bool open(const std::string& dir, uint64_t id, uint64_t budget_bytes, std::string* err);
    // longest stored token prefix that is strictly shorter than `prompt` (null if none)
    bool longest(const std::vector<int>& prompt, Entry* out) const;                   // copy: the writer thread may change the index
    bool has(const std::vector<int>& toks) const;
    void save_async(const std::vector<int>& toks, std::vector<uint8_t>&& blob);       // atomic (tmp + rename) write on a background thread
    bool load(const Entry& e, std::vector<uint8_t>* blob);                            // verifies identity + CRC; a corrupt or foreign file is deleted
    void touch(const Entry& e);                                                       // LRU: a used snapshot is the last to be evicted
    size_t entries() const;
    uint64_t bytes() const;
    void flush();                                                                     // wait for queued writes
    uint64_t saved() const { return n_saved_; }

private:
    struct Job { std::vector<int> toks; std::vector<uint8_t> blob; };
    void writer_loop();
    void enforce_budget_locked(const std::string& keep);
    static std::string name_for(uint64_t id, const std::vector<int>& toks);
    std::string dir_; uint64_t id_ = 0, budget_ = 0, n_saved_ = 0;
    mutable std::mutex m_;
    std::vector<Entry> entries_;
    std::deque<Job> jobs_; std::condition_variable cv_, idle_cv_; bool stop_ = false, busy_ = false; std::thread th_;
};

}  // namespace ferro
