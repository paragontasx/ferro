// threadpool.h — pinned spin-pool tuned for 4 small cores
#pragma once
#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace ferro {

class Pool {
public:
    void init(int nthreads);
    static Pool& get();
    // calls fn(lo, hi) partitioned across pool threads; caller participates
    void parallel_for(int n, const std::function<void(int, int)>& fn);
    void shutdown();
    int threads() const { return (int)workers_.size() + 1; }

private:
    struct Worker { std::thread th; };
    std::vector<Worker> workers_;
    std::atomic<int> next_{0};
    std::atomic<int> remaining_{0};
    std::atomic<bool> quit_{false};
    std::atomic<uint64_t> gen_{0};
    const std::function<void(int, int)>* task_ = nullptr;
    int total_ = 0, chunk_ = 1, nchunks_ = 0;
    std::thread::id owner_{};
};

} // namespace ferro
