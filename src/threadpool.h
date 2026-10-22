// threadpool.h — pinned pool tuned for 4 small cores: workers spin briefly after each parallel_for, then park on a futex
#pragma once
#include <atomic>
#include <cstdint>
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
    void wait_work(uint32_t seen);          // spin up to FERRO_POOL_SPIN_US (default 200), then park until gen_ != seen or quit_
    std::vector<Worker> workers_;
    std::atomic<int> next_{0};
    std::atomic<int> remaining_{0};
    std::atomic<bool> quit_{false};
    std::atomic<uint32_t> gen_{0};          // generation counter; also the futex word parked workers sleep on
    std::atomic<int> sleepers_{0};          // parked workers: parallel_for issues the wake-up syscall only when > 0
    const std::function<void(int, int)>* task_ = nullptr;
    int total_ = 0, chunk_ = 1, nchunks_ = 0;
    std::thread::id owner_{};
};

} // namespace ferro
