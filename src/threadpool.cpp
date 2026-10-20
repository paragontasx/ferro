// threadpool.cpp
#include "threadpool.h"
#include <sched.h>
#include <immintrin.h>
#include <cstdlib>
#include <algorithm>

namespace ferro {

static constexpr int kSpinLimit = 40000;

// flush denormals to zero (FTZ|DAZ): softmax tails over long sequences otherwise produce denormal
// FMAs that cost ~100x on x86 (measured: 4.5 min -> seconds for 2x2.2k-token attention). MXCSR is per thread.
static inline void set_ftz_daz() { _mm_setcsr(_mm_getcsr() | 0x8040); }

void Pool::init(int nthreads) {
    set_ftz_daz();
    if (!workers_.empty()) return;
    owner_ = std::this_thread::get_id();
    if (nthreads < 1) nthreads = 1;
    // main thread participates in every parallel_for: pin it to core 0 so it
    // cannot drift onto a worker's core or wake idle cores (power clock win).
    if (getenv("FERRO_PINMAIN") && nthreads <= (int)std::thread::hardware_concurrency()) {
        cpu_set_t s0; CPU_ZERO(&s0); CPU_SET(0, &s0);
        pthread_setaffinity_np(pthread_self(), sizeof(s0), &s0);
    }
    for (int i = 1; i < nthreads; ++i) {
        workers_.push_back({std::thread([this, i] {
            cpu_set_t set; CPU_ZERO(&set); CPU_SET(i % std::max(1u, std::thread::hardware_concurrency()), &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
            set_ftz_daz();
            uint64_t seen = 0;
            while (!quit_.load(std::memory_order_relaxed)) {
                // spin for the next generation
                int spins = 0;
                while (gen_.load(std::memory_order_acquire) == seen && !quit_.load(std::memory_order_relaxed)) {
                    if (++spins < kSpinLimit) _mm_pause();
                    else { std::this_thread::yield(); }
                }
                if (quit_.load(std::memory_order_relaxed)) break;
                seen = gen_.load(std::memory_order_acquire);
                // work-steal chunks
                // chunk index bound = nchunks_ (comparing it with the item count made c * chunk_ overflow int for
                // large fan-outs: a negative lo ran the task on out-of-range rows)
                for (;;) {
                    const int c = next_.fetch_add(1, std::memory_order_relaxed);
                    if (c >= nchunks_) break;
                    const int lo = (int)std::min<long long>((long long)c * chunk_, total_);
                    const int hi = (int)std::min<long long>((long long)lo + chunk_, total_);
                    if (lo < hi) (*task_)(lo, hi);
                }
                remaining_.fetch_sub(1, std::memory_order_release);
            }
        })});
    }
}

Pool& Pool::get() { static Pool p; return p; }

void Pool::shutdown() {
    quit_.store(true, std::memory_order_release);
    gen_.fetch_add(1, std::memory_order_release);
    for (auto& w : workers_) if (w.th.joinable()) w.th.join();
    workers_.clear();
}

void Pool::parallel_for(int n, const std::function<void(int, int)>& fn) {
    int nw = (int)workers_.size();
    // not re-entrant: only the owner thread fans out; any other thread (e.g. the GPU driver) runs inline
    if (nw == 0 || n <= 1 || std::this_thread::get_id() != owner_) { fn(0, n); return; }
    // chunk sizing: aim for >=2 chunks per worker for balance, but no chunk < 4 rows
    int chunk = std::max(1, std::min((n + (nw + 1) * 2 - 1) / ((nw + 1) * 2), n / (nw + 1)));
    if (chunk < 1) chunk = 1;
    total_ = n; chunk_ = chunk;
    task_ = &fn;
    const int nchunks = (int)(((long long)n + chunk - 1) / chunk);
    nchunks_ = nchunks;
    remaining_.store(nw, std::memory_order_relaxed);
    next_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_release);
    // main thread also steals chunks
    for (;;) {
        int c = next_.fetch_add(1, std::memory_order_relaxed);
        if (c >= nchunks) break;
        const int lo = (int)((long long)c * chunk_), hi = (int)std::min<long long>((long long)lo + chunk_, n);
        fn(lo, hi);
    }
    // wait for workers
    int spins = 0;
    while (remaining_.load(std::memory_order_acquire) > 0) {
        if (++spins < kSpinLimit) _mm_pause();
        else std::this_thread::yield();
    }
}

} // namespace ferro
