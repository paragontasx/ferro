// threadpool.cpp
#include "threadpool.h"
#include <sched.h>
#include <immintrin.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>

namespace ferro {

static constexpr int kSpinLimit = 40000;

// flush denormals to zero (FTZ|DAZ): softmax tails over long sequences otherwise produce denormal
// FMAs that cost ~100x on x86 (measured: 4.5 min -> seconds for 2x2.2k-token attention). MXCSR is per thread.
static inline void set_ftz_daz() { _mm_setcsr(_mm_getcsr() | 0x8040); }

// Idle workers park on the generation counter, which doubles as a futex word (the build is C++17: no std::atomic::wait). They used
// to spin and then sched_yield() forever, which kept one core busy for the whole life of the process even with nothing to do -
// measured on an idle `serve`: 3.9 h of CPU in 3.8 h of uptime, 62% of it in the kernel - inside the N100's shared CPU+iGPU budget.
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "the futex word must be a plain 32-bit integer");
static inline void futex_wait(std::atomic<uint32_t>* w, uint32_t expected) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(w), FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
}
static inline void futex_wake_all(std::atomic<uint32_t>* w) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(w), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}
// How long a worker keeps spinning after its last task before it parks: back-to-back parallel_for calls of a CPU forward pass
// stay on the spin path and never pay a wake-up. FERRO_POOL_SPIN_US overrides (0 = park at once).
static int spin_us() {
    static const int us = [] { const char* e = getenv("FERRO_POOL_SPIN_US"); return e ? std::max(0, atoi(e)) : 200; }();
    return us;
}

void Pool::wait_work(uint32_t seen) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto budget = std::chrono::microseconds(spin_us());
    for (unsigned i = 1; gen_.load(std::memory_order_acquire) == seen && !quit_.load(std::memory_order_relaxed); ++i) {
        _mm_pause();
        if ((i & 63) == 0 && std::chrono::steady_clock::now() - t0 >= budget) break;
    }
    while (gen_.load(std::memory_order_acquire) == seen && !quit_.load(std::memory_order_acquire)) {
        // announce, then re-check: pairs with parallel_for's seq_cst gen_ increment + sleepers_ load, so a dispatch is either seen
        // here or sees this sleeper and wakes it (no lost wake-up); futex_wait itself returns at once when gen_ has moved on
        sleepers_.fetch_add(1, std::memory_order_seq_cst);
        if (gen_.load(std::memory_order_seq_cst) == seen && !quit_.load(std::memory_order_seq_cst)) futex_wait(&gen_, seen);
        sleepers_.fetch_sub(1, std::memory_order_relaxed);
    }
}

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
            uint32_t seen = 0;
            for (;;) {
                wait_work(seen);                                  // spin briefly, then sleep until the next parallel_for or shutdown
                if (quit_.load(std::memory_order_acquire)) break;
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
    quit_.store(true, std::memory_order_seq_cst);
    gen_.fetch_add(1, std::memory_order_seq_cst);
    futex_wake_all(&gen_);                                        // unconditional: a worker may be about to park
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
    gen_.fetch_add(1, std::memory_order_seq_cst);                 // publishes the task (release) and orders it before the sleepers_ read
    if (sleepers_.load(std::memory_order_seq_cst) > 0) futex_wake_all(&gen_);   // only parked workers need the syscall
    // main thread also steals chunks
    for (;;) {
        int c = next_.fetch_add(1, std::memory_order_relaxed);
        if (c >= nchunks) break;
        const int lo = (int)((long long)c * chunk_), hi = (int)std::min<long long>((long long)lo + chunk_, n);
        fn(lo, hi);
    }
    // wait for workers (bounded by the work just handed out)
    int spins = 0;
    while (remaining_.load(std::memory_order_acquire) > 0) {
        if (++spins < kSpinLimit) _mm_pause();
        else std::this_thread::yield();
    }
}

} // namespace ferro
