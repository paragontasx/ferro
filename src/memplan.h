// memplan.h — memory planning and admission for `ferro-embed serve` / `ferro-embed plan`.
// Device buffers on an iGPU are ordinary system RAM that cannot be swapped out, so everything resident counts against the container limit.
#pragma once
#include <string>
#include <vector>

namespace ferro {

struct ServerOptions;

struct PlanLine { std::string what; double mb = 0; std::string note; };
struct MemPlan {
    std::vector<PlanLine> lines;
    double need_mb = 0, limit_mb = 0;          // limit_mb 0 = no container limit found
    bool fits = true;
    std::vector<std::string> advice;           // concrete ways to fit (when it does not)
};

double cgroup_limit_mb();                      // memory.max (cgroup v2) or v1 limit; FERRO_MEM_LIMIT_MB overrides (tests); 0 = unlimited / unknown
double cgroup_used_mb();                       // anon + shmem of the container (what device buffers and heap count as)
// What the configuration needs, from GGUF headers and file sizes only (nothing is loaded):
//   ~700 MB runtime + 1.3 x every GGUF (weights are re-laid-out on the device) + the LM's KV cache + the prompt-prefix cache at its budget.
// Calibrated on measured anon+shmem (350M + d1 + EG2 + mmproj: measured 3373 MB, estimated 3457 MB).
MemPlan plan_memory(const ServerOptions& o);
void print_plan(const MemPlan& p, FILE* out);
// Transient memory a request with media needs while it is processed (MB). 0 for text-only bodies. Images are measured (header dimensions);
// audio / video are conservative estimates from the payload size.
double media_cost_mb(const std::string& request_body);

}  // namespace ferro
