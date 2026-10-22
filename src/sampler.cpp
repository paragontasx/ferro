// sampler.cpp - see sampler.h
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace ferro {

int Sampler::sample(float* logits, int n, const std::vector<int>& history) {
    if (p_.repetition_penalty != 1.0f && p_.repeat_last_n != 0 && !history.empty()) {
        std::unordered_set<int> seen;
        size_t b = (p_.repeat_last_n > 0 && (size_t)p_.repeat_last_n < history.size()) ? history.size() - (size_t)p_.repeat_last_n : 0;
        if (p_.penalty_from > 0) b = std::max(b, (size_t)p_.penalty_from);   // scope "generated": prompt tokens are never penalized
        for (size_t i = b; i < history.size(); ++i) seen.insert(history[i]);
        for (int id : seen)
            if (id >= 0 && id < n) logits[id] = logits[id] > 0 ? logits[id] / p_.repetition_penalty : logits[id] * p_.repetition_penalty;
    }
    if (p_.temperature <= 0.0f || p_.top_k == 1) {                       // greedy
        int best = 0;
        for (int i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    }
    // candidates: top-k by logit (all tokens when top_k == 0)
    cand_.clear();
    const int k = (p_.top_k > 0 && p_.top_k < n) ? p_.top_k : n;
    if (k < n) {
        cand_.reserve((size_t)n);
        for (int i = 0; i < n; ++i) cand_.emplace_back(logits[i], i);
        std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        cand_.resize((size_t)k);
    } else {
        for (int i = 0; i < n; ++i) cand_.emplace_back(logits[i], i);
        std::sort(cand_.begin(), cand_.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    }
    // softmax at temperature (candidates are sorted by logit, descending)
    const float mx = cand_[0].first;
    double sum = 0;
    std::vector<double> pr(cand_.size());
    for (size_t i = 0; i < cand_.size(); ++i) { pr[i] = std::exp((double)(cand_[i].first - mx) / p_.temperature); sum += pr[i]; }
    for (double& v : pr) v /= sum;
    size_t keep = cand_.size();
    if (p_.min_p > 0.0f) { const double thr = pr[0] * p_.min_p; while (keep > 1 && pr[keep - 1] < thr) --keep; }
    if (p_.top_p < 1.0f) { double c = 0; size_t i = 0; for (; i < keep; ++i) { c += pr[i]; if (c >= p_.top_p) { ++i; break; } } keep = std::max<size_t>(1, std::min(keep, i)); }
    double tot = 0; for (size_t i = 0; i < keep; ++i) tot += pr[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng_), c = 0;
    for (size_t i = 0; i < keep; ++i) { c += pr[i]; if (r <= c) return cand_[i].second; }
    return cand_[keep - 1].second;
}

}  // namespace ferro
