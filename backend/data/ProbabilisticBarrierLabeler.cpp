#include "ProbabilisticBarrierLabeler.h"
#include <cmath>
#include <algorithm>

namespace {
    // Sigmoid function centered at barrier, with steepness parameter
    double sigmoid(double x, double center, double steepness) {
        return 1.0 / (1.0 + std::exp(-steepness * (x - center)));
    }
}

std::vector<LabeledEvent> ProbabilisticBarrierLabeler::label(
    const std::vector<PreprocessedRow>& data,
    const std::vector<size_t>& event_indices,
    double profit_multiple,
    double stop_multiple,
    int vertical_barrier
) const {
    constexpr double STEEPNESS = 5.0; // Could be user-configurable
    std::vector<LabeledEvent> results;
    for (size_t event_idx : event_indices) {
        if (event_idx >= data.size()) continue;
        const auto& entry = data[event_idx];
        double pt = entry.price + profit_multiple * entry.volatility;
        double sl = entry.price - stop_multiple * entry.volatility;
        size_t end_idx = std::min(event_idx + size_t(vertical_barrier), data.size() - 1);
        double prob_up = 0.0;
        double prob_down = 0.0;
        int steps = 0;
        for (size_t i = event_idx + 1; i <= end_idx; ++i) {
            double p_up = sigmoid(data[i].price, pt, STEEPNESS);
            double p_down = 1.0 - sigmoid(data[i].price, sl, STEEPNESS);
            prob_up += p_up;
            prob_down += p_down;
            ++steps;
        }
        // Aggregate: mean probability over window
        prob_up /= std::max(1, steps);
        prob_down /= std::max(1, steps);
        double prob_time = 1.0 - prob_up - prob_down;
        prob_time = std::max(0.0, prob_time); // ensure non-negative
        // Normalize
        double sum = prob_up + prob_down + prob_time;
        if (sum > 0) {
            prob_up /= sum;
            prob_down /= sum;
            prob_time /= sum;
        }
        // Compute soft label in [-1, 1]
        double soft_label = prob_up - prob_down;
        soft_label = std::max(-1.0, std::min(1.0, soft_label));
        // Store soft label in LabeledEvent (add field if needed)
        results.push_back(LabeledEvent{
            entry.timestamp,
            data[end_idx].timestamp,
            0, // hard label not used
            entry.price,
            data[end_idx].price,
            soft_label // requires LabeledEvent to have this field
        });
    }
    return results;
}
