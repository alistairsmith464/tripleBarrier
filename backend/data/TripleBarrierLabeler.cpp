#include "TripleBarrierLabeler.h"
#include <algorithm>
#include <cmath>

std::vector<LabeledEvent> TripleBarrierLabeler::label(
    const std::vector<PreprocessedRow>& data,
    const std::vector<size_t>& event_indices,
    double profit_multiple,
    double stop_multiple,
    int vertical_barrier
) {
    std::vector<LabeledEvent> results;
    for (size_t event_idx : event_indices) {
        if (event_idx >= data.size()) continue;
        const auto& entry = data[event_idx];
        double pt = entry.price + profit_multiple * entry.volatility;
        double sl = entry.price - stop_multiple * entry.volatility;
        size_t end_idx = std::min(event_idx + size_t(vertical_barrier), data.size() - 1);
        int label = 0;
        size_t exit_idx = end_idx;
        size_t profit_hit = data.size();
        size_t stop_hit = data.size();
        for (size_t i = event_idx + 1; i <= end_idx; ++i) {
            bool profit = data[i].price >= pt;
            bool stop = data[i].price <= sl;
            if (profit && stop) {
                // Both hit on same bar, profit takes precedence
                profit_hit = i;
                stop_hit = i;
                break;
            }
            if (profit && profit_hit == data.size()) profit_hit = i;
            if (stop && stop_hit == data.size()) stop_hit = i;
            if (profit_hit != data.size() && stop_hit != data.size()) break;
        }
        if (profit_hit < stop_hit) {
            label = +1;
            exit_idx = profit_hit;
        } else if (stop_hit < profit_hit) {
            label = -1;
            exit_idx = stop_hit;
        } else if (profit_hit == stop_hit && profit_hit != data.size()) {
            // Both hit on same bar, profit takes precedence
            label = +1;
            exit_idx = profit_hit;
        } else {
            label = 0;
            exit_idx = end_idx;
        }
        results.push_back(LabeledEvent{
            entry.timestamp,
            data[exit_idx].timestamp,
            label,
            entry.price,
            data[exit_idx].price
        });
    }
    return results;
}
