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
        for (size_t i = event_idx + 1; i <= end_idx; ++i) {
            if (data[i].price >= pt) {
                label = +1;
                exit_idx = i;
                break;
            } else if (data[i].price <= sl) {
                label = -1;
                exit_idx = i;
                break;
            }
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
