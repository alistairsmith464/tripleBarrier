#include "HardBarrierLabeler.h"
#include "OverlapPurger.h"
#include <algorithm>
#include <cmath>
#include <iostream>

std::vector<LabeledEvent> HardBarrierLabeler::label(
    const std::vector<PreprocessedRow>& data,
    const std::vector<size_t>& event_indices,
    double profit_multiple,
    double stop_multiple,
    int vertical_barrier
) const {
    std::vector<LabeledEvent> results;
    
    std::vector<size_t> purged_indices = OverlapPurger::purgeOverlappingEvents(event_indices, vertical_barrier);
    
    std::cout << "[INFO] Barrier labeling with overlap prevention:" << std::endl;
    std::cout << "  - Original events: " << event_indices.size() << std::endl;
    std::cout << "  - Purged events: " << purged_indices.size() << std::endl;
    std::cout << "  - Removed overlaps: " << (event_indices.size() - purged_indices.size()) << std::endl;
    
    for (size_t event_idx : purged_indices) {
        if (event_idx >= data.size()) continue;
        const auto& entry = data[event_idx];
        
        double pt = entry.price * (1.0 + profit_multiple * entry.volatility);
        double sl = entry.price * (1.0 - stop_multiple * entry.volatility);
        size_t end_idx = std::min(event_idx + size_t(vertical_barrier), data.size() - 1);
        int label = 0;
        size_t exit_idx = end_idx;
        size_t profit_hit = data.size();
        size_t stop_hit = data.size();
        size_t profit_hit_time = data.size();
        size_t stop_hit_time = data.size();
        
        for (size_t i = event_idx + 1; i <= end_idx; ++i) {
            bool profit = data[i].price >= pt;
            bool stop = data[i].price <= sl;
            if (profit && stop) {
                profit_hit = i;
                stop_hit = i;
                profit_hit_time = i - event_idx;
                stop_hit_time = i - event_idx;
                break;
            }
            if (profit && profit_hit == data.size()) {
                profit_hit = i;
                profit_hit_time = i - event_idx;
            }
            if (stop && stop_hit == data.size()) {
                stop_hit = i;
                stop_hit_time = i - event_idx;
            }
            if (profit_hit != data.size() && stop_hit != data.size()) break;
        }
        if (profit_hit < stop_hit) {
            label = +1;
            exit_idx = profit_hit;
        } else if (stop_hit < profit_hit) {
            label = -1;
            exit_idx = stop_hit;
        } else if (profit_hit == stop_hit && profit_hit != data.size()) {
            label = +1;
            exit_idx = profit_hit;
        } else {
            label = 0;
            exit_idx = end_idx;
        }
        
        int periods_to_exit = static_cast<int>(exit_idx - event_idx);
        
        results.push_back(LabeledEvent{
            entry.timestamp,
            data[exit_idx].timestamp,
            label,
            entry.price,
            data[exit_idx].price,
            periods_to_exit,
            0.0,
            static_cast<double>(periods_to_exit) / static_cast<double>(vertical_barrier),
            1.0,
            false,
            pt,
            sl,
            entry.volatility,
            data[exit_idx].price
        });
    }
    return results;
}
