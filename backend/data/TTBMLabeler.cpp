#include "TTBMLabeler.h"
#include <algorithm>
#include <cmath>

TTBMLabeler::TTBMLabeler(BarrierConfig::TTBMDecayType decay_type,
                         double lambda, double alpha, double beta)
    : decay_type_(decay_type), lambda_(lambda), alpha_(alpha), beta_(beta) {
}

std::vector<LabeledEvent> TTBMLabeler::label(
    const std::vector<PreprocessedRow>& data,
    const std::vector<size_t>& event_indices,
    double profit_multiple,
    double stop_multiple,
    int vertical_barrier
) const {
    std::vector<LabeledEvent> results;
    
    for (size_t event_idx : event_indices) {
        if (event_idx >= data.size()) continue;
        
        const auto& entry = data[event_idx];
        double pt = entry.price * (1.0 + profit_multiple * entry.volatility);
        double sl = entry.price * (1.0 - stop_multiple * entry.volatility);
        size_t end_idx = std::min(event_idx + size_t(vertical_barrier), data.size() - 1);
        
        // Track first barrier breach
        int hard_label = 0;
        size_t exit_idx = end_idx;
        size_t barrier_hit_time = vertical_barrier;  // Default to max time if no barrier hit
        
        size_t profit_hit = data.size();
        size_t stop_hit = data.size();
        
        // Find first barrier breach
        for (size_t i = event_idx + 1; i <= end_idx; ++i) {
            bool profit = data[i].price >= pt;
            bool stop = data[i].price <= sl;
            
            if (profit && stop) {
                // Both barriers hit simultaneously - prefer profit
                profit_hit = i;
                stop_hit = i;
                break;
            }
            if (profit && profit_hit == data.size()) {
                profit_hit = i;
            }
            if (stop && stop_hit == data.size()) {
                stop_hit = i;
            }
            if (profit_hit != data.size() && stop_hit != data.size()) break;
        }
        
        // Determine which barrier was hit first and when
        if (profit_hit < stop_hit) {
            hard_label = +1;
            exit_idx = profit_hit;
            barrier_hit_time = profit_hit - event_idx;
        } else if (stop_hit < profit_hit) {
            hard_label = -1;
            exit_idx = stop_hit;
            barrier_hit_time = stop_hit - event_idx;
        } else if (profit_hit == stop_hit && profit_hit != data.size()) {
            hard_label = +1;
            exit_idx = profit_hit;
            barrier_hit_time = profit_hit - event_idx;
        } else {
            // Vertical barrier hit (no horizontal barrier breached)
            hard_label = 0;
            exit_idx = end_idx;
            barrier_hit_time = vertical_barrier;
        }
        
        // Calculate TTBM label
        double time_ratio = static_cast<double>(barrier_hit_time) / static_cast<double>(vertical_barrier);
        time_ratio = std::min(1.0, time_ratio);  // Ensure it doesn't exceed 1.0
        
        double decay_factor = applyDecay(time_ratio);
        double ttbm_label = hard_label * decay_factor;
        
        // Ensure TTBM label is within [-1, +1]
        ttbm_label = std::max(-1.0, std::min(1.0, ttbm_label));
        
        int periods_to_exit = static_cast<int>(exit_idx - event_idx);
        
        results.push_back(LabeledEvent{
            entry.timestamp,
            data[exit_idx].timestamp,
            hard_label,
            entry.price,
            data[exit_idx].price,
            periods_to_exit,
            ttbm_label,
            time_ratio,
            decay_factor
        });
    }
    
    return results;
}

double TTBMLabeler::exponentialDecay(double time_ratio) const {
    // f(t_b, t_v) = e^(-λ * t_b / t_v)
    return std::exp(-lambda_ * time_ratio);
}

double TTBMLabeler::linearDecay(double time_ratio) const {
    // f(t_b, t_v) = 1 - α * t_b / t_v
    return std::max(0.0, 1.0 - alpha_ * time_ratio);
}

double TTBMLabeler::hyperbolicDecay(double time_ratio) const {
    // f(t_b, t_v) = 1 / (1 + β * t_b / t_v)
    return 1.0 / (1.0 + beta_ * time_ratio);
}

double TTBMLabeler::applyDecay(double time_ratio) const {
    switch (decay_type_) {
        case BarrierConfig::Exponential:
            return exponentialDecay(time_ratio);
        case BarrierConfig::Linear:
            return linearDecay(time_ratio);
        case BarrierConfig::Hyperbolic:
            return hyperbolicDecay(time_ratio);
        default:
            return exponentialDecay(time_ratio);
    }
}
