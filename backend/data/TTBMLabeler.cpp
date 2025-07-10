#include "TTBMLabeler.h"
#include "Constants.h"
#include <algorithm>
#include <cmath>

TTBMLabeler::TTBMLabeler(BarrierConfig::TTBMDecayType decay_type,
                         double lambda, double alpha, double beta)
    : decay_type_(decay_type), lambda_(lambda), alpha_(alpha), beta_(beta) {
    if (lambda <= 0.0) throw std::invalid_argument("lambda must be positive for exponential decay");
    if (alpha < 0.0 || alpha > 1.0) throw std::invalid_argument("alpha must be between 0 and 1 for linear decay");
    if (beta <= 0.0) throw std::invalid_argument("beta must be positive for hyperbolic decay");
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
        if (entry.volatility <= 0.0) continue;
        
        double pt = entry.price * (1.0 + profit_multiple * entry.volatility);
        double sl = entry.price * (1.0 - stop_multiple * entry.volatility);
        
        size_t end_idx = std::min(event_idx + size_t(vertical_barrier), data.size() - 1);
        
        int hard_label = 0;
        size_t exit_idx = end_idx;
        size_t barrier_hit_time = vertical_barrier;
        
        size_t profit_hit = SIZE_MAX;
        size_t stop_hit = SIZE_MAX;
        
        for (size_t i = event_idx + 1; i <= end_idx; ++i) {
            bool profit = data[i].price >= pt;
            bool stop = data[i].price <= sl;
            
            if (profit && profit_hit == SIZE_MAX) {
                profit_hit = i;
            }
            if (stop && stop_hit == SIZE_MAX) {
                stop_hit = i;
            }
            
            if (profit_hit != SIZE_MAX && stop_hit != SIZE_MAX) break;
        }
        
        if (profit_hit != SIZE_MAX && stop_hit != SIZE_MAX) {
            if (profit_hit < stop_hit) {
                hard_label = +1;
                exit_idx = profit_hit;
                barrier_hit_time = profit_hit - event_idx;
            } else if (stop_hit < profit_hit) {
                hard_label = -1;
                exit_idx = stop_hit;
                barrier_hit_time = stop_hit - event_idx;
            } else {
                hard_label = +1;
                exit_idx = profit_hit;
                barrier_hit_time = profit_hit - event_idx;
            }
        } else if (profit_hit != SIZE_MAX) {
            hard_label = +1;
            exit_idx = profit_hit;
            barrier_hit_time = profit_hit - event_idx;
        } else if (stop_hit != SIZE_MAX) {
            hard_label = -1;
            exit_idx = stop_hit;
            barrier_hit_time = stop_hit - event_idx;
        } else {
            hard_label = 0;
            exit_idx = end_idx;
            barrier_hit_time = vertical_barrier;
        }
          double time_elapsed_ratio = static_cast<double>(barrier_hit_time) / static_cast<double>(vertical_barrier);
        
        double decay_factor = applyDecay(time_elapsed_ratio);
        double ttbm_label = hard_label * decay_factor;

        int periods_to_exit = static_cast<int>(exit_idx - event_idx);
        
        if (exit_idx >= data.size()) {
            exit_idx = data.size() - 1;
        }
        
        results.push_back(LabeledEvent{
            entry.timestamp,
            data[exit_idx].timestamp,
            hard_label,
            entry.price,
            data[exit_idx].price,
            periods_to_exit,
            ttbm_label,
            time_elapsed_ratio,
            decay_factor,
            true,
            pt,
            sl,
            entry.volatility,
            data[exit_idx].price
        });
    }
    
    return results;
}

double TTBMLabeler::exponentialDecay(double time_ratio) const {
    return std::exp(-lambda_ * time_ratio);
}

double TTBMLabeler::linearDecay(double time_ratio) const {
    return std::max(0.0, 1.0 - alpha_ * time_ratio);
}

double TTBMLabeler::hyperbolicDecay(double time_ratio) const {
    if (std::abs(beta_ * time_ratio) > Constants::Validation::MAX_HYPERBOLIC_BETA_TIME) {
        return 0.0;
    }
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
            throw std::invalid_argument("Unknown decay type");
    }
}
