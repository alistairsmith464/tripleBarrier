#pragma once
#include "IBarrierLabeler.h"
#include "PreprocessedRow.h"
#include "LabeledEvent.h"
#include "BarrierConfig.h"
#include <vector>

class TTBMLabeler : public IBarrierLabeler {
public:
    // Constructor taking decay configuration
    TTBMLabeler(BarrierConfig::TTBMDecayType decay_type = BarrierConfig::Exponential,
                double lambda = 1.0, double alpha = 0.5, double beta = 1.0);
    
    std::vector<LabeledEvent> label(
        const std::vector<PreprocessedRow>& data,
        const std::vector<size_t>& event_indices,
        double profit_multiple,
        double stop_multiple,
        int vertical_barrier
    ) const override;

private:
    BarrierConfig::TTBMDecayType decay_type_;
    double lambda_;  // Exponential decay rate
    double alpha_;   // Linear decay factor
    double beta_;    // Hyperbolic decay steepness
    
    // Decay functions
    double exponentialDecay(double time_ratio) const;
    double linearDecay(double time_ratio) const;
    double hyperbolicDecay(double time_ratio) const;
    
    // Apply the configured decay function
    double applyDecay(double time_ratio) const;
};
