#pragma once
#include <stdexcept>

struct BarrierConfig {
    double profit_multiple;
    double stop_multiple;
    int vertical_window;

    bool use_cusum = false;
    double cusum_threshold = 5.0;
    enum LabelingType { Hard, TTBM } labeling_type = Hard;
    
    enum TTBMDecayType { Exponential, Linear, Hyperbolic } ttbm_decay_type = Exponential;
    
    double ttbm_lambda = 0.3;
    double ttbm_alpha = 0.2;
    double ttbm_beta = 0.5;
    
    void validate() const {
        if (profit_multiple <= 0.0) {
            throw std::invalid_argument("BarrierConfig: profit_multiple must be positive");
        }
        if (stop_multiple <= 0.0) {
            throw std::invalid_argument("BarrierConfig: stop_multiple must be positive");
        }
        if (vertical_window <= 0) {
            throw std::invalid_argument("BarrierConfig: vertical_window must be positive");
        }
        if (use_cusum && cusum_threshold <= 0.0) {
            throw std::invalid_argument("BarrierConfig: cusum_threshold must be positive");
        }
        if (labeling_type == TTBM) {
            if (ttbm_lambda <= 0.0) {
                throw std::invalid_argument("BarrierConfig: ttbm_lambda must be positive (fixed at 2.0)");
            }
            if (ttbm_alpha < 0.0 || ttbm_alpha > 1.0) {
                throw std::invalid_argument("BarrierConfig: ttbm_alpha must be in [0, 1] (fixed at 0.8)");
            }
            if (ttbm_beta <= 0.0) {
                throw std::invalid_argument("BarrierConfig: ttbm_beta must be positive (fixed at 3.0)");
            }
        }
    }
};
