#pragma once
#include <stdexcept>

struct BarrierConfig {
    double profit_multiple;
    double stop_multiple;
    int vertical_window;
    // CUSUM options
    bool use_cusum = false;
    double cusum_threshold = 5.0;
    // Labeling type
    enum LabelingType { Hard, Probabilistic } labeling_type = Hard;
    void validate() const {
        if (stop_multiple > 1.0) {
            throw std::invalid_argument("BarrierConfig: stop_multiple cannot be greater than 1.0");
        }
        if (profit_multiple <= 0.0) {
            throw std::invalid_argument("BarrierConfig: profit_multiple must be positive");
        }
        if (stop_multiple < 0.0) {
            throw std::invalid_argument("BarrierConfig: stop_multiple must be non-negative");
        }
        if (vertical_window <= 0) {
            throw std::invalid_argument("BarrierConfig: vertical_window must be positive");
        }
        if (use_cusum && cusum_threshold <= 0.0) {
            throw std::invalid_argument("BarrierConfig: cusum_threshold must be positive");
        }
    }
};
