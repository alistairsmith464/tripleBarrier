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
    enum LabelingType { Hard } labeling_type = Hard;
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
    }
};
