#pragma once
#include <stdexcept>

struct BarrierConfig {
    double profit_multiple;
    double stop_multiple;
    int vertical_window;

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
    }
};
