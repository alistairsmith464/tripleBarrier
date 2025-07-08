#pragma once
#include <string>

struct LabeledEvent {
    std::string entry_time;
    std::string exit_time;
    int label;
    double entry_price;
    double exit_price;
    int periods_to_exit = 0;
    
    // TTBM (Time-to-Barrier Modification) fields
    double ttbm_label = 0.0;  // Continuous label in [-1, +1] incorporating time decay
    double time_to_barrier_ratio = 1.0;  // t_b / t_v (normalized time to first barrier)
    double decay_factor = 1.0;  // The decay function value applied
};
