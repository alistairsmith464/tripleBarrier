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
    double time_elapsed_ratio = 0.0;  // Fraction of time elapsed before barrier hit (t_b / t_v)
    double decay_factor = 1.0;  // The decay function value applied
    
    // Barrier type information
    bool is_ttbm = false;  // true if TTBM labeling was used, false for hard labeling
    
    // Additional barrier information
    double profit_barrier = 0.0;  // The calculated profit barrier level
    double stop_barrier = 0.0;    // The calculated stop barrier level
    double entry_volatility = 0.0; // Volatility at event entry time
    double trigger_price = 0.0;    // Actual price that triggered the barrier exit
};
