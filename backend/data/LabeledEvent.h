#pragma once
#include <string>

struct LabeledEvent {
    std::string entry_time;
    std::string exit_time;
    int label;
    double entry_price;
    double exit_price;
    int periods_to_exit = 0;
    
    double ttbm_label = 0.0;
    double time_elapsed_ratio = 0.0; 
    double decay_factor = 1.0; 
    
    bool is_ttbm = false;
    
    double profit_barrier = 0.0;
    double stop_barrier = 0.0;   
    double entry_volatility = 0.0;
    double trigger_price = 0.0;
};
