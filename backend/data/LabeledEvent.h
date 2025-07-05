#pragma once
#include <string>

struct LabeledEvent {
    std::string entry_time;
    std::string exit_time;
    int label; // +1, -1, 0
    double entry_price;
    double exit_price;
    double soft_label = 0.0; // For probabilistic labeling
};
