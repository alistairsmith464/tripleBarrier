#pragma once
#include <string>

struct LabeledEvent {
    std::string entry_time;
    std::string exit_time;
    int label;
    double entry_price;
    double exit_price;
    int periods_to_exit = 0;
};
