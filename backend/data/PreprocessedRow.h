#pragma once
#include <string>
#include <optional>

struct PreprocessedRow {
    std::string timestamp;
    double price;
    std::optional<double> open, high, low, close, volume;
    double log_return = 0.0;
    double volatility = 0.0;
    bool is_event = false;
};
