#pragma once
#include <string>
#include <optional>

struct DataRow {
    std::string timestamp;
    double price;
    std::optional<double> open;
    std::optional<double> high;
    std::optional<double> low;
    std::optional<double> close;
    std::optional<double> volume;
};
