#pragma once
#include <vector>
#include <cstddef>

class CUSUMFilter {
public:
    // Returns indices of detected events
    static std::vector<size_t> detect(const std::vector<double>& prices, const std::vector<double>& volatility, double threshold);
};
