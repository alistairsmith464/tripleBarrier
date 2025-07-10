#pragma once
#include <vector>
#include <cstddef>

class CUSUMFilter {
public:
    static std::vector<size_t> detect(const std::vector<double>& prices, const std::vector<double>& volatility, double threshold);
    static std::vector<size_t> detectWithGap(const std::vector<double>& prices, 
                                            const std::vector<double>& volatility, 
                                            double threshold, int min_gap);

private:
    static std::vector<size_t> enforceMinimumGap(const std::vector<size_t>& events, int min_gap);
};
