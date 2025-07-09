#pragma once
#include <vector>
#include <cmath>

namespace VolatilityCalculator {
    std::vector<double> rollingStdDev(const std::vector<double>& logReturns, int window) {
        std::vector<double> result(logReturns.size(), std::nan("") );

        if (window <= 1 || logReturns.size() < window) return result;

        for (size_t i = window - 1; i < logReturns.size(); ++i) {
            double sum = 0, sum2 = 0;

            for (int j = int(i) - window + 1; j <= int(i); ++j) {
                sum += logReturns[j];
                sum2 += logReturns[j] * logReturns[j];
            }

            double mean = sum / window;
            double var = (sum2 / window) - (mean * mean);
            result[i] = std::sqrt(std::max(0.0, var));
        }
        
        return result;
    }
}
