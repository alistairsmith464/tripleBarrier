#include "CUSUMFilter.h"
#include <cmath>

std::vector<size_t> CUSUMFilter::detect(const std::vector<double>& prices, const std::vector<double>& volatility, double threshold) {
    std::vector<size_t> events;
    if (prices.size() < 2 || prices.size() != volatility.size()) return events;
    if (threshold <= 0) return events;
    double s_pos = 0.0, s_neg = 0.0;
    for (size_t i = 1; i < prices.size(); ++i) {
        double diff = prices[i] - prices[i-1];
        double scaled = (volatility[i] > 0) ? diff / volatility[i] : 0.0;
        s_pos = std::max(0.0, s_pos + scaled);
        s_neg = std::min(0.0, s_neg + scaled);
        if (s_pos > threshold) {
            events.push_back(i);
            s_pos = 0.0;
            s_neg = 0.0;
        } else if (s_neg < -threshold) {
            events.push_back(i);
            s_pos = 0.0;
            s_neg = 0.0;
        }
    }
    return events;
}
