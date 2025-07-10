#include "CUSUMFilter.h"
#include <cmath>
#include <algorithm>

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

std::vector<size_t> CUSUMFilter::detectWithGap(const std::vector<double>& prices, 
                                               const std::vector<double>& volatility, 
                                               double threshold, int min_gap) {
    std::vector<size_t> events = detect(prices, volatility, threshold);
    return enforceMinimumGap(events, min_gap);
}

std::vector<size_t> CUSUMFilter::enforceMinimumGap(const std::vector<size_t>& events, int min_gap) {
    if (events.empty() || min_gap <= 0) return events;
    
    std::vector<size_t> filtered;
    filtered.reserve(events.size());
    
    for (size_t event : events) {
        bool valid = true;
        
        for (size_t existing : filtered) {
            if (abs(static_cast<int>(event) - static_cast<int>(existing)) < min_gap) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            filtered.push_back(event);
        }
    }
    
    return filtered;
}
