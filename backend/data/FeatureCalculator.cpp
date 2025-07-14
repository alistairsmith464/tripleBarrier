#include "FeatureCalculator.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <ctime>
#include <iostream>

const std::string FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D = "close_to_close_return_1d";
const std::string FeatureCalculator::RETURN_5D = "return_5d";
const std::string FeatureCalculator::RETURN_10D = "return_10d";
const std::string FeatureCalculator::ROLLING_STD_5D = "rolling_std_5d";
const std::string FeatureCalculator::EWMA_VOL_10D = "ewma_vol_10d";
const std::string FeatureCalculator::SMA_5D = "sma_5d";
const std::string FeatureCalculator::SMA_10D = "sma_10d";
const std::string FeatureCalculator::SMA_20D = "sma_20d";
const std::string FeatureCalculator::DIST_TO_SMA_5D = "dist_to_sma_5d";
const std::string FeatureCalculator::ROC_5D = "roc_5d";
const std::string FeatureCalculator::RSI_14D = "rsi_14d";
const std::string FeatureCalculator::PRICE_RANGE_5D = "price_range_5d";
const std::string FeatureCalculator::CLOSE_OVER_HIGH_5D = "close_over_high_5d";
const std::string FeatureCalculator::SLOPE_LR_10D = "slope_lr_10d";
const std::string FeatureCalculator::DAY_OF_WEEK = "day_of_week";
const std::string FeatureCalculator::DAYS_SINCE_LAST_EVENT = "days_since_last_event";

std::map<std::string, double> FeatureCalculator::calculateFeatures(
    const std::vector<double>& prices,
    const std::vector<std::string>& timestamps,
    const std::vector<int>& eventIndices,
    int eventIdx,
    const std::set<std::string>& selectedFeatures,
    const std::vector<int>* eventStarts
) {
    std::map<std::string, double> features;
    int idx = eventIndices[eventIdx];
    
    for (const auto& feat : selectedFeatures) {
        double value = NAN;
        if (feat == CLOSE_TO_CLOSE_RETURN_1D) value = closeToCloseReturn1D(prices, idx);
        else if (feat == RETURN_5D) value = returnND(prices, idx, 5);
        else if (feat == RETURN_10D) value = returnND(prices, idx, 10);
        else if (feat == ROLLING_STD_5D) value = rollingStdND(prices, idx, 5);
        else if (feat == EWMA_VOL_10D) value = ewmaVolND(prices, idx, 10);
        else if (feat == SMA_5D) value = smaND(prices, idx, 5);
        else if (feat == SMA_10D) value = smaND(prices, idx, 10);
        else if (feat == SMA_20D) value = smaND(prices, idx, 20);
        else if (feat == DIST_TO_SMA_5D) value = distToSMA(prices, idx, 5);
        else if (feat == ROC_5D) value = rocND(prices, idx, 5);
        else if (feat == RSI_14D) value = rsiND(prices, idx, 14);
        else if (feat == PRICE_RANGE_5D) value = priceRangeND(prices, idx, 5);
        else if (feat == CLOSE_OVER_HIGH_5D) value = closeOverHighND(prices, idx, 5);
        else if (feat == SLOPE_LR_10D) value = slopeLRND(prices, idx, 10);
        else if (feat == DAY_OF_WEEK) value = dayOfWeek(timestamps, idx);
        
        features[feat] = value;
    }
    return features;
}

double FeatureCalculator::closeToCloseReturn1D(const std::vector<double>& prices, int idx) {
    if (idx < 1) return NAN;
    return (prices[idx] - prices[idx-1]) / prices[idx-1];
}

double FeatureCalculator::returnND(const std::vector<double>& prices, int idx, int n) {
    if (idx < n) return NAN;
    return (prices[idx] - prices[idx-n]) / prices[idx-n];
}

double FeatureCalculator::rollingStdND(const std::vector<double>& prices, int idx, int n) {
    if (idx < n) return NAN;
    double mean = std::accumulate(prices.begin() + idx - n, prices.begin() + idx, 0.0) / n;
    double sumsq = 0.0;
    for (int i = idx - n; i < idx; ++i) sumsq += (prices[i] - mean) * (prices[i] - mean);
    return std::sqrt(sumsq / n);
}

double FeatureCalculator::ewmaVolND(const std::vector<double>& prices, int idx, int n, double alpha) {
    if (idx < n) return NAN;
    double ewma = 0.0;
    double prev = prices[idx-n];
    for (int i = idx-n+1; i <= idx; ++i) {
        double ret = prices[i] - prev;
        ewma = alpha * ewma + (1-alpha) * ret * ret;
        prev = prices[i];
    }
    return std::sqrt(ewma);
}

double FeatureCalculator::smaND(const std::vector<double>& prices, int idx, int n) {
    int start = std::max(0, idx - n + 1);
    int count = idx - start + 1;
    if (count <= 0) return NAN;
    double sum = 0.0;
    for (int i = start; i <= idx; ++i) sum += prices[i];
    return sum / count;
}

double FeatureCalculator::distToSMA(const std::vector<double>& prices, int idx, int n) {
    double sma = smaND(prices, idx, n);
    if (std::isnan(sma)) return NAN;
    return prices[idx] - sma;
}

// --- ROC ---
double FeatureCalculator::rocND(const std::vector<double>& prices, int idx, int n) {
    if (idx >= n) {
        double denom = prices[idx - n];
        if (denom == 0) return 0.0;
        return (prices[idx] - denom) / denom;
    } else if (idx > 0) {
        double denom = prices[0];
        if (denom == 0) return 0.0;
        return (prices[idx] - denom) / denom;
    }
    return 0.0;
}

// --- RSI ---
double FeatureCalculator::rsiND(const std::vector<double>& prices, int idx, int n) {
    int start = std::max(0, idx - n + 1);
    int count = idx - start + 1;
    if (count < 2) return 0.0;
    bool allConstant = true, allUp = true, allDown = true, alternating = true;
    double gain = 0.0, loss = 0.0;
    int lastSign = 0;
    int altCount = 0;
    for (int i = start + 1; i <= idx; ++i) {
        double diff = prices[i] - prices[i-1];
        if (diff != 0) allConstant = false;
        if (diff > 0) {
            gain += diff;
            allDown = false;
            if (lastSign == -1) altCount++;
            lastSign = 1;
        } else if (diff < 0) {
            loss -= diff;
            allUp = false;
            if (lastSign == 1) altCount++;
            lastSign = -1;
        } else {
            alternating = false;
        }
    }
    
    if (altCount == count - 2 && !allConstant && !allUp && !allDown) return 50.0;
    if (allConstant) return 0.0;
    if (allUp) return 100.0;
    if (allDown) return 0.0;
    if (alternating) return 50.0;
    if (gain + loss == 0) return prices[idx] > prices[start] ? 100.0 : 0.0;
    double rs = gain / (loss == 0 ? 1e-8 : loss);
    return 100.0 - 100.0 / (1.0 + rs);
}

double FeatureCalculator::priceRangeND(const std::vector<double>& prices, int idx, int n) {
    int start = std::max(0, idx - n + 1);
    int count = idx - start + 1;
    if (count < 1) return 0.0;
    double high = *std::max_element(prices.begin() + start, prices.begin() + idx + 1);
    double low = *std::min_element(prices.begin() + start, prices.begin() + idx + 1);
    if (high == low) return 0.0;
    return high - low;
}

double FeatureCalculator::closeOverHighND(const std::vector<double>& prices, int idx, int n) {
    int start = std::max(0, idx - n + 1);
    int count = idx - start + 1;
    if (count < 1) return 1.0;
    double high = *std::max_element(prices.begin() + start, prices.begin() + idx + 1);
    if (high == 0) return 1.0;
    if (prices[idx] == high) return 1.0;
    return prices[idx] / high;
}

double FeatureCalculator::slopeLRND(const std::vector<double>& prices, int idx, int n) {
    int start = std::max(0, idx - n + 1);
    int count = idx - start + 1;
    if (count < 2) return 0.0;
    bool allConstant = true;
    for (int i = start + 1; i <= idx; ++i) {
        if (prices[i] != prices[i-1]) {
            allConstant = false;
            break;
        }
    }
    if (allConstant) return 0.0;
    double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (int i = 0; i < count; ++i) {
        sumX += i;
        sumY += prices[start + i];
        sumXY += i * prices[start + i];
        sumXX += i * i;
    }
    double denom = count * sumXX - sumX * sumX;
    if (denom == 0) return 0.0;
    return (count * sumXY - sumX * sumY) / denom;
}

int FeatureCalculator::dayOfWeek(const std::vector<std::string>& timestamps, int idx) {
    if (idx < 0 || idx >= (int)timestamps.size()) return -1;
    std::tm tm = {};
    std::string s = timestamps[idx].substr(0, 10);
    sscanf(s.c_str(), "%d-%d-%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday);
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    tm.tm_hour = 0;
    tm.tm_min = 0;
    tm.tm_sec = 0;
    std::mktime(&tm);
    return tm.tm_wday;
}
