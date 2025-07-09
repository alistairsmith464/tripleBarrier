#include "FeatureExtractor.h"
#include "FeatureCalculator.h"
#include "DataCleaningUtils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>

std::map<std::string, std::string> FeatureExtractor::getFeatureMapping() {
    return {
        {"Close-to-close return for the previous day", FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D},
        {"Return over the past 5 days", FeatureCalculator::RETURN_5D},
        {"Return over the past 10 days", FeatureCalculator::RETURN_10D},
        {"Rolling standard deviation of daily returns over the last 5 days", FeatureCalculator::ROLLING_STD_5D},
        {"EWMA volatility over 10 days", FeatureCalculator::EWMA_VOL_10D},
        {"5-day simple moving average (SMA)", FeatureCalculator::SMA_5D},
        {"10-day SMA", FeatureCalculator::SMA_10D},
        {"20-day SMA", FeatureCalculator::SMA_20D},
        {"Distance between current close price and 5-day SMA", FeatureCalculator::DIST_TO_SMA_5D},
        {"Rate of Change (ROC) over 5 days", FeatureCalculator::ROC_5D},
        {"Relative Strength Index (RSI) over 14 days", FeatureCalculator::RSI_14D},
        {"5-day high minus 5-day low (price range)", FeatureCalculator::PRICE_RANGE_5D},
        {"Current close price relative to 5-day high", FeatureCalculator::CLOSE_OVER_HIGH_5D},
        {"Slope of linear regression of close prices over 10 days", FeatureCalculator::SLOPE_LR_10D},
        {"Day of the week", FeatureCalculator::DAY_OF_WEEK},
        {"Days since last event", FeatureCalculator::DAYS_SINCE_LAST_EVENT}
    };
}

FeatureExtractor::FeatureExtractionResult FeatureExtractor::extractFeaturesForClassification(
    const std::set<std::string>& selectedFeatures,
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents
) {
    FeatureExtractionResult result;
    
    // Convert selected features to backend IDs
    auto featureMap = getFeatureMapping();
    std::set<std::string> backendFeatures;
    for (const std::string& feat : selectedFeatures) {
        if (featureMap.count(feat)) {
            backendFeatures.insert(featureMap[feat]);
        }
    }
    
    // Extract price and timestamp data
    std::vector<double> prices;
    std::vector<std::string> timestamps;
    for (const auto& row : rows) {
        prices.push_back(row.price);
        timestamps.push_back(row.timestamp);
    }
    
    // Find event indices
    std::vector<int> eventIndices = findEventIndices(rows, labeledEvents);
    
    // Calculate features for each event
    for (size_t i = 0; i < eventIndices.size(); ++i) {
        auto features = FeatureCalculator::calculateFeatures(
            prices, timestamps, eventIndices, int(i), backendFeatures
        );
        
        result.features.push_back(features);
        result.labels.push_back(labeledEvents[i].label);
        result.returns.push_back(labeledEvents[i].exit_price - labeledEvents[i].entry_price);
    }
    
    return result;
}

FeatureExtractor::FeatureExtractionResult FeatureExtractor::extractFeaturesForRegression(
    const std::set<std::string>& selectedFeatures,
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents
) {
    FeatureExtractionResult result;
    
    // Convert selected features to backend IDs
    auto featureMap = getFeatureMapping();
    std::set<std::string> backendFeatures;
    for (const std::string& feat : selectedFeatures) {
        if (featureMap.count(feat)) {
            backendFeatures.insert(featureMap[feat]);
        }
    }
    
    // Extract price and timestamp data
    std::vector<double> prices;
    std::vector<std::string> timestamps;
    for (const auto& row : rows) {
        prices.push_back(row.price);
        timestamps.push_back(row.timestamp);
    }
    
    // Find event indices
    std::vector<int> eventIndices = findEventIndices(rows, labeledEvents);
    
    // Calculate features for each event
    for (size_t i = 0; i < eventIndices.size(); ++i) {
        auto baseFeatures = FeatureCalculator::calculateFeatures(
            prices, timestamps, eventIndices, int(i), backendFeatures
        );
        
        // Enhance features with additional calculations
        auto enhancedFeatures = enhanceFeatures(baseFeatures, rows[eventIndices[i]]);
        
        result.features.push_back(enhancedFeatures);
        result.labels_double.push_back(labeledEvents[i].ttbm_label);
        result.returns.push_back(labeledEvents[i].exit_price - labeledEvents[i].entry_price);
    }
    
    // Clean invalid values
    for (auto& featureRow : result.features) {
        for (auto& kv : featureRow) {
            if (std::isnan(kv.second) || std::isinf(kv.second)) {
                kv.second = 0.0;
            }
        }
    }
    
    // Apply robust scaling
    applyRobustScaling(result.features);
    
    // Debug output
    if (!result.labels_double.empty()) {
        double min_label = *std::min_element(result.labels_double.begin(), result.labels_double.end());
        double max_label = *std::max_element(result.labels_double.begin(), result.labels_double.end());
        double mean_label = std::accumulate(result.labels_double.begin(), result.labels_double.end(), 0.0) / result.labels_double.size();
        
        int zero_count = 0, positive_count = 0, negative_count = 0;
        for (double label : result.labels_double) {
            if (std::abs(label) < 0.01) zero_count++;
            else if (label > 0) positive_count++;
            else negative_count++;
        }
        
        std::cout << "[DEBUG] Regression: Predicting TTBM labels" << std::endl;
        std::cout << "  Sample size: " << result.labels_double.size() << std::endl;
        std::cout << "  Range: [" << min_label << ", " << max_label << "], Mean: " << mean_label << std::endl;
        std::cout << "  Positive: " << positive_count << " (" << (100.0*positive_count/result.labels_double.size()) << "%), "
                  << "Negative: " << negative_count << " (" << (100.0*negative_count/result.labels_double.size()) << "%), "
                  << "Zero: " << zero_count << " (" << (100.0*zero_count/result.labels_double.size()) << "%)" << std::endl;
    }
    
    return result;
}

std::vector<int> FeatureExtractor::findEventIndices(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents
) {
    std::vector<int> eventIndices;
    for (const auto& event : labeledEvents) {
        auto it = std::find_if(rows.begin(), rows.end(), 
            [&](const PreprocessedRow& r) { return r.timestamp == event.entry_time; });
        if (it != rows.end()) {
            eventIndices.push_back(int(std::distance(rows.begin(), it)));
        }
    }
    return eventIndices;
}

std::map<std::string, double> FeatureExtractor::enhanceFeatures(
    const std::map<std::string, double>& baseFeatures,
    const PreprocessedRow& row
) {
    std::map<std::string, double> enhanced = baseFeatures;
    
    // Volume-based features
    if (row.volume.has_value()) {
        double volume = row.volume.value();
        enhanced["volume"] = volume;
        
        if (baseFeatures.count(FeatureCalculator::RETURN_5D)) {
            enhanced["volume_return_5d"] = volume * baseFeatures.at(FeatureCalculator::RETURN_5D);
        }
        if (baseFeatures.count(FeatureCalculator::ROLLING_STD_5D)) {
            enhanced["volume_volatility_5d"] = volume * baseFeatures.at(FeatureCalculator::ROLLING_STD_5D);
        }
    }
    
    // Volatility-adjusted return
    if (baseFeatures.count(FeatureCalculator::RETURN_5D) && baseFeatures.count(FeatureCalculator::ROLLING_STD_5D)) {
        double return_5d = baseFeatures.at(FeatureCalculator::RETURN_5D);
        double vol_5d = baseFeatures.at(FeatureCalculator::ROLLING_STD_5D);
        if (vol_5d > 1e-10) {
            enhanced["volatility_adjusted_return_5d"] = return_5d / vol_5d;
        }
    }
    
    // Momentum-volatility ratio
    if (baseFeatures.count(FeatureCalculator::ROC_5D) && baseFeatures.count(FeatureCalculator::EWMA_VOL_10D)) {
        double roc_5d = baseFeatures.at(FeatureCalculator::ROC_5D);
        double vol_10d = baseFeatures.at(FeatureCalculator::EWMA_VOL_10D);
        enhanced["momentum_vol_ratio"] = roc_5d * vol_10d;
    }
    
    // SMA distance volatility-adjusted
    if (baseFeatures.count(FeatureCalculator::DIST_TO_SMA_5D) && baseFeatures.count(FeatureCalculator::ROLLING_STD_5D)) {
        double dist_sma = baseFeatures.at(FeatureCalculator::DIST_TO_SMA_5D);
        double vol_5d = baseFeatures.at(FeatureCalculator::ROLLING_STD_5D);
        if (vol_5d > 1e-10) {
            enhanced["sma_distance_vol_adj"] = dist_sma / vol_5d;
        }
    }
    
    // RSI momentum
    if (baseFeatures.count(FeatureCalculator::RSI_14D) && baseFeatures.count(FeatureCalculator::RETURN_5D)) {
        double rsi = baseFeatures.at(FeatureCalculator::RSI_14D);
        double return_5d = baseFeatures.at(FeatureCalculator::RETURN_5D);
        enhanced["rsi_momentum"] = (rsi - 50.0) * return_5d;
    }
    
    return enhanced;
}

void FeatureExtractor::applyRobustScaling(std::vector<std::map<std::string, double>>& features) {
    if (features.empty()) return;
    
    std::map<std::string, double> feature_medians;
    std::map<std::string, double> feature_iqrs;
    
    // Initialize feature names
    for (const auto& featureRow : features) {
        for (const auto& kv : featureRow) {
            if (feature_medians.find(kv.first) == feature_medians.end()) {
                feature_medians[kv.first] = 0.0;
                feature_iqrs[kv.first] = 0.0;
            }
        }
    }
    
    // Calculate medians and IQRs
    for (const auto& kv : feature_medians) {
        std::vector<double> values;
        for (const auto& featureRow : features) {
            auto it = featureRow.find(kv.first);
            if (it != featureRow.end()) {
                values.push_back(it->second);
            }
        }
        
        if (!values.empty()) {
            std::sort(values.begin(), values.end());
            size_t n = values.size();
            
            // Calculate median
            if (n % 2 == 0) {
                feature_medians[kv.first] = (values[n/2-1] + values[n/2]) / 2.0;
            } else {
                feature_medians[kv.first] = values[n/2];
            }
            
            // Calculate IQR
            double q1 = values[n/4];
            double q3 = values[3*n/4];
            feature_iqrs[kv.first] = q3 - q1;
            if (feature_iqrs[kv.first] < 1e-10) feature_iqrs[kv.first] = 1.0;
        }
    }
    
    // Apply scaling
    for (auto& featureRow : features) {
        for (auto& kv : featureRow) {
            kv.second = (kv.second - feature_medians[kv.first]) / feature_iqrs[kv.first];
        }
    }
    
    std::cout << "[DEBUG] Applied robust scaling (median/IQR)" << std::endl;
}
