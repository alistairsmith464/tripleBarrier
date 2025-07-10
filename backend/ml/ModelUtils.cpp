#include "ModelUtils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <set>
#include <iostream>

namespace MLPipeline {

std::vector<std::vector<float>> 
ModelUtils::toFloatMatrix(const std::vector<std::map<std::string, double>>& X, bool validate_input) {
    if (validate_input && X.empty()) {
        throw std::invalid_argument("Input matrix cannot be empty");
    }
    
    std::vector<std::vector<float>> result;
    result.reserve(X.size());
    
    for (const auto& row : X) {
        std::vector<float> float_row;
        float_row.reserve(row.size());
        
        for (const auto& [key, value] : row) {
            if (validate_input && (std::isnan(value) || std::isinf(value))) {
                throw std::invalid_argument("Input contains NaN or Inf values");
            }
            float_row.push_back(static_cast<float>(value));
        }
        result.push_back(std::move(float_row));
    }
    
    return result;
}

std::vector<float> 
ModelUtils::toFloatVecInt(const std::vector<int>& y, bool validate_input) {
    if (validate_input && y.empty()) {
        throw std::invalid_argument("Input vector cannot be empty");
    }
    
    std::vector<float> result;
    result.reserve(y.size());
    std::transform(y.begin(), y.end(), std::back_inserter(result),
                   [](int val) { return static_cast<float>(val); });
    
    return result;
}

std::vector<float> 
ModelUtils::toFloatVecDouble(const std::vector<double>& y, bool validate_input) {
    if (validate_input && y.empty()) {
        throw std::invalid_argument("Input vector cannot be empty");
    }
    
    std::vector<float> result;
    result.reserve(y.size());
    
    for (double val : y) {
        if (validate_input && (std::isnan(val) || std::isinf(val))) {
            throw std::invalid_argument("Input contains NaN or Inf values");
        }
        result.push_back(static_cast<float>(val));
    }
    
    return result;
}

// Feature preprocessing implementation
ModelUtils::PreprocessingResult ModelUtils::preprocessFeatures(
    const std::vector<std::map<std::string, double>>& X,
    const PreprocessingConfig& config) {
    
    PreprocessingResult result;
    result.processed_data = X;
    
    if (X.empty()) {
        return result;
    }
    
    // Collect all feature names
    std::set<std::string> all_features;
    for (const auto& sample : X) {
        for (const auto& [key, value] : sample) {
            all_features.insert(key);
        }
    }
    
    result.feature_names.assign(all_features.begin(), all_features.end());
    
    // Remove constant features
    if (config.remove_constant_features) {
        auto constant_features = selectFeaturesByVariance(X, 1e-10);
        for (const auto& feature : result.feature_names) {
            if (std::find(constant_features.begin(), constant_features.end(), feature) == constant_features.end()) {
                result.removed_features.push_back(feature);
            }
        }
        
        // Remove constant features from data
        for (auto& sample : result.processed_data) {
            for (const auto& removed_feature : result.removed_features) {
                sample.erase(removed_feature);
            }
        }
        
        // Update feature names
        result.feature_names = constant_features;
    }
    
    // Remove highly correlated features
    if (config.remove_correlated_features) {
        auto low_corr_features = selectFeaturesByCorrelation(result.processed_data, config.correlation_threshold);
        
        std::vector<std::string> corr_removed_features;
        for (const auto& feature : result.feature_names) {
            if (std::find(low_corr_features.begin(), low_corr_features.end(), feature) == low_corr_features.end()) {
                corr_removed_features.push_back(feature);
            }
        }
        
        result.removed_features.insert(result.removed_features.end(), 
                                     corr_removed_features.begin(), corr_removed_features.end());
        
        // Remove correlated features from data
        for (auto& sample : result.processed_data) {
            for (const auto& removed_feature : corr_removed_features) {
                sample.erase(removed_feature);
            }
        }
        
        // Update feature names
        result.feature_names = low_corr_features;
    }
    
    // Apply scaling
    if (config.scale_features) {
        for (const auto& feature_name : result.feature_names) {
            auto feature_values = extractFeature(result.processed_data, feature_name);
            
            double mean, scale_param;
            if (config.scaling_method == "standard") {
                mean = std::accumulate(feature_values.begin(), feature_values.end(), 0.0) / feature_values.size();
                double variance = 0.0;
                for (double val : feature_values) {
                    variance += (val - mean) * (val - mean);
                }
                variance /= feature_values.size();
                scale_param = std::sqrt(variance);
            } else if (config.scaling_method == "minmax") {
                auto [min_it, max_it] = std::minmax_element(feature_values.begin(), feature_values.end());
                mean = *min_it;
                scale_param = *max_it - *min_it;
            }
            
            result.scaling_params[feature_name] = {mean, scale_param};
            
            // Apply scaling to data
            if (scale_param > 1e-10) { // Avoid division by zero
                for (auto& sample : result.processed_data) {
                    auto it = sample.find(feature_name);
                    if (it != sample.end()) {
                        if (config.scaling_method == "standard") {
                            it->second = (it->second - mean) / scale_param;
                        } else if (config.scaling_method == "minmax") {
                            it->second = (it->second - mean) / scale_param;
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

std::vector<std::map<std::string, double>> ModelUtils::applyPreprocessing(
    const std::vector<std::map<std::string, double>>& X,
    const PreprocessingResult& preprocessing_info) {
    
    std::vector<std::map<std::string, double>> result = X;
    
    // Remove features that were removed during preprocessing
    for (auto& sample : result) {
        for (const auto& removed_feature : preprocessing_info.removed_features) {
            sample.erase(removed_feature);
        }
    }
    
    // Apply scaling
    for (auto& sample : result) {
        for (auto& [feature_name, value] : sample) {
            auto scaling_it = preprocessing_info.scaling_params.find(feature_name);
            if (scaling_it != preprocessing_info.scaling_params.end()) {
                double mean = scaling_it->second.first;
                double scale = scaling_it->second.second;
                
                if (scale > 1e-10) { // Avoid division by zero
                    value = (value - mean) / scale;
                }
            }
        }
    }
    
    return result;
}

// Feature engineering implementation
std::vector<std::map<std::string, double>> ModelUtils::createPolynomialFeatures(
    const std::vector<std::map<std::string, double>>& X, int degree) {
    
    if (degree < 1) {
        throw std::invalid_argument("Polynomial degree must be at least 1");
    }
    
    std::vector<std::map<std::string, double>> result = X;
    
    if (degree == 1) {
        return result;
    }
    
    // Collect all feature names
    std::set<std::string> feature_names;
    for (const auto& sample : X) {
        for (const auto& [key, value] : sample) {
            feature_names.insert(key);
        }
    }
    
    // Add polynomial features
    for (int d = 2; d <= degree; ++d) {
        for (auto& sample : result) {
            for (const auto& feature_name : feature_names) {
                auto it = sample.find(feature_name);
                if (it != sample.end()) {
                    std::string poly_name = feature_name + "_poly" + std::to_string(d);
                    sample[poly_name] = std::pow(it->second, d);
                }
            }
        }
    }
    
    return result;
}

std::vector<std::map<std::string, double>> ModelUtils::createInteractionFeatures(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<std::pair<std::string, std::string>>& interactions) {
    
    std::vector<std::map<std::string, double>> result = X;
    
    if (interactions.empty()) {
        // Create all pairwise interactions
        std::set<std::string> feature_names;
        for (const auto& sample : X) {
            for (const auto& [key, value] : sample) {
                feature_names.insert(key);
            }
        }
        
        for (auto& sample : result) {
            for (auto it1 = feature_names.begin(); it1 != feature_names.end(); ++it1) {
                for (auto it2 = std::next(it1); it2 != feature_names.end(); ++it2) {
                    auto val1_it = sample.find(*it1);
                    auto val2_it = sample.find(*it2);
                    
                    if (val1_it != sample.end() && val2_it != sample.end()) {
                        std::string interaction_name = *it1 + "_x_" + *it2;
                        sample[interaction_name] = val1_it->second * val2_it->second;
                    }
                }
            }
        }
    } else {
        // Create specified interactions
        for (auto& sample : result) {
            for (const auto& [feature1, feature2] : interactions) {
                auto val1_it = sample.find(feature1);
                auto val2_it = sample.find(feature2);
                
                if (val1_it != sample.end() && val2_it != sample.end()) {
                    std::string interaction_name = feature1 + "_x_" + feature2;
                    sample[interaction_name] = val1_it->second * val2_it->second;
                }
            }
        }
    }
    
    return result;
}

// Feature selection implementation
std::vector<std::string> ModelUtils::selectFeaturesByVariance(
    const std::vector<std::map<std::string, double>>& X,
    double variance_threshold) {
    
    std::vector<std::string> selected_features;
    
    if (X.empty()) {
        return selected_features;
    }
    
    // Collect all feature names
    std::set<std::string> all_features;
    for (const auto& sample : X) {
        for (const auto& [key, value] : sample) {
            all_features.insert(key);
        }
    }
    
    // Calculate variance for each feature
    for (const auto& feature_name : all_features) {
        auto feature_values = extractFeature(X, feature_name);
        double variance = calculateVariance(feature_values);
        
        if (variance > variance_threshold) {
            selected_features.push_back(feature_name);
        }
    }
    
    return selected_features;
}

std::vector<std::string> ModelUtils::selectFeaturesByCorrelation(
    const std::vector<std::map<std::string, double>>& X,
    double correlation_threshold) {
    
    std::vector<std::string> selected_features;
    
    if (X.empty()) {
        return selected_features;
    }
    
    // Collect all feature names
    std::set<std::string> all_features;
    for (const auto& sample : X) {
        for (const auto& [key, value] : sample) {
            all_features.insert(key);
        }
    }
    
    std::vector<std::string> feature_list(all_features.begin(), all_features.end());
    std::vector<bool> keep_feature(feature_list.size(), true);
    
    // Check correlations between all pairs
    for (size_t i = 0; i < feature_list.size(); ++i) {
        if (!keep_feature[i]) continue;
        
        auto feature_i_values = extractFeature(X, feature_list[i]);
        
        for (size_t j = i + 1; j < feature_list.size(); ++j) {
            if (!keep_feature[j]) continue;
            
            auto feature_j_values = extractFeature(X, feature_list[j]);
            double correlation = std::abs(calculateCorrelation(feature_i_values, feature_j_values));
            
            if (correlation > correlation_threshold) {
                // Keep the feature with higher variance
                double var_i = calculateVariance(feature_i_values);
                double var_j = calculateVariance(feature_j_values);
                
                if (var_i >= var_j) {
                    keep_feature[j] = false;
                } else {
                    keep_feature[i] = false;
                    break; // Don't check more features against feature i
                }
            }
        }
    }
    
    // Collect selected features
    for (size_t i = 0; i < feature_list.size(); ++i) {
        if (keep_feature[i]) {
            selected_features.push_back(feature_list[i]);
        }
    }
    
    return selected_features;
}

// Model validation implementation
bool ModelUtils::validateModelInputs(const std::vector<std::vector<float>>& X,
                                    const std::vector<float>& y) {
    if (X.empty() || y.empty()) {
        throw std::invalid_argument("Input data cannot be empty");
    }
    
    if (X.size() != y.size()) {
        throw std::invalid_argument("Features and targets must have the same number of samples");
    }
    
    // Check for consistent feature dimensions
    if (!X.empty()) {
        size_t n_features = X[0].size();
        for (size_t i = 1; i < X.size(); ++i) {
            if (X[i].size() != n_features) {
                throw std::invalid_argument("All samples must have the same number of features");
            }
        }
    }
    
    // Check for NaN/Inf values
    for (const auto& sample : X) {
        for (float value : sample) {
            if (std::isnan(value) || std::isinf(value)) {
                throw std::invalid_argument("Input contains NaN or Inf values");
            }
        }
    }
    
    for (float value : y) {
        if (std::isnan(value) || std::isinf(value)) {
            throw std::invalid_argument("Target contains NaN or Inf values");
        }
    }
    
    return true;
}

void ModelUtils::checkDataConsistency(const std::vector<std::map<std::string, double>>& X_train,
                                     const std::vector<std::map<std::string, double>>& X_test) {
    if (X_train.empty() || X_test.empty()) {
        throw std::invalid_argument("Training and test data cannot be empty");
    }
    
    // Collect feature names from both datasets
    std::set<std::string> train_features, test_features;
    
    for (const auto& sample : X_train) {
        for (const auto& [key, value] : sample) {
            train_features.insert(key);
        }
    }
    
    for (const auto& sample : X_test) {
        for (const auto& [key, value] : sample) {
            test_features.insert(key);
        }
    }
    
    // Check for missing features
    std::vector<std::string> missing_in_test, missing_in_train;
    
    std::set_difference(train_features.begin(), train_features.end(),
                       test_features.begin(), test_features.end(),
                       std::back_inserter(missing_in_test));
    
    std::set_difference(test_features.begin(), test_features.end(),
                       train_features.begin(), train_features.end(),
                       std::back_inserter(missing_in_train));
    
    if (!missing_in_test.empty()) {
        std::string error_msg = "Features missing in test data: ";
        for (const auto& feature : missing_in_test) {
            error_msg += feature + " ";
        }
        throw std::invalid_argument(error_msg);
    }
    
    if (!missing_in_train.empty()) {
        std::string error_msg = "Features missing in training data: ";
        for (const auto& feature : missing_in_train) {
            error_msg += feature + " ";
        }
        std::invalid_argument(error_msg);
    }
}

// Feature importance implementation
std::vector<ModelUtils::FeatureImportance> ModelUtils::computeFeatureImportance(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<double>& y,
    const std::string& method) {
    
    std::vector<FeatureImportance> importance_scores;
    
    if (X.empty() || y.empty() || X.size() != y.size()) {
        return importance_scores;
    }
    
    // Collect all feature names
    std::set<std::string> all_features;
    for (const auto& sample : X) {
        for (const auto& [key, value] : sample) {
            all_features.insert(key);
        }
    }
    
    if (method == "correlation") {
        size_t rank = 1;
        for (const auto& feature_name : all_features) {
            auto feature_values = extractFeature(X, feature_name);
            double correlation = std::abs(calculateCorrelation(feature_values, y));
            
            FeatureImportance importance;
            importance.feature_name = feature_name;
            importance.importance_score = correlation;
            importance.rank = rank++;
            
            importance_scores.push_back(importance);
        }
        
        // Sort by importance score (descending)
        std::sort(importance_scores.begin(), importance_scores.end(),
                 [](const FeatureImportance& a, const FeatureImportance& b) {
                     return a.importance_score > b.importance_score;
                 });
        
        // Update ranks
        for (size_t i = 0; i < importance_scores.size(); ++i) {
            importance_scores[i].rank = i + 1;
        }
    }
    
    return importance_scores;
}

// Data quality analysis implementation
ModelUtils::DataQualityReport ModelUtils::analyzeDataQuality(
    const std::vector<std::map<std::string, double>>& X) {
    
    DataQualityReport report;
    
    if (X.empty()) {
        return report;
    }
    
    // Collect all feature names
    std::set<std::string> all_features;
    for (const auto& sample : X) {
        for (const auto& [key, value] : sample) {
            all_features.insert(key);
        }
    }
    
    report.total_features = all_features.size();
    
    // Analyze each feature
    for (const auto& feature_name : all_features) {
        auto feature_values = extractFeature(X, feature_name);
        
        // Calculate variance
        double variance = calculateVariance(feature_values);
        report.feature_variance[feature_name] = variance;
        
        // Check if constant
        if (variance < 1e-10) {
            report.constant_features++;
        }
        
        // Calculate completeness
        double completeness = (static_cast<double>(feature_values.size()) / X.size()) * 100.0;
        report.feature_completeness[feature_name] = completeness;
    }
    
    // Check correlations
    std::vector<std::string> feature_list(all_features.begin(), all_features.end());
    for (size_t i = 0; i < feature_list.size(); ++i) {
        auto feature_i_values = extractFeature(X, feature_list[i]);
        
        for (size_t j = i + 1; j < feature_list.size(); ++j) {
            auto feature_j_values = extractFeature(X, feature_list[j]);
            double correlation = std::abs(calculateCorrelation(feature_i_values, feature_j_values));
            
            if (correlation > 0.95) {
                report.correlated_pairs.push_back({feature_list[i], feature_list[j]});
            }
        }
    }
    
    report.high_correlation_pairs = report.correlated_pairs.size();
    
    return report;
}

// Private utility functions
double ModelUtils::calculateVariance(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    
    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    
    return variance / values.size();
}

double ModelUtils::calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) {
        return 0.0;
    }
    
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    
    double numerator = 0.0;
    double sum_sq_x = 0.0;
    double sum_sq_y = 0.0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        double diff_x = x[i] - mean_x;
        double diff_y = y[i] - mean_y;
        
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }
    
    double denominator = std::sqrt(sum_sq_x * sum_sq_y);
    
    return (denominator > 1e-10) ? numerator / denominator : 0.0;
}

std::vector<double> ModelUtils::extractFeature(const std::vector<std::map<std::string, double>>& X, 
                                              const std::string& feature_name) {
    std::vector<double> values;
    values.reserve(X.size());
    
    for (const auto& sample : X) {
        auto it = sample.find(feature_name);
        if (it != sample.end()) {
            values.push_back(it->second);
        }
    }
    
    return values;
}

// Legacy function wrappers for backward compatibility
std::vector<std::vector<float>> 
toFloatMatrix(const std::vector<std::map<std::string, double>>& X) {
    return ModelUtils::toFloatMatrix(X, false); // No validation for backward compatibility
}

std::vector<float> 
toFloatVecInt(const std::vector<int>& y) {
    return ModelUtils::toFloatVecInt(y, false); // No validation for backward compatibility
}

std::vector<float> 
toFloatVecDouble(const std::vector<double>& y) {
    return ModelUtils::toFloatVecDouble(y, false); // No validation for backward compatibility
}

}
