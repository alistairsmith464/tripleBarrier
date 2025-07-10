#include "DataUtils.h"
#include "MLPipeline.h"
#include "MLSplits.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <set>

namespace MLPipeline {

template<typename T>
std::tuple<std::vector<std::map<std::string, double>>, std::vector<T>, std::vector<double>>
DataProcessor::cleanData(const std::vector<std::map<std::string, double>>& X, 
                        const std::vector<T>& y, 
                        const std::vector<double>& returns,
                        const CleaningOptions& options) {
    if (X.size() != y.size() || X.size() != returns.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }
    
    std::vector<std::map<std::string, double>> X_clean;
    std::vector<T> y_clean;
    std::vector<double> returns_clean;
    
    X_clean.reserve(X.size());
    y_clean.reserve(y.size());
    returns_clean.reserve(returns.size());
    
    size_t nan_count = 0, inf_count = 0, outlier_count = 0;
    
    std::vector<bool> is_outlier(X.size(), false);
    if (options.remove_outliers) {
        is_outlier = detectOutliers(returns, options.outlier_threshold);
    }
    
    for (size_t i = 0; i < X.size(); ++i) {
        bool valid = true;
        
        if (options.remove_nan || options.remove_inf) {
            for (const auto& kv : X[i]) {
                if (options.remove_nan && std::isnan(kv.second)) {
                    valid = false;
                    nan_count++;
                    break;
                }
                if (options.remove_inf && std::isinf(kv.second)) {
                    valid = false;
                    inf_count++;
                    break;
                }
            }
        }
        
        if (valid && (options.remove_nan || options.remove_inf)) {
            if (options.remove_nan && std::isnan(returns[i])) {
                valid = false;
                nan_count++;
            }
            if (options.remove_inf && std::isinf(returns[i])) {
                valid = false;
                inf_count++;
            }
        }
        
        if (valid && options.remove_outliers && is_outlier[i]) {
            valid = false;
            outlier_count++;
        }
        
        if (valid) {
            X_clean.push_back(X[i]);
            y_clean.push_back(y[i]);
            returns_clean.push_back(returns[i]);
        }
    }
    
    if (X_clean.empty()) {
        throw std::runtime_error("No valid data remaining after cleaning");
    }
    
    if (options.log_cleaning) {
        std::cout << "Data cleaning results:" << std::endl;
        std::cout << "  Original samples: " << X.size() << std::endl;
        std::cout << "  Clean samples: " << X_clean.size() << std::endl;
        std::cout << "  Removed NaN: " << nan_count << std::endl;
        std::cout << "  Removed Inf: " << inf_count << std::endl;
        std::cout << "  Removed outliers: " << outlier_count << std::endl;
    }
    
    if (options.normalize_features) {
        X_clean = normalizeFeatures(X_clean);
    }
    
    return std::make_tuple(std::move(X_clean), std::move(y_clean), std::move(returns_clean));
}

std::vector<std::map<std::string, double>> 
DataProcessor::normalizeFeatures(const std::vector<std::map<std::string, double>>& X,
                                const std::map<std::string, std::pair<double, double>>& stats) {
    if (X.empty()) return X;
    
    std::map<std::string, std::pair<double, double>> normalization_stats = stats;
    
    if (normalization_stats.empty()) {
        normalization_stats = calculateNormalizationStats(X);
    }
    
    std::vector<std::map<std::string, double>> X_normalized = X;
    
    for (auto& sample : X_normalized) {
        for (auto& feature : sample) {
            const std::string& feature_name = feature.first;
            double& value = feature.second;
            
            auto stats_it = normalization_stats.find(feature_name);
            if (stats_it != normalization_stats.end()) {
                double mean = stats_it->second.first;
                double std = stats_it->second.second;
                
                if (std > 1e-10) {
                    value = (value - mean) / std;
                }
            }
        }
    }
    
    return X_normalized;
}

std::map<std::string, std::pair<double, double>>
DataProcessor::calculateNormalizationStats(const std::vector<std::map<std::string, double>>& X) {
    std::map<std::string, std::pair<double, double>> stats;
    
    if (X.empty()) return stats;
    
    std::set<std::string> all_features;
    for (const auto& sample : X) {
        for (const auto& feature : sample) {
            all_features.insert(feature.first);
        }
    }
    
    for (const std::string& feature_name : all_features) {
        std::vector<double> values;
        values.reserve(X.size());
        
        for (const auto& sample : X) {
            auto it = sample.find(feature_name);
            if (it != sample.end()) {
                values.push_back(it->second);
            }
        }
        
        if (!values.empty()) {
            double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            
            double variance = 0.0;
            for (double value : values) {
                variance += (value - mean) * (value - mean);
            }
            variance /= values.size();
            double std = std::sqrt(variance);
            
            stats[feature_name] = {mean, std};
        }
    }
    
    return stats;
}

std::vector<bool> DataProcessor::detectOutliers(const std::vector<double>& values, double threshold) {
    std::vector<bool> is_outlier(values.size(), false);
    
    if (values.empty()) return is_outlier;
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    
    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();
    double std = std::sqrt(variance);
    
    if (std > 1e-10) { 
        for (size_t i = 0; i < values.size(); ++i) {
            double z_score = std::abs((values[i] - mean) / std);
            is_outlier[i] = (z_score > threshold);
        }
    }
    
    return is_outlier;
}

DataProcessor::DataQuality DataProcessor::analyzeDataQuality(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<double>& returns) {
    
    DataQuality quality;
    quality.total_samples = X.size();
    quality.valid_samples = 0;
    quality.nan_count = 0;
    quality.inf_count = 0;
    quality.outlier_count = 0;
    
    std::set<std::string> all_features;
    for (const auto& sample : X) {
        for (const auto& feature : sample) {
            all_features.insert(feature.first);
        }
    }
    
    for (const std::string& feature : all_features) {
        quality.feature_completeness[feature] = 0.0;
    }
    
    auto outliers = detectOutliers(returns);
    
    for (size_t i = 0; i < X.size(); ++i) {
        bool sample_valid = true;
        
        for (const std::string& feature : all_features) {
            auto it = X[i].find(feature);
            if (it != X[i].end()) {
                if (std::isnan(it->second)) {
                    quality.nan_count++;
                    sample_valid = false;
                } else if (std::isinf(it->second)) {
                    quality.inf_count++;
                    sample_valid = false;
                } else {
                    quality.feature_completeness[feature] += 1.0;
                }
            }
        }
        
        if (i < returns.size()) {
            if (std::isnan(returns[i])) {
                quality.nan_count++;
                sample_valid = false;
            } else if (std::isinf(returns[i])) {
                quality.inf_count++;
                sample_valid = false;
            }
        }
        
        if (i < outliers.size() && outliers[i]) {
            quality.outlier_count++;
        }
        
        if (sample_valid) {
            quality.valid_samples++;
        }
    }
    
    for (auto& pair : quality.feature_completeness) {
        pair.second = (pair.second / quality.total_samples) * 100.0;
    }
    
    return quality;
}

template<typename T>
std::vector<T> select_rows(const std::vector<T>& data, const std::vector<size_t>& idxs) {
    std::vector<T> result;
    result.reserve(idxs.size());
    
    for (size_t idx : idxs) {
        if (idx >= data.size()) {
            throw std::out_of_range("Index " + std::to_string(idx) + " out of range for data size " + std::to_string(data.size()));
        }
        result.push_back(data[idx]);
    }
    
    return result;
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const SplitConfig& config) {
    if (data_size == 0) {
        throw std::invalid_argument("Data size cannot be zero");
    }
    
    std::vector<size_t> train_idx, val_idx, test_idx;
    
    switch (config.strategy) {
        case SplitStrategy::CHRONOLOGICAL: {
            // Chronological splits for time series - maintains temporal order
            std::cout << "[INFO] Using CHRONOLOGICAL splits - temporal order preserved" << std::endl;
            
            size_t n_test = static_cast<size_t>(data_size * config.test_size);
            size_t n_val = static_cast<size_t>(data_size * config.val_size);
            
            // Apply embargo period to prevent information leakage between sets
            size_t embargo_size = static_cast<size_t>(config.embargo);
            
            // Adjust sizes to account for embargo periods
            size_t total_embargo = 2 * embargo_size; // Between train-val and val-test
            if (total_embargo >= data_size) {
                throw std::invalid_argument("Embargo period too large for dataset size");
            }
            
            size_t usable_size = data_size - total_embargo;
            n_test = static_cast<size_t>(usable_size * config.test_size);
            n_val = static_cast<size_t>(usable_size * config.val_size);
            size_t n_train = usable_size - n_test - n_val;
            
            train_idx.reserve(n_train);
            val_idx.reserve(n_val);
            test_idx.reserve(n_test);
            
            // Training set: earliest data
            for (size_t i = 0; i < n_train; ++i) train_idx.push_back(i);
            
            // Validation set: middle data (with embargo gap)
            size_t val_start = n_train + embargo_size;
            for (size_t i = val_start; i < val_start + n_val; ++i) val_idx.push_back(i);
            
            // Test set: latest data (with embargo gap)
            size_t test_start = val_start + n_val + embargo_size;
            for (size_t i = test_start; i < test_start + n_test; ++i) test_idx.push_back(i);
            
            if (embargo_size > 0) {
                std::cout << "[INFO] Applied embargo period of " << embargo_size << " samples between sets" << std::endl;
            }
            
            break;
        }
        
        case SplitStrategy::RANDOM: {
            // WARNING: Random splits are DANGEROUS for time series financial data!
            // This can cause severe temporal data leakage and inflated performance metrics
            std::cout << "[CRITICAL WARNING] Using RANDOM splits on time series data!" << std::endl;
            std::cout << "[CRITICAL WARNING] This WILL cause temporal data leakage and invalid results!" << std::endl;
            std::cout << "[CRITICAL WARNING] Consider using CHRONOLOGICAL or PURGED_KFOLD instead!" << std::endl;
            
            std::vector<size_t> indices(data_size);
            std::iota(indices.begin(), indices.end(), 0);
            
            if (config.shuffle) {
                std::cout << "[CRITICAL WARNING] Shuffling time series data - this breaks temporal order!" << std::endl;
                std::mt19937 rng(config.random_seed);
                std::shuffle(indices.begin(), indices.end(), rng);
            }
            
            size_t n_test = static_cast<size_t>(data_size * config.test_size);
            size_t n_val = static_cast<size_t>(data_size * config.val_size);
            size_t n_train = data_size - n_test - n_val;
            
            train_idx.assign(indices.begin(), indices.begin() + n_train);
            val_idx.assign(indices.begin() + n_train, indices.begin() + n_train + n_val);
            test_idx.assign(indices.begin() + n_train + n_val, indices.end());
            break;
        }
        
        case SplitStrategy::PURGED_KFOLD: {
            // Purged K-fold for time series with embargo periods
            std::cout << "[INFO] Using PURGED_KFOLD splits with embargo=" << config.embargo << " for time series" << std::endl;
            
            auto folds = MLSplitUtils::purgedKFoldSplit(data_size, config.n_splits, config.embargo);
            if (!folds.empty()) {
                train_idx = std::move(folds[0].train_indices);
                val_idx = std::move(folds[0].val_indices);
                // Use last fold's validation set as test set
                test_idx = std::move(folds.back().val_indices);
            }
            break;
        }
        
        default:
            throw std::invalid_argument("Unsupported split strategy");
    }
    
    return std::make_tuple(std::move(train_idx), std::move(val_idx), std::move(test_idx));
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const PipelineConfig& config) {
    SplitConfig split_config;
    split_config.test_size = config.test_size;
    split_config.val_size = config.val_size;
    // FORCE chronological splits for financial time series data
    split_config.strategy = SplitStrategy::CHRONOLOGICAL;  // Override any other setting
    split_config.n_splits = config.n_splits;
    split_config.embargo = config.embargo;
    split_config.shuffle = false;  // NEVER shuffle time series data
    
    std::cout << "[INFO] PipelineConfig conversion: Enforcing CHRONOLOGICAL splits for time series safety" << std::endl;
    
    return createSplits(data_size, split_config);
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const UnifiedPipelineConfig& config) {
    SplitConfig split_config;
    split_config.test_size = config.test_size;
    split_config.val_size = config.val_size;
    split_config.strategy = SplitStrategy::CHRONOLOGICAL;
    split_config.embargo = config.embargo;
    split_config.shuffle = false;
    
    return createSplits(data_size, split_config);
}

// Time-series safe split function for financial data - ALWAYS uses chronological splits
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, double test_size, double val_size, bool enforce_chronological) {
    if (data_size == 0) {
        throw std::invalid_argument("Data size cannot be zero");
    }
    
    // For financial time series data, ALWAYS use chronological splits to prevent temporal leakage
    SplitConfig split_config;
    split_config.test_size = test_size;
    split_config.val_size = val_size;
    split_config.strategy = enforce_chronological ? SplitStrategy::CHRONOLOGICAL : SplitStrategy::CHRONOLOGICAL; // Force chronological
    split_config.shuffle = false; // Never shuffle time series data
    
    std::cout << "[WARNING] Using CHRONOLOGICAL splits to prevent temporal data leakage in time series financial data" << std::endl;
    
    return createSplits(data_size, split_config);
}

template<typename T>
std::tuple<std::vector<std::map<std::string, double>>, std::vector<T>, std::vector<double>>
cleanData(const std::vector<std::map<std::string, double>>& X, 
          const std::vector<T>& y, 
          const std::vector<double>& returns) {
    return DataProcessor::cleanData(X, y, returns);
}

template std::tuple<std::vector<std::map<std::string, double>>, std::vector<int>, std::vector<double>>
DataProcessor::cleanData(const std::vector<std::map<std::string, double>>&, const std::vector<int>&, const std::vector<double>&, const DataProcessor::CleaningOptions&);

template std::tuple<std::vector<std::map<std::string, double>>, std::vector<double>, std::vector<double>>
DataProcessor::cleanData(const std::vector<std::map<std::string, double>>&, const std::vector<double>&, const std::vector<double>&, const DataProcessor::CleaningOptions&);

template std::tuple<std::vector<std::map<std::string, double>>, std::vector<int>, std::vector<double>>
cleanData(const std::vector<std::map<std::string, double>>&, const std::vector<int>&, const std::vector<double>&);

template std::tuple<std::vector<std::map<std::string, double>>, std::vector<double>, std::vector<double>>
cleanData(const std::vector<std::map<std::string, double>>&, const std::vector<double>&, const std::vector<double>&);

template std::vector<std::map<std::string, double>> 
select_rows(const std::vector<std::map<std::string, double>>&, const std::vector<size_t>&);

template std::vector<int> 
select_rows(const std::vector<int>&, const std::vector<size_t>&);

template std::vector<double> 
select_rows(const std::vector<double>&, const std::vector<size_t>&);

}
