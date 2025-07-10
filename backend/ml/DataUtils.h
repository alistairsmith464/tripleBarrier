#pragma once
#include <vector>
#include <map>
#include <string>
#include <tuple>

namespace MLPipeline {

// Forward declarations
struct PipelineConfig;
struct UnifiedPipelineConfig;

// Data cleaning and preprocessing utilities
class DataProcessor {
public:
    // Data cleaning options
    struct CleaningOptions {
        bool remove_nan = true;
        bool remove_inf = true;
        bool remove_outliers = false;
        double outlier_threshold = 3.0; // z-score threshold
        bool normalize_features = false;
        bool log_cleaning = false;
    };
    
    // Data cleaning with comprehensive validation
    template<typename T>
    static std::tuple<std::vector<std::map<std::string, double>>, std::vector<T>, std::vector<double>>
    cleanData(const std::vector<std::map<std::string, double>>& X, 
              const std::vector<T>& y, 
              const std::vector<double>& returns,
              const CleaningOptions& options = CleaningOptions{});
    
    // Feature normalization utilities
    static std::vector<std::map<std::string, double>> 
    normalizeFeatures(const std::vector<std::map<std::string, double>>& X,
                     const std::map<std::string, std::pair<double, double>>& stats = {});
    
    static std::map<std::string, std::pair<double, double>>
    calculateNormalizationStats(const std::vector<std::map<std::string, double>>& X);
    
    // Outlier detection and removal
    static std::vector<bool> detectOutliers(const std::vector<double>& values, double threshold = 3.0);
    
    // Data quality diagnostics
    struct DataQuality {
        size_t total_samples;
        size_t valid_samples;
        size_t nan_count;
        size_t inf_count;
        size_t outlier_count;
        std::map<std::string, double> feature_completeness;
    };
    
    static DataQuality analyzeDataQuality(const std::vector<std::map<std::string, double>>& X,
                                        const std::vector<double>& returns);
};

// Efficient data selection utilities
template<typename T>
std::vector<T> select_rows(const std::vector<T>& data, const std::vector<size_t>& idxs);

// Enhanced data splitting with multiple strategies
enum class SplitStrategy {
    CHRONOLOGICAL,
    PURGED_KFOLD,
    STRATIFIED,
    RANDOM
};

struct SplitConfig {
    SplitStrategy strategy = SplitStrategy::CHRONOLOGICAL;
    double test_size = 0.2;
    double val_size = 0.2;
    int n_splits = 5;
    int embargo = 0;
    int random_seed = 42;
    bool shuffle = true;
};

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const SplitConfig& config);

// Backward compatibility
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const PipelineConfig& config);

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const UnifiedPipelineConfig& config);

}
