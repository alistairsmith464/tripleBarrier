#pragma once
#include <vector>
#include <map>
#include <string>
#include <tuple>

namespace MLPipeline {

struct PipelineConfig;
struct UnifiedPipelineConfig;

class DataProcessor {
public:
    struct CleaningOptions {
        bool remove_nan;
        bool remove_inf;
        bool remove_outliers;
        double outlier_threshold;
        bool normalize_features;
        
        CleaningOptions() : remove_nan(true), remove_inf(true), remove_outliers(false), 
                           outlier_threshold(3.0), normalize_features(false) {}
    };
    
    template<typename T>
    static std::tuple<std::vector<std::map<std::string, double>>, std::vector<T>, std::vector<double>>
    cleanData(const std::vector<std::map<std::string, double>>& X, 
              const std::vector<T>& y, 
              const std::vector<double>& returns,
              const CleaningOptions& options = CleaningOptions{});
    
    static std::vector<std::map<std::string, double>> 
    normalizeFeatures(const std::vector<std::map<std::string, double>>& X,
                     const std::map<std::string, std::pair<double, double>>& stats = {});
    
    static std::map<std::string, std::pair<double, double>>
    calculateNormalizationStats(const std::vector<std::map<std::string, double>>& X);
    
    static std::vector<bool> detectOutliers(const std::vector<double>& values, double threshold = 3.0);
    
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

template<typename T>
std::vector<T> select_rows(const std::vector<T>& data, const std::vector<size_t>& idxs);

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
    bool shuffle = false; 
};

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const SplitConfig& config);

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const PipelineConfig& config);

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, const UnifiedPipelineConfig& config);

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
createSplits(size_t data_size, double test_size, double val_size, bool enforce_chronological = true);

}
