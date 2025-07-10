#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>

namespace MLPipeline {

// Enhanced model utilities with comprehensive preprocessing
class ModelUtils {
public:
    // Data conversion utilities with validation
    static std::vector<std::vector<float>> 
    toFloatMatrix(const std::vector<std::map<std::string, double>>& X, 
                  bool validate_input = true);
    
    static std::vector<float> 
    toFloatVecInt(const std::vector<int>& y, bool validate_input = true);
    
    static std::vector<float> 
    toFloatVecDouble(const std::vector<double>& y, bool validate_input = true);
    
    // Feature preprocessing utilities
    struct PreprocessingConfig {
        bool normalize_features = false;
        bool scale_features = false;
        bool encode_categorical = false;
        bool remove_constant_features = false;
        bool remove_correlated_features = false;
        double correlation_threshold = 0.95;
        std::string scaling_method = "standard"; // "standard", "minmax", "robust"
    };
    
    struct PreprocessingResult {
        std::vector<std::map<std::string, double>> processed_data;
        std::vector<std::string> feature_names;
        std::map<std::string, std::pair<double, double>> scaling_params; // mean/std or min/max
        std::vector<std::string> removed_features;
    };
    
    static PreprocessingResult preprocessFeatures(
        const std::vector<std::map<std::string, double>>& X,
        const PreprocessingConfig& config = PreprocessingConfig{});
    
    static std::vector<std::map<std::string, double>> applyPreprocessing(
        const std::vector<std::map<std::string, double>>& X,
        const PreprocessingResult& preprocessing_info);
    
    // Feature engineering utilities
    static std::vector<std::map<std::string, double>> 
    createPolynomialFeatures(const std::vector<std::map<std::string, double>>& X, 
                            int degree = 2);
    
    static std::vector<std::map<std::string, double>>
    createInteractionFeatures(const std::vector<std::map<std::string, double>>& X,
                             const std::vector<std::pair<std::string, std::string>>& interactions = {});
    
    // Feature selection utilities
    static std::vector<std::string> selectFeaturesByVariance(
        const std::vector<std::map<std::string, double>>& X,
        double variance_threshold = 0.01);
    
    static std::vector<std::string> selectFeaturesByCorrelation(
        const std::vector<std::map<std::string, double>>& X,
        double correlation_threshold = 0.95);
    
    // Model validation utilities
    static bool validateModelInputs(const std::vector<std::vector<float>>& X,
                                   const std::vector<float>& y);
    
    static void checkDataConsistency(const std::vector<std::map<std::string, double>>& X_train,
                                    const std::vector<std::map<std::string, double>>& X_test);
    
    // Feature importance utilities
    struct FeatureImportance {
        std::string feature_name;
        double importance_score;
        double rank;
    };
    
    static std::vector<FeatureImportance> 
    computeFeatureImportance(const std::vector<std::map<std::string, double>>& X,
                            const std::vector<double>& y,
                            const std::string& method = "correlation");
    
    // Data quality checks
    struct DataQualityReport {
        size_t total_features;
        size_t constant_features;
        size_t high_correlation_pairs;
        std::map<std::string, double> feature_variance;
        std::map<std::string, double> feature_completeness;
        std::vector<std::pair<std::string, std::string>> correlated_pairs;
    };
    
    static DataQualityReport analyzeDataQuality(
        const std::vector<std::map<std::string, double>>& X);

private:
    static double calculateVariance(const std::vector<double>& values);
    static double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y);
    static std::vector<double> extractFeature(const std::vector<std::map<std::string, double>>& X, 
                                             const std::string& feature_name);
};

// Legacy function wrappers for backward compatibility
std::vector<std::vector<float>> 
toFloatMatrix(const std::vector<std::map<std::string, double>>& X);

std::vector<float> 
toFloatVecInt(const std::vector<int>& y);

std::vector<float> 
toFloatVecDouble(const std::vector<double>& y);

}
