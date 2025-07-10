#pragma once
#include <vector>
#include <map>
#include <string>
#include "XGBoostModel.h"
#include "PortfolioSimulator.h"

namespace MLPipeline {  
    struct PipelineResult {
        std::vector<int> predictions;
        std::vector<double> probabilities;
        PortfolioSimulation portfolio;
    };

    struct RegressionPipelineResult {
        std::vector<double> predictions;
        PortfolioSimulation portfolio;
    };

    enum SplitType {
        Chronological,
        PurgedKFold
    };

    enum class BarrierType {
        HARD,
        SOFT
    };

    // Original pipeline configuration structure (deprecated, use UnifiedPipelineConfig)
    struct PipelineConfig {
        double test_size = 0.2;
        double val_size = 0.2;
        int n_rounds = 100;
        int max_depth = 6;
        int nthread = 4;
        std::string objective = "binary:logistic";
        
        SplitType split_type = Chronological;
        int n_splits = 5;
        int embargo = 0;
        double train_ratio = 0.6;
        double val_ratio = 0.2; 
        double test_ratio = 0.2;
        int random_seed = 42;
        XGBoostConfig xgb_config;
        
        // Validation method
        bool validate() const {
            double total = train_ratio + val_ratio + test_ratio;
            return std::abs(total - 1.0) < 1e-6 && 
                   train_ratio > 0 && val_ratio >= 0 && test_ratio >= 0;
        }
    };

    struct HyperparameterGrid {
        std::vector<int> n_rounds = {50, 100, 200};
        std::vector<int> max_depth = {3, 5, 7};
        std::vector<double> learning_rate = {0.01, 0.1, 0.3};
        std::vector<double> subsample = {0.8, 1.0};
        std::vector<double> colsample_bytree = {0.8, 1.0};
    };

    struct UnifiedPipelineConfig {
        double test_size = 0.2;
        double val_size = 0.2;
        int n_rounds = 100;
        int max_depth = 6;
        int nthread = 4;
        std::string objective = "binary:logistic";
        double learning_rate = 0.1;
        double subsample = 1.0;
        double colsample_bytree = 1.0;
        BarrierType barrier_type = BarrierType::HARD;
        HyperparameterGrid hyperparameter_grid;
    };

    // Template-based pipeline to reduce code duplication
    template<typename LabelType>
    struct PipelineTraits {};
    
    template<>
    struct PipelineTraits<int> {
        using ResultType = PipelineResult;
        using ModelPredictionType = std::vector<int>;
        static constexpr bool is_classification = true;
    };
    
    template<>
    struct PipelineTraits<double> {
        using ResultType = RegressionPipelineResult;
        using ModelPredictionType = std::vector<double>;
        static constexpr bool is_classification = false;
    };

    // Main pipeline functions
    PipelineResult runPipeline(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        const std::vector<double>& returns,
        const PipelineConfig& config
    );

    PipelineResult runPipelineWithTuning(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        const std::vector<double>& returns,
        const PipelineConfig& config
    );

    RegressionPipelineResult runPipelineRegression(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& returns,
        const PipelineConfig& config
    );

    RegressionPipelineResult runPipelineRegressionWithTuning(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& returns,
        const PipelineConfig& config
    );

    // UnifiedPipelineConfig overloads
    PipelineResult runPipeline(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        const std::vector<double>& returns,
        const UnifiedPipelineConfig& config
    );

    PipelineResult runPipelineWithTuning(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        const std::vector<double>& returns,
        UnifiedPipelineConfig config
    );

    RegressionPipelineResult runPipelineRegression(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& returns,
        const UnifiedPipelineConfig& config
    );

    RegressionPipelineResult runPipelineRegressionWithTuning(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& returns,
        UnifiedPipelineConfig config
    );
}
