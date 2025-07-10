#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "../data/FeatureExtractor.h"
#include "../data/LabeledEvent.h"
#include "../data/PreprocessedRow.h"
#include "XGBoostModel.h"
#include "PortfolioSimulator.h"

namespace MLPipeline {

class BarrierMLStrategy {
public:
    virtual ~BarrierMLStrategy() = default;
    
    struct TrainingConfig {
        double test_size = 0.2;
        double val_size = 0.2;
        int n_rounds = 100;
        int max_depth = 6;
        int nthread = 4;
        double learning_rate = 0.1;
        double subsample = 1.0;
        double colsample_bytree = 1.0;
        int random_seed = 42;
    };
    
    struct PredictionResult {
        std::vector<double> predictions;     
        std::vector<double> trading_signals;   
        std::vector<double> confidence_scores; 
        PortfolioSimulation portfolio_result;
        bool success = false;
        std::string error_message;
    };
    
    virtual FeatureExtractor::FeatureExtractionResult extractFeatures(
        const std::set<std::string>& selectedFeatures,
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents) = 0;
    
    virtual PredictionResult trainAndPredict(
        const FeatureExtractor::FeatureExtractionResult& features,
        const std::vector<double>& returns,
        const TrainingConfig& config) = 0;
    
    virtual std::string getStrategyName() const = 0;
    virtual std::string getModelObjective() const = 0;
    
protected:
    std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>> 
    createTrainValTestSplits(size_t data_size, const TrainingConfig& config);
    
    PortfolioSimulation runPortfolioSimulation(
        const std::vector<double>& trading_signals,
        const std::vector<double>& returns,
        const PortfolioConfig& portfolio_config = PortfolioConfig{});
};

class HardBarrierStrategy : public BarrierMLStrategy {
public:
    FeatureExtractor::FeatureExtractionResult extractFeatures(
        const std::set<std::string>& selectedFeatures,
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents) override;
    
    PredictionResult trainAndPredict(
        const FeatureExtractor::FeatureExtractionResult& features,
        const std::vector<double>& returns,
        const TrainingConfig& config) override;
    
    std::string getStrategyName() const override { return "Hard Barrier"; }
    std::string getModelObjective() const override { return "multi:softprob"; }

private:
    std::vector<double> convertClassificationToTradingSignals(
        const std::vector<int>& predictions,
        const std::vector<double>& probabilities);
};

class TTBMStrategy : public BarrierMLStrategy {
public:
    FeatureExtractor::FeatureExtractionResult extractFeatures(
        const std::set<std::string>& selectedFeatures,
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents) override;
    
    PredictionResult trainAndPredict(
        const FeatureExtractor::FeatureExtractionResult& features,
        const std::vector<double>& returns,
        const TrainingConfig& config) override;
    
    std::string getStrategyName() const override { return "TTBM (Time-To-Barrier Meta-Labeling)"; }
    std::string getModelObjective() const override { return "reg:squarederror"; }

private:
    std::vector<double> convertRegressionToTradingSignals(
        const std::vector<double>& predictions);
    
    std::vector<double> normalizeToTradingRange(
        const std::vector<double>& raw_predictions);
};

class BarrierMLStrategyFactory {
public:
    enum class StrategyType {
        HARD_BARRIER,
        TTBM
    };
    
    static std::unique_ptr<BarrierMLStrategy> createStrategy(StrategyType type);
    static StrategyType getStrategyType(bool use_ttbm);
};

class UnifiedMLPipeline {
public:
    struct PipelineConfig {
        BarrierMLStrategyFactory::StrategyType strategy_type;
        BarrierMLStrategy::TrainingConfig training_config;
        PortfolioConfig portfolio_config;
        std::set<std::string> selected_features;
        bool enable_hyperparameter_tuning = false;
        bool enable_detailed_logging = true;
    };
    
    struct PipelineResult {
        BarrierMLStrategy::PredictionResult prediction_result;
        std::string strategy_name;
        std::map<std::string, double> performance_metrics;
        bool success = false;
        std::string error_message;
    };
    
    static PipelineResult runPipeline(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const PipelineConfig& config);

private:
    static std::map<std::string, double> calculatePerformanceMetrics(
        const BarrierMLStrategy::PredictionResult& result,
        const FeatureExtractor::FeatureExtractionResult& features);
};

} // namespace MLPipeline
