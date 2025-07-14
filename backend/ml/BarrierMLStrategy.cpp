#include "BarrierMLStrategy.h"
#include "MLSplits.h"
#include "ModelUtils.h"
#include "MetricsCalculator.h"
#include "DataUtils.h"
#include "../utils/Exceptions.h"
#include "../utils/ErrorHandling.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace MLPipeline {

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>> 
BarrierMLStrategy::createTrainValTestSplits(size_t data_size, const TrainingConfig& config) {
    return createSplits(data_size, config.test_size, config.val_size);
}

PortfolioSimulation BarrierMLStrategy::runPortfolioSimulation(
    const std::vector<double>& trading_signals,
    const std::vector<double>& returns,
    const PortfolioConfig& portfolio_config) {
    
    return simulate_portfolio(trading_signals, returns, portfolio_config);
}

FeatureExtractor::FeatureExtractionResult HardBarrierStrategy::extractFeatures(
    const std::set<std::string>& selectedFeatures,
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents) {
    
    return FeatureExtractor::extractFeaturesForClassification(selectedFeatures, rows, labeledEvents);
}

BarrierMLStrategy::PredictionResult HardBarrierStrategy::trainAndPredict(
    const FeatureExtractor::FeatureExtractionResult& features,
    const std::vector<double>& returns,
    const TrainingConfig& config) {
    
    using namespace TripleBarrier;
    
    PredictionResult result;
    
    try {
        Validation::validateNotEmpty(features.features, "features");
        Validation::validateNotEmpty(features.labels, "classification_labels");
        Validation::validateNotEmpty(returns, "returns");
        
        if (features.features.size() != features.labels.size()) {
            throw DataValidationException(
                "Size mismatch: features (" + std::to_string(features.features.size()) + 
                ") vs labels (" + std::to_string(features.labels.size()) + ")"
            );
        }
        
        Validation::validateRange(config.test_size, 0.0, 0.8, "test_size");
        Validation::validateRange(config.val_size, 0.0, 0.8, "val_size");
        Validation::validatePositive(config.learning_rate, "learning_rate");
        
        DataProcessor::CleaningOptions cleaning_opts;
        cleaning_opts.remove_nan = true;
        cleaning_opts.remove_inf = true;
        cleaning_opts.remove_outliers = true;
        
        auto [X_clean, y_clean, returns_clean] = DataProcessor::cleanData(
            features.features, features.labels, returns, cleaning_opts);
        
        Validation::validateNotEmpty(X_clean, "cleaned_features");
        Validation::validateNotEmpty(y_clean, "cleaned_labels");
        
        if (X_clean.size() < 10) {
            throw DataProcessingException("Insufficient data after cleaning", 
                                        "samples: " + std::to_string(X_clean.size()));
        }
        
        auto [train_idx, val_idx, test_idx] = createTrainValTestSplits(X_clean.size(), config);

        if (train_idx.empty()) {
            throw DataProcessingException("No training samples available after split");
        }
        
        auto X_train = toFloatMatrix(select_rows(X_clean, train_idx));
        auto y_train = toFloatVecInt(select_rows(y_clean, train_idx));
        
        std::vector<size_t> eval_idx = val_idx.empty() ? test_idx : val_idx;
        if (eval_idx.empty()) {
            throw DataProcessingException("No evaluation samples available after split");
        }
        
        auto X_eval = toFloatMatrix(select_rows(X_clean, eval_idx));
        auto returns_eval = select_rows(returns_clean, eval_idx);
        
        XGBoostConfig model_config;
        model_config.n_rounds = config.n_rounds;
        model_config.max_depth = config.max_depth;
        model_config.nthread = config.nthread;
        model_config.objective = getModelObjective();
        model_config.learning_rate = config.learning_rate;
        model_config.subsample = config.subsample;
        model_config.colsample_bytree = config.colsample_bytree;
        model_config.num_class = 3;
        
        if (model_config.n_rounds <= 0) {
            throw HyperparameterException("n_rounds must be positive", "n_rounds");
        }
        if (model_config.max_depth <= 0) {
            throw HyperparameterException("max_depth must be positive", "max_depth");
        }
        
        XGBoostModel model;
        try {
            model.fit(X_train, y_train, model_config);
        } catch (const BaseException& e) {
            throw ModelTrainingException("XGBoost training failed: " + std::string(e.what()), e.context());
        } catch (const std::exception& e) {
            throw ModelTrainingException("XGBoost training failed", e.what());
        }
        
        if (!model.is_trained()) {
            throw ModelTrainingException("Model training completed but model is not in trained state");
        }
        
        std::vector<int> y_pred;
        std::vector<float> y_prob;
        
        try {
            y_pred = model.predict(X_eval);
            y_prob = model.predict_proba(X_eval);
        } catch (const BaseException& e) {
            throw ModelPredictionException("XGBoost prediction failed: " + std::string(e.what()), e.context());
        } catch (const std::exception& e) {
            throw ModelPredictionException("XGBoost prediction failed", e.what());
        }
        
        Validation::validateNotEmpty(y_pred, "predictions");
        Validation::validateNotEmpty(y_prob, "probabilities");
        
        if (y_pred.size() != eval_idx.size()) {
            throw ModelPredictionException("Prediction count mismatch", 
                                         "expected: " + std::to_string(eval_idx.size()) + 
                                         ", got: " + std::to_string(y_pred.size()));
        }
        
        result.predictions.assign(y_pred.begin(), y_pred.end());
        result.confidence_scores.assign(y_prob.begin(), y_prob.end());
        
        try {
            result.trading_signals = convertClassificationToTradingSignals(y_pred, std::vector<double>(y_prob.begin(), y_prob.end()));
            result.portfolio_result = runPortfolioSimulation(result.trading_signals, returns_eval);   
        } catch (const BaseException& e) {
            throw PortfolioException("Portfolio simulation failed: " + std::string(e.what()), e.context());
        } catch (const std::exception& e) {
            throw PortfolioException("Portfolio simulation failed", e.what());
        }
        
        result.success = true;
        return result;
        
    } catch (const BaseException& e) {
        result.error_message = e.full_message();
        result.success = false;
        return result;
    } catch (const std::exception& e) {
        auto converted = ExceptionUtils::convertException(e, "Hard Barrier Strategy Training");
        result.error_message = converted->full_message();
        result.success = false;
        return result;
    }
}

std::vector<double> HardBarrierStrategy::convertClassificationToTradingSignals(
    const std::vector<int>& predictions,
    const std::vector<double>& probabilities) {
    
    std::vector<double> signals;
    signals.reserve(predictions.size());
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        double signal = 0.0;
        
        int pred = predictions[i];
        if (pred == 2) {
            signal = 1.0;
        } else if (pred == 0) {
            signal = -1.0;
        } else {
            signal = 0.0;
        }
        
        signals.push_back(signal);
    }
    
    return signals;
}

FeatureExtractor::FeatureExtractionResult TTBMStrategy::extractFeatures(
    const std::set<std::string>& selectedFeatures,
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents) {
    
    return FeatureExtractor::extractFeaturesForRegression(selectedFeatures, rows, labeledEvents);
}

BarrierMLStrategy::PredictionResult TTBMStrategy::trainAndPredict(
    const FeatureExtractor::FeatureExtractionResult& features,
    const std::vector<double>& returns,
    const TrainingConfig& config) {
    
    PredictionResult result;
    
    try {
        if (features.features.empty() || features.labels_double.empty()) {
            result.error_message = "No valid regression data available";
            return result;
        }
        
        DataProcessor::CleaningOptions cleaning_opts;
        cleaning_opts.remove_nan = true;
        cleaning_opts.remove_inf = true;
        cleaning_opts.remove_outliers = true;
        
        auto [X_clean, y_clean, returns_clean] = DataProcessor::cleanData(
            features.features, features.labels_double, returns, cleaning_opts);
        
        auto [train_idx, val_idx, test_idx] = createTrainValTestSplits(X_clean.size(), config);
        
        auto X_train = toFloatMatrix(select_rows(X_clean, train_idx));
        auto y_train = toFloatVecDouble(select_rows(y_clean, train_idx));
        
        std::vector<size_t> eval_idx = val_idx.empty() ? test_idx : val_idx;
        auto X_eval = toFloatMatrix(select_rows(X_clean, eval_idx));
        auto returns_eval = select_rows(returns_clean, eval_idx);
        
        if (!y_train.empty()) {
            float min_train = *std::min_element(y_train.begin(), y_train.end());
            float max_train = *std::max_element(y_train.begin(), y_train.end());
            float sum_train = std::accumulate(y_train.begin(), y_train.end(), 0.0f);
            float mean_train = sum_train / y_train.size();
        }
        
        XGBoostConfig model_config;
        model_config.n_rounds = config.n_rounds;
        model_config.max_depth = config.max_depth;
        model_config.nthread = config.nthread;
        model_config.objective = getModelObjective();
        model_config.learning_rate = config.learning_rate;
        model_config.subsample = config.subsample;
        model_config.colsample_bytree = config.colsample_bytree;
        
        XGBoostModel model;
        model.fit(X_train, y_train, model_config);
        
       auto y_pred_raw = model.predict_raw(X_eval); 
        
        if (!y_pred_raw.empty()) {
            float min_pred = *std::min_element(y_pred_raw.begin(), y_pred_raw.end());
            float max_pred = *std::max_element(y_pred_raw.begin(), y_pred_raw.end());
            float sum_pred = std::accumulate(y_pred_raw.begin(), y_pred_raw.end(), 0.0f);
            float mean_pred = sum_pred / y_pred_raw.size();
        }
        
        result.predictions.assign(y_pred_raw.begin(), y_pred_raw.end());
        result.trading_signals = convertRegressionToTradingSignals(result.predictions);
        
        result.confidence_scores.reserve(result.predictions.size());
        for (double pred : result.predictions) {
            result.confidence_scores.push_back(std::abs(pred));
        }
        
        result.portfolio_result = runPortfolioSimulation(
            result.trading_signals, returns_eval);
        
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("TTBM training failed: ") + e.what();
        result.success = false;
    }
    
    return result;
}

std::vector<double> TTBMStrategy::convertRegressionToTradingSignals(
    const std::vector<double>& predictions) {
    
    return normalizeToTradingRange(predictions);
}

std::vector<double> TTBMStrategy::normalizeToTradingRange(
    const std::vector<double>& raw_predictions) {
    
    if (raw_predictions.empty()) {
        return {};
    }
    
    double max_abs = 0.0;
    for (double pred : raw_predictions) {
        max_abs = std::max(max_abs, std::abs(pred));
    }
    
    std::vector<double> normalized;
    normalized.reserve(raw_predictions.size());
    
    if (max_abs > 1e-6) {
        for (double pred : raw_predictions) {
            normalized.push_back(pred / max_abs);
        }
    } else {
        normalized.assign(raw_predictions.size(), 0.0);
    }
    
    return normalized;
}

std::unique_ptr<BarrierMLStrategy> BarrierMLStrategyFactory::createStrategy(StrategyType type) {
    switch (type) {
        case StrategyType::HARD_BARRIER:
            return std::make_unique<HardBarrierStrategy>();
        case StrategyType::TTBM:
            return std::make_unique<TTBMStrategy>();
        default:
            throw std::invalid_argument("Unknown strategy type");
    }
}

BarrierMLStrategyFactory::StrategyType BarrierMLStrategyFactory::getStrategyType(bool use_ttbm) {
    return use_ttbm ? StrategyType::TTBM : StrategyType::HARD_BARRIER;
}

UnifiedMLPipeline::PipelineResult UnifiedMLPipeline::runPipeline(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const PipelineConfig& config) {
    
    PipelineResult pipeline_result;
    
    try {
        auto strategy = BarrierMLStrategyFactory::createStrategy(config.strategy_type);
        pipeline_result.strategy_name = strategy->getStrategyName();
        
        auto features = strategy->extractFeatures(config.selected_features, rows, labeledEvents);
        
        std::vector<double> returns;
        returns.reserve(rows.size());
        for (const auto& row : rows) {
            returns.push_back(row.log_return);
        }
        
        pipeline_result.prediction_result = strategy->trainAndPredict(features, returns, config.training_config);
        
        if (pipeline_result.prediction_result.success) {
            pipeline_result.performance_metrics = calculatePerformanceMetrics(
                pipeline_result.prediction_result, features);
            pipeline_result.success = true;
        } else {
            pipeline_result.error_message = pipeline_result.prediction_result.error_message;
            pipeline_result.success = false;
        }
        
    } catch (const std::exception& e) {
        pipeline_result.error_message = std::string("Pipeline failed: ") + e.what();
        pipeline_result.success = false;
    }
    
    return pipeline_result;
}

std::map<std::string, double> UnifiedMLPipeline::calculatePerformanceMetrics(
    const BarrierMLStrategy::PredictionResult& result,
    const FeatureExtractor::FeatureExtractionResult& features) {
    
    std::map<std::string, double> metrics;
    
    try {
        MetricsCalculator calc;
        
        const auto& portfolio = result.portfolio_result;
        metrics["total_return"] = portfolio.total_return;
        metrics["max_drawdown"] = portfolio.max_drawdown;
        metrics["total_trades"] = static_cast<double>(portfolio.total_trades);
        metrics["win_rate"] = portfolio.win_rate;
        
        if (!features.labels.empty() && result.predictions.size() == features.labels.size()) {
            std::vector<int> y_pred_int;
            for (double pred : result.predictions) {
                y_pred_int.push_back(pred > 0.5 ? 1 : 0);
            }
            
            metrics["accuracy"] = calc.calculateAccuracy(features.labels, y_pred_int);
            metrics["precision"] = calc.calculatePrecision(features.labels, y_pred_int);
            metrics["recall"] = calc.calculateRecall(features.labels, y_pred_int);
            metrics["f1_score"] = calc.calculateF1Score(features.labels, y_pred_int);
        }
        
        if (!features.labels_double.empty() && result.predictions.size() == features.labels_double.size()) {
            metrics["r2_score"] = calc.calculateR2Score(features.labels_double, result.predictions);
            metrics["mae"] = calc.calculateMAE(features.labels_double, result.predictions);
            metrics["rmse"] = calc.calculateRMSE(features.labels_double, result.predictions);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Warning: Metrics calculation failed: " << e.what() << std::endl;
    }
    
    return metrics;
}

} // namespace MLPipeline
