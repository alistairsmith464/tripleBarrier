#include "BarrierMLStrategy.h"
#include "MLSplits.h"
#include "ModelUtils.h"
#include "MetricsCalculator.h"
#include "DataUtils.h"
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
    
    return simulate_portfolio(trading_signals, returns, false, portfolio_config);
}

FeatureExtractor::FeatureExtractionResult HardBarrierStrategy::extractFeatures(
    const std::set<std::string>& selectedFeatures,
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents) {
    
    std::cout << "[DEBUG] HardBarrierStrategy: Extracting features for classification" << std::endl;
    return FeatureExtractor::extractFeaturesForClassification(selectedFeatures, rows, labeledEvents);
}

BarrierMLStrategy::PredictionResult HardBarrierStrategy::trainAndPredict(
    const FeatureExtractor::FeatureExtractionResult& features,
    const std::vector<double>& returns,
    const TrainingConfig& config) {
    
    PredictionResult result;
    
    try {
        std::cout << "[DEBUG] HardBarrierStrategy: Training classification model" << std::endl;
        std::cout << "  - Training samples: " << features.features.size() << std::endl;
        std::cout << "  - Classification labels: " << features.labels.size() << std::endl;
        
        if (features.features.empty() || features.labels.empty()) {
            result.error_message = "No valid classification data available";
            return result;
        }
        
        DataProcessor::CleaningOptions cleaning_opts;
        cleaning_opts.remove_nan = true;
        cleaning_opts.remove_inf = true;
        cleaning_opts.remove_outliers = true;
        cleaning_opts.log_cleaning = true;
        
        auto [X_clean, y_clean, returns_clean] = DataProcessor::cleanData(
            features.features, features.labels, returns, cleaning_opts);
        
        auto [train_idx, val_idx, test_idx] = createTrainValTestSplits(X_clean.size(), config);
        
        auto X_train = toFloatMatrix(select_rows(X_clean, train_idx));
        auto y_train = toFloatVecInt(select_rows(y_clean, train_idx));
        
        std::vector<size_t> eval_idx = val_idx.empty() ? test_idx : val_idx;
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
        
        XGBoostModel model;
        model.fit(X_train, y_train, model_config);
        
        auto y_pred = model.predict(X_eval); 
        auto y_prob = model.predict_proba(X_eval); 
        
        std::cout << "[DEBUG] HardBarrierStrategy: Predictions completed" << std::endl;
        std::cout << "  - Predictions: " << y_pred.size() << std::endl;
        std::cout << "  - Probabilities: " << y_prob.size() << std::endl;
        
        result.predictions.assign(y_pred.begin(), y_pred.end());
        result.confidence_scores.assign(y_prob.begin(), y_prob.end());
        
        result.trading_signals = convertClassificationToTradingSignals(y_pred, 
            std::vector<double>(y_prob.begin(), y_prob.end()));
        
        result.portfolio_result = runPortfolioSimulation(
            result.trading_signals, returns_eval);
        
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Hard barrier training failed: ") + e.what();
        result.success = false;
    }
    
    return result;
}

std::vector<double> HardBarrierStrategy::convertClassificationToTradingSignals(
    const std::vector<int>& predictions,
    const std::vector<double>& probabilities) {
    
    std::vector<double> signals;
    signals.reserve(predictions.size());
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (i < probabilities.size()) {
            double signal = probabilities[i];
            if (predictions[i] == 0) {
                signal = -signal; 
            }
            signals.push_back(signal);
        } else {
            signals.push_back(predictions[i] == 1 ? 1.0 : -1.0);
        }
    }
    
    std::cout << "[DEBUG] HardBarrier trading signals: ";
    for (size_t i = 0; i < std::min(size_t(10), signals.size()); ++i) {
        std::cout << signals[i] << " ";
    }
    std::cout << std::endl;
    
    return signals;
}

FeatureExtractor::FeatureExtractionResult TTBMStrategy::extractFeatures(
    const std::set<std::string>& selectedFeatures,
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents) {
    
    std::cout << "[DEBUG] TTBMStrategy: Extracting features for regression" << std::endl;
    return FeatureExtractor::extractFeaturesForRegression(selectedFeatures, rows, labeledEvents);
}

BarrierMLStrategy::PredictionResult TTBMStrategy::trainAndPredict(
    const FeatureExtractor::FeatureExtractionResult& features,
    const std::vector<double>& returns,
    const TrainingConfig& config) {
    
    PredictionResult result;
    
    try {
        std::cout << "[DEBUG] TTBMStrategy: Training regression model" << std::endl;
        std::cout << "  - Training samples: " << features.features.size() << std::endl;
        std::cout << "  - Regression labels: " << features.labels_double.size() << std::endl;
        
        if (features.features.empty() || features.labels_double.empty()) {
            result.error_message = "No valid regression data available";
            return result;
        }
        
        DataProcessor::CleaningOptions cleaning_opts;
        cleaning_opts.remove_nan = true;
        cleaning_opts.remove_inf = true;
        cleaning_opts.remove_outliers = true;
        cleaning_opts.log_cleaning = true;
        
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
            std::cout << "[DEBUG] Training label stats: [" << min_train << ", " << max_train 
                     << "], mean: " << mean_train << std::endl;
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
        
        std::cout << "[DEBUG] TTBMStrategy: Making predictions using predict_raw..." << std::endl;
        auto y_pred_raw = model.predict_raw(X_eval); 
        
        if (!y_pred_raw.empty()) {
            float min_pred = *std::min_element(y_pred_raw.begin(), y_pred_raw.end());
            float max_pred = *std::max_element(y_pred_raw.begin(), y_pred_raw.end());
            float sum_pred = std::accumulate(y_pred_raw.begin(), y_pred_raw.end(), 0.0f);
            float mean_pred = sum_pred / y_pred_raw.size();
            std::cout << "[DEBUG] Raw predictions stats: [" << min_pred << ", " << max_pred 
                     << "], mean: " << mean_pred << std::endl;
            std::cout << "[DEBUG] First 10 raw predictions: ";
            for (size_t i = 0; i < std::min(size_t(10), y_pred_raw.size()); ++i) {
                std::cout << y_pred_raw[i] << " ";
            }
            std::cout << std::endl;
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
    
    std::cout << "[DEBUG] TTBM trading signals (normalized): ";
    for (size_t i = 0; i < std::min(size_t(10), normalized.size()); ++i) {
        std::cout << normalized[i] << " ";
    }
    std::cout << std::endl;
    
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
        std::cout << "[DEBUG] UnifiedMLPipeline: Starting with strategy: ";
        
        auto strategy = BarrierMLStrategyFactory::createStrategy(config.strategy_type);
        pipeline_result.strategy_name = strategy->getStrategyName();
        
        std::cout << pipeline_result.strategy_name << std::endl;
        
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
        metrics["sharpe_ratio"] = portfolio.sharpe_ratio;
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
