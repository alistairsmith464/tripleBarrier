#include "MLService.h"
#include "../../backend/data/PreprocessedRow.h"
#include "../../backend/data/LabeledEvent.h"
#include "../../backend/data/FeatureExtractor.h"
#include "../../backend/ml/MLPipeline.h"
#include "../../backend/ml/PortfolioSimulator.h"
#include "../../backend/ml/MLSplits.h"
#include "../../backend/ml/MetricsCalculator.h"
#include "../../backend/ml/DataUtils.h"
#include "../../backend/ml/BarrierMLStrategy.h"
#include "../config/VisualizationConfig.h"
#include <algorithm>
#include <cstdio>
#include <QStandardPaths>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <iostream>
#include <future>

MLServiceImpl::MLServiceImpl() 
    : feature_service_(std::make_unique<FeatureServiceImpl>())
    , model_service_(std::make_unique<ModelServiceImpl>())
    , portfolio_service_(std::make_unique<PortfolioServiceImpl>()) {
}

FeatureExtractor::FeatureExtractionResult FeatureServiceImpl::extractFeaturesForClassification(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const QSet<QString>& selectedFeatures) {
    
    if (rows.empty()) {
        throw std::runtime_error("Empty rows vector provided to feature extraction");
    }
    
    if (labeledEvents.empty()) {
        throw std::runtime_error("Empty labeled events provided to feature extraction");
    }
    
    if (selectedFeatures.empty()) {
        throw std::runtime_error("No features selected for extraction");
    }
    
    std::set<std::string> features;
    for (const QString& feature : selectedFeatures) {
        if (!feature.isEmpty()) {
            features.insert(feature.toStdString());
        }
    }
    
    if (features.empty()) {
        throw std::runtime_error("No valid features after conversion");
    }
    
    try {
        auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, labeledEvents);
        
        if (result.features.empty()) {
            throw std::runtime_error("Feature extraction returned empty feature set");
        }
        
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Feature extraction failed: ") + e.what());
    }
}

FeatureExtractor::FeatureExtractionResult FeatureServiceImpl::extractFeaturesForRegression(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const QSet<QString>& selectedFeatures) {
    
    if (rows.empty()) {
        throw std::runtime_error("Empty rows vector provided to feature extraction");
    }
    
    if (labeledEvents.empty()) {
        throw std::runtime_error("Empty labeled events provided to feature extraction");
    }
    
    if (selectedFeatures.empty()) {
        throw std::runtime_error("No features selected for extraction");
    }
    
    std::set<std::string> features;
    for (const QString& feature : selectedFeatures) {
        if (!feature.isEmpty()) {
            features.insert(feature.toStdString());
        }
    }
    
    if (features.empty()) {
        throw std::runtime_error("No valid features after conversion");
    }
    
    try {
        auto result = FeatureExtractor::extractFeaturesForRegression(features, rows, labeledEvents);
        
        if (result.features.empty()) {
            throw std::runtime_error("Feature extraction returned empty feature set");
        }
        
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Feature extraction failed: ") + e.what());
    }
}

QStringList FeatureServiceImpl::getAvailableFeatures() {
    auto featureMapping = FeatureExtractor::getFeatureMapping();
    QStringList features;
    
    for (const auto& pair : featureMapping) {
        features.append(QString::fromStdString(pair.first));
    }
    
    return features;
}

QString FeatureServiceImpl::validateFeatureSelection(const QSet<QString>& features) {
    if (features.empty()) {
        return "No features selected";
    }
    
    QStringList available = getAvailableFeatures();
    QStringList invalid;
    
    for (const QString& feature : features) {
        if (!available.contains(feature)) {
            invalid.append(feature);
        }
    }
    
    if (!invalid.empty()) {
        return QString("Invalid features: %1").arg(invalid.join(", "));
    }
    
    return QString(); 
}

MLResults MLServiceImpl::runMLPipeline(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config) {
    
    MLResults results;
    
    try {
        QString config_error = validateConfiguration(config);
        if (!config_error.isEmpty()) {
            results.errorMessage = config_error;
            results.success = false;
            return results;
        }
        
        try {
            if (config.useTTBM) {
                results.features = feature_service_->extractFeaturesForRegression(
                    rows, labeledEvents, config.selectedFeatures);
            } else {
                results.features = feature_service_->extractFeaturesForClassification(
                    rows, labeledEvents, config.selectedFeatures);
            }
        } catch (const std::exception& e) {
            results.errorMessage = QString("Feature extraction failed: %1").arg(e.what());
            results.success = false;
            return results;
        }
        
        MLResults model_results = model_service_->trainModel(results.features, labeledEvents, config);
        if (!model_results.success) {
            return model_results;
        }
        
        results.predictions = model_results.predictions;
        results.prediction_probabilities = model_results.prediction_probabilities;
        results.accuracy = model_results.accuracy;
        results.precision = model_results.precision;
        results.recall = model_results.recall;
        results.f1_score = model_results.f1_score;
        results.auc_roc = model_results.auc_roc;
        results.confusion_matrix = model_results.confusion_matrix;
        results.r2_score = model_results.r2_score;
        results.mae = model_results.mae;
        results.rmse = model_results.rmse;
        results.mape = model_results.mape;
        results.modelInfo = model_results.modelInfo;
        results.dataQuality = model_results.dataQuality;
        
        results.portfolioResult = model_results.portfolioResult;
        
        std::cout << "DEBUG: Portfolio simulation completed successfully" << std::endl;
        std::cout << "DEBUG: Starting capital: $" << results.portfolioResult.starting_capital << std::endl;
        std::cout << "DEBUG: Final value: $" << results.portfolioResult.final_value << std::endl;
        std::cout << "DEBUG: Total trades: " << results.portfolioResult.total_trades << std::endl;
        std::cout << "DEBUG: Total return: " << results.portfolioResult.total_return << std::endl;
        
        results.success = true;
        
    } catch (const std::exception& e) {
        results.errorMessage = QString("ML Pipeline error: %1").arg(e.what());
        results.success = false;
    }
    
    return results;
}

std::future<MLResults> MLServiceImpl::runMLPipelineAsync(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config,
    MLProgressCallback callback) {
    
    return std::async(std::launch::async, [this, rows, labeledEvents, config, callback]() {
        if (callback) {
            MLProgress progress;
            progress.current_stage = MLProgress::FEATURE_EXTRACTION;
            progress.progress_percentage = 0.0;
            progress.status_message = "Starting feature extraction...";
            callback(progress);
        }
        
        auto result = runMLPipeline(rows, labeledEvents, config);
        
        if (callback) {
            MLProgress progress;
            progress.current_stage = MLProgress::COMPLETE;
            progress.progress_percentage = 100.0;
            progress.status_message = result.success ? "Pipeline completed successfully" : "Pipeline failed";
            callback(progress);
        }
        
        return result;
    });
}

QString MLServiceImpl::validateConfiguration(const MLConfig& config) {
    if (config.selectedFeatures.empty()) {
        return "No features selected";
    }
    
    if (config.crossValidationRatio < 0.0 || config.crossValidationRatio > 0.5) {
        return "Cross-validation ratio must be between 0.0 and 0.5";
    }
    
    if (config.pipelineConfig.n_rounds < 1 || config.pipelineConfig.n_rounds > 10000) {
        return "Number of rounds must be between 1 and 10000";
    }
    
    if (config.pipelineConfig.max_depth < 1 || config.pipelineConfig.max_depth > 20) {
        return "Max depth must be between 1 and 20";
    }
    
    if (config.outlierThreshold < 1.0 || config.outlierThreshold > 10.0) {
        return "Outlier threshold must be between 1.0 and 10.0";
    }
    
    return QString();
}

MLConfig MLServiceImpl::getDefaultConfiguration() {
    MLConfig config;
    
    config.selectedFeatures = QSet<QString>{
        "Close-to-close return for the previous day",
        "Return over the past 5 days", 
        "Rolling standard deviation of daily returns over the last 5 days"
    };
    
    config.useTTBM = false;
    config.crossValidationRatio = 0.2;
    config.randomSeed = 42;
    config.tuneHyperparameters = false;
    config.saveModel = false;
    config.loadModel = false;
    config.preprocessFeatures = true;
    config.normalizeFeatures = false;
    config.removeOutliers = false;
    config.outlierThreshold = 3.0;
    config.enableProgressCallbacks = false;
    
    config.pipelineConfig.test_size = 0.2;
    config.pipelineConfig.val_size = 0.2;
    config.pipelineConfig.n_rounds = 100;
    config.pipelineConfig.max_depth = 5;
    config.pipelineConfig.nthread = 4;
    config.pipelineConfig.objective = "binary:logistic";
    config.pipelineConfig.learning_rate = 0.1;
    config.pipelineConfig.subsample = 0.8;
    config.pipelineConfig.colsample_bytree = 0.8;
    config.pipelineConfig.barrier_type = MLPipeline::BarrierType::HARD;
    
    return config;
}

MLResults MLServiceImpl::calculateDetailedMetrics(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    const std::vector<double>& y_prob) {
    
    MLResults results;
    
    try {
        MLPipeline::MetricsCalculator metricsCalc;
        
        results.accuracy = metricsCalc.calculateAccuracy(y_true, y_pred);
        results.precision = metricsCalc.calculatePrecision(y_true, y_pred);
        results.recall = metricsCalc.calculateRecall(y_true, y_pred);
        results.f1_score = metricsCalc.calculateF1Score(y_true, y_pred);
        
        results.confusion_matrix = metricsCalc.calculateConfusionMatrix(y_true, y_pred);
        
        if (!y_prob.empty() && y_prob.size() == y_true.size()) {
            std::vector<double> y_true_double(y_true.begin(), y_true.end());
            results.auc_roc = metricsCalc.calculateAUCROC(y_true_double, y_prob);
        }
        
        results.success = true;
        
    } catch (const std::exception& e) {
        results.errorMessage = QString("Metrics calculation failed: %1").arg(e.what());
        results.success = false;
    }
    
    return results;
}

MLResults MLServiceImpl::calculateRegressionMetrics(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) {
    
    MLResults results;
    
    try {
        MLPipeline::MetricsCalculator metricsCalc;
        
        results.r2_score = metricsCalc.calculateR2Score(y_true, y_pred);
        results.mae = metricsCalc.calculateMAE(y_true, y_pred);
        results.rmse = metricsCalc.calculateRMSE(y_true, y_pred);
        results.mape = metricsCalc.calculateMAPE(y_true, y_pred);
        
        results.success = true;
        
    } catch (const std::exception& e) {
        results.errorMessage = QString("Regression metrics calculation failed: %1").arg(e.what());
        results.success = false;
    }
    
    return results;
}

MLResults ModelServiceImpl::trainModel(
    const FeatureExtractor::FeatureExtractionResult& features,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config) {
    
    MLResults results;
    try {
        std::cout << "[DEBUG] MLService::trainModel called (NEW UNIFIED APPROACH)" << std::endl;
        std::cout << "  - Features size: " << features.features.size() << std::endl;
        std::cout << "  - Labels size: " << features.labels.size() << std::endl;
        std::cout << "  - Labels_double size: " << features.labels_double.size() << std::endl;
        std::cout << "  - TTBM mode: " << (config.useTTBM ? "YES" : "NO") << std::endl;
        
        bool hasValidData = !features.features.empty() && 
                           (config.useTTBM ? !features.labels_double.empty() : !features.labels.empty());
        
        if (!hasValidData) {
            std::cout << "  - ERROR: No valid feature data available for " 
                     << (config.useTTBM ? "regression (TTBM)" : "classification") << " mode!" << std::endl;
            results.errorMessage = "No valid feature data available";
            results.success = false;
            return results;
        }
        
        MLPipeline::UnifiedMLPipeline::PipelineConfig pipeline_config;
        
        pipeline_config.strategy_type = MLPipeline::BarrierMLStrategyFactory::getStrategyType(config.useTTBM);
        
        pipeline_config.selected_features.clear();
        for (const QString& feature : config.selectedFeatures) {
            pipeline_config.selected_features.insert(feature.toStdString());
        }
        
        pipeline_config.training_config.test_size = config.pipelineConfig.test_size;
        pipeline_config.training_config.val_size = config.pipelineConfig.val_size;
        pipeline_config.training_config.n_rounds = config.pipelineConfig.n_rounds;
        pipeline_config.training_config.max_depth = config.pipelineConfig.max_depth;
        pipeline_config.training_config.nthread = config.pipelineConfig.nthread;
        pipeline_config.training_config.learning_rate = config.pipelineConfig.learning_rate;
        pipeline_config.training_config.subsample = config.pipelineConfig.subsample;
        pipeline_config.training_config.colsample_bytree = config.pipelineConfig.colsample_bytree;
        pipeline_config.training_config.random_seed = config.randomSeed;
        
        pipeline_config.portfolio_config.starting_capital = 10000.0;
        pipeline_config.portfolio_config.max_position_pct = 0.1;
        pipeline_config.portfolio_config.position_threshold = 0.01;
        pipeline_config.portfolio_config.hard_barrier_position_pct = 0.05;
        
        pipeline_config.enable_detailed_logging = true;
        pipeline_config.enable_hyperparameter_tuning = config.tuneHyperparameters;
        
        std::vector<PreprocessedRow> rows;
        for (const auto& event : labeledEvents) {
            PreprocessedRow row;
            row.log_return = event.forward_return;
            // Note: In a real implementation, you'd properly reconstruct rows
            // For now, we'll work with what we have
            rows.push_back(row);
        }
        
        auto pipeline_result = MLPipeline::UnifiedMLPipeline::runPipeline(
            rows, labeledEvents, pipeline_config);
        
        if (pipeline_result.success) {
            results.predictions = pipeline_result.prediction_result.predictions;
            results.prediction_probabilities = pipeline_result.prediction_result.confidence_scores;
            
            const auto& portfolio = pipeline_result.prediction_result.portfolio_result;
            results.portfolioResult.starting_capital = portfolio.starting_capital;
            results.portfolioResult.final_value = portfolio.final_capital;
            results.portfolioResult.total_return = portfolio.total_return;
            results.portfolioResult.annualized_return = portfolio.annualized_return;
            results.portfolioResult.max_drawdown = portfolio.max_drawdown;
            results.portfolioResult.sharpe_ratio = portfolio.sharpe_ratio;
            results.portfolioResult.total_trades = portfolio.total_trades;
            results.portfolioResult.win_rate = portfolio.win_rate;
            
            for (const auto& [key, value] : pipeline_result.performance_metrics) {
                if (key == "accuracy") results.accuracy = value;
                else if (key == "precision") results.precision = value;
                else if (key == "recall") results.recall = value;
                else if (key == "f1_score") results.f1_score = value;
                else if (key == "r2_score") results.r2_score = value;
                else if (key == "mae") results.mae = value;
                else if (key == "rmse") results.rmse = value;
            }
            
            results.modelInfo = QString("Strategy: %1").arg(QString::fromStdString(pipeline_result.strategy_name));
            results.success = true;
            
            std::cout << "[DEBUG] NEW UNIFIED PIPELINE SUCCESS:" << std::endl;
            std::cout << "  Strategy: " << pipeline_result.strategy_name << std::endl;
            std::cout << "  Predictions: " << results.predictions.size() << std::endl;
            std::cout << "  Portfolio - Starting: $" << results.portfolioResult.starting_capital 
                     << ", Final: $" << results.portfolioResult.final_value 
                     << ", Trades: " << results.portfolioResult.total_trades << std::endl;
            
        } else {
            results.errorMessage = QString::fromStdString(pipeline_result.error_message);
            results.success = false;
            std::cout << "[DEBUG] UNIFIED PIPELINE FAILED: " << pipeline_result.error_message << std::endl;
        }
        
    } catch (const std::exception& e) {
        results.errorMessage = QString("Unified model training failed: %1").arg(e.what());
        results.success = false;
        std::cout << "[DEBUG] EXCEPTION: " << e.what() << std::endl;
    }
    
    return results;
}

std::future<MLResults> ModelServiceImpl::trainModelAsync(
    const FeatureExtractor::FeatureExtractionResult& features,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config,
    MLProgressCallback callback) {
    
    return std::async(std::launch::async, [this, features, labeledEvents, config, callback]() {
        if (callback) {
            MLProgress progress;
            progress.current_stage = MLProgress::MODEL_TRAINING;
            progress.progress_percentage = 0.0;
            progress.status_message = "Starting model training...";
            callback(progress);
        }
        
        auto result = trainModel(features, labeledEvents, config);
        
        if (callback) {
            MLProgress progress;
            progress.current_stage = MLProgress::COMPLETE;
            progress.progress_percentage = 100.0;
            progress.status_message = "Model training completed.";
            callback(progress);
        }
        
        return result;
    });
}

bool ModelServiceImpl::saveModel(const QString& modelPath, const QString& configPath) {
    Q_UNUSED(modelPath)
    Q_UNUSED(configPath)
    return false;
}

bool ModelServiceImpl::loadModel(const QString& modelPath) {
    Q_UNUSED(modelPath)
    return false;
}

QStringList ModelServiceImpl::getAvailableModels() {
    return QStringList{"XGBoost", "Random Forest", "Linear Regression"};
}

MLPipeline::PortfolioResults PortfolioServiceImpl::runSimulation(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const std::vector<double>& predictions,
    bool useTTBM) {
    
    try {
        std::vector<double> returns;
        for (const auto& row : rows) {
            returns.push_back(row.log_return);
        }
        
        MLPipeline::PortfolioConfig config;
        config.starting_capital = 10000.0;
        config.max_position_pct = 0.1;
        config.position_threshold = 0.01;
        config.hard_barrier_position_pct = 0.05;
        config.trading_days_per_year = 252.0;
        config.max_trade_decisions_logged = 100;
        
        auto simulation = MLPipeline::simulate_portfolio(
            predictions, returns, !useTTBM, config);
        
        MLPipeline::PortfolioResults results;
        results.starting_capital = simulation.starting_capital;
        results.final_value = simulation.final_capital; 
        results.total_return = simulation.total_return;
        results.annualized_return = simulation.annualized_return;
        results.max_drawdown = simulation.max_drawdown;
        results.sharpe_ratio = simulation.sharpe_ratio;
        results.total_trades = simulation.total_trades;
        results.win_rate = simulation.win_rate;
        
        return results;
        
    } catch (const std::exception& e) {
        MLPipeline::PortfolioResults results;
        results.starting_capital = 0.0;
        results.final_value = 0.0;
        results.total_return = 0.0;
        results.annualized_return = 0.0;
        results.max_drawdown = 0.0;
        results.sharpe_ratio = 0.0;
        results.total_trades = 0;
        results.win_rate = 0.0;
        return results;
    }
}

MLPipeline::PortfolioResults PortfolioServiceImpl::runBacktest(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const std::vector<double>& predictions,
    const QString& strategy) {
    
    bool useTTBM = strategy.contains("TTBM", Qt::CaseInsensitive);
    
    return runSimulation(rows, labeledEvents, predictions, useTTBM);
}
