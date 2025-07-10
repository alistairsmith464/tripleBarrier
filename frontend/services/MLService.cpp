#include "MLService.h"
#include "../../backend/data/PreprocessedRow.h"
#include "../../backend/data/LabeledEvent.h"
#include "../../backend/data/FeatureExtractor.h"
#include "../../backend/ml/MLPipeline.h"
#include "../../backend/ml/PortfolioSimulator.h"
#include "../../backend/ml/MLSplits.h"
#include "../../backend/ml/MetricsCalculator.h"
#include "../../backend/ml/DataUtils.h"
#include "../config/VisualizationConfig.h"
#include <algorithm>
#include <cstdio>
#include <QStandardPaths>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <iostream>
#include <future>

// MLServiceImpl constructor
MLServiceImpl::MLServiceImpl() 
    : feature_service_(std::make_unique<FeatureServiceImpl>())
    , model_service_(std::make_unique<ModelServiceImpl>())
    , portfolio_service_(std::make_unique<PortfolioServiceImpl>()) {
}

// FeatureServiceImpl implementation
FeatureExtractor::FeatureExtractionResult FeatureServiceImpl::extractFeaturesForClassification(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const QSet<QString>& selectedFeatures) {
    
    // Validate inputs
    if (rows.empty()) {
        throw std::runtime_error("Empty rows vector provided to feature extraction");
    }
    
    if (labeledEvents.empty()) {
        throw std::runtime_error("Empty labeled events provided to feature extraction");
    }
    
    if (selectedFeatures.empty()) {
        throw std::runtime_error("No features selected for extraction");
    }
    
    // Convert QSet to std::set
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
        // Use the actual backend FeatureExtractor
        auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, labeledEvents);
        
        // Validate the result
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
    
    // Validate inputs
    if (rows.empty()) {
        throw std::runtime_error("Empty rows vector provided to feature extraction");
    }
    
    if (labeledEvents.empty()) {
        throw std::runtime_error("Empty labeled events provided to feature extraction");
    }
    
    if (selectedFeatures.empty()) {
        throw std::runtime_error("No features selected for extraction");
    }
    
    // Convert QSet to std::set
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
        // Use the actual backend FeatureExtractor
        auto result = FeatureExtractor::extractFeaturesForRegression(features, rows, labeledEvents);
        
        // Validate the result
        if (result.features.empty()) {
            throw std::runtime_error("Feature extraction returned empty feature set");
        }
        
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Feature extraction failed: ") + e.what());
    }
}

QStringList FeatureServiceImpl::getAvailableFeatures() {
    // Get actual features from the backend feature mapping
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
    
    return QString(); // Empty string means valid
}

// ModelServiceImpl implementation
MLResults MLServiceImpl::runMLPipeline(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config) {
    
    MLResults results;
    
    try {
        // Validate configuration
        QString config_error = validateConfiguration(config);
        if (!config_error.isEmpty()) {
            results.errorMessage = config_error;
            results.success = false;
            return results;
        }
        
        // Extract features
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
        
        // Train model
        MLResults model_results = model_service_->trainModel(results.features, labeledEvents, config);
        if (!model_results.success) {
            return model_results;
        }
        
        // Copy model results
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
        
        // Copy portfolio results from the ML pipeline
        results.portfolioResult = model_results.portfolioResult;
        
        // Portfolio results are already included in model_results from the ML pipeline
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
    
    return QString(); // Valid configuration
}

MLConfig MLServiceImpl::getDefaultConfiguration() {
    MLConfig config;
    
    // Use actual feature names from the backend
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
    
    // Set default unified pipeline configuration
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
        // Create MetricsCalculator instance
        MLPipeline::MetricsCalculator metricsCalc;
        
        // Basic classification metrics
        results.accuracy = metricsCalc.calculateAccuracy(y_true, y_pred);
        results.precision = metricsCalc.calculatePrecision(y_true, y_pred);
        results.recall = metricsCalc.calculateRecall(y_true, y_pred);
        results.f1_score = metricsCalc.calculateF1Score(y_true, y_pred);
        
        // Confusion matrix
        results.confusion_matrix = metricsCalc.calculateConfusionMatrix(y_true, y_pred);
        
        // AUC-ROC if probabilities are provided
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
        // Create MetricsCalculator instance
        MLPipeline::MetricsCalculator metricsCalc;
        
        // Regression metrics
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

// ModelServiceImpl implementation
MLResults ModelServiceImpl::trainModel(
    const FeatureExtractor::FeatureExtractionResult& features,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config) {
    
    MLResults results;
    try {
        // Check if we have valid feature data
        std::cout << "[DEBUG] MLService::trainModel called" << std::endl;
        std::cout << "  - Features size: " << features.features.size() << std::endl;
        std::cout << "  - Labels size: " << features.labels.size() << std::endl;
        std::cout << "  - Labels_double size: " << features.labels_double.size() << std::endl;
        std::cout << "  - Labeled events size: " << labeledEvents.size() << std::endl;
        
        // Check for valid feature data based on the mode (TTBM uses regression, others use classification)
        bool hasValidData = !features.features.empty() && 
                           (config.useTTBM ? !features.labels_double.empty() : !features.labels.empty());
        
        if (!hasValidData) {
            std::cout << "  - ERROR: No valid feature data available for " 
                     << (config.useTTBM ? "regression (TTBM)" : "classification") << " mode!" << std::endl;
            results.errorMessage = "No valid feature data available";
            results.success = false;
            return results;
        }
        
        // Convert unified config to the actual pipeline config
        MLPipeline::PipelineConfig pipelineConfig;
        pipelineConfig.test_size = config.pipelineConfig.test_size;
        pipelineConfig.val_size = config.pipelineConfig.val_size;
        pipelineConfig.n_rounds = config.pipelineConfig.n_rounds;
        pipelineConfig.max_depth = config.pipelineConfig.max_depth;
        pipelineConfig.nthread = config.pipelineConfig.nthread;
        
        // Set objective based on whether TTBM is used
        if (config.useTTBM) {
            pipelineConfig.objective = "reg:squarederror";  // Regression for TTBM
        } else {
            pipelineConfig.objective = config.pipelineConfig.objective;  // Classification
        }
        
        // Run the actual ML pipeline with the correct parameters
        MLPipeline::PipelineResult pipelineResults;
        if (config.useTTBM) {
            // For TTBM, use regression pipeline
            auto regressionResults = MLPipeline::runPipelineRegression(
                features.features, features.labels_double, features.returns, pipelineConfig);
            
            // Convert regression results to standard format
            results.predictions = regressionResults.predictions;
            results.portfolioResult.starting_capital = regressionResults.portfolio.starting_capital;
            results.portfolioResult.final_value = regressionResults.portfolio.final_capital;  // final_capital -> final_value
            results.portfolioResult.total_return = regressionResults.portfolio.total_return;
            results.portfolioResult.annualized_return = regressionResults.portfolio.annualized_return;
            results.portfolioResult.max_drawdown = regressionResults.portfolio.max_drawdown;
            results.portfolioResult.sharpe_ratio = regressionResults.portfolio.sharpe_ratio;
            results.portfolioResult.total_trades = regressionResults.portfolio.total_trades;
            results.portfolioResult.win_rate = regressionResults.portfolio.win_rate;
            
            // Add detailed trade statistics
            results.portfolioResult.winning_trades = static_cast<int>(regressionResults.portfolio.total_trades * regressionResults.portfolio.win_rate);
            results.portfolioResult.losing_trades = regressionResults.portfolio.total_trades - results.portfolioResult.winning_trades;
            results.portfolioResult.avg_trade_return = regressionResults.portfolio.total_return / (regressionResults.portfolio.total_trades > 0 ? regressionResults.portfolio.total_trades : 1);
            results.portfolioResult.best_trade = 0.0;
            results.portfolioResult.worst_trade = 0.0;
        } else {
            // For classification, use the standard pipeline
            pipelineResults = MLPipeline::runPipeline(
                features.features, features.labels, features.returns, pipelineConfig);
            
            // Convert results
            results.predictions.clear();
            for (int pred : pipelineResults.predictions) {
                results.predictions.push_back(static_cast<double>(pred));
            }
            results.prediction_probabilities = pipelineResults.probabilities;
            
            // Portfolio results - fix field name mapping
            results.portfolioResult.starting_capital = pipelineResults.portfolio.starting_capital;
            results.portfolioResult.final_value = pipelineResults.portfolio.final_capital;  // final_capital -> final_value
            results.portfolioResult.total_return = pipelineResults.portfolio.total_return;
            results.portfolioResult.annualized_return = pipelineResults.portfolio.annualized_return;
            results.portfolioResult.max_drawdown = pipelineResults.portfolio.max_drawdown;
            results.portfolioResult.sharpe_ratio = pipelineResults.portfolio.sharpe_ratio;
            results.portfolioResult.total_trades = pipelineResults.portfolio.total_trades;
            results.portfolioResult.win_rate = pipelineResults.portfolio.win_rate;
            
            std::cout << "DEBUG: Copied portfolio results:" << std::endl;
            std::cout << "  Starting capital: $" << results.portfolioResult.starting_capital << std::endl;
            std::cout << "  Final value: $" << results.portfolioResult.final_value << std::endl;
            std::cout << "  Total trades: " << results.portfolioResult.total_trades << std::endl;
            std::cout << "  Total return: " << results.portfolioResult.total_return << std::endl;
            
            // Add detailed trade statistics that might be missing
            results.portfolioResult.winning_trades = static_cast<int>(pipelineResults.portfolio.total_trades * pipelineResults.portfolio.win_rate);
            results.portfolioResult.losing_trades = pipelineResults.portfolio.total_trades - results.portfolioResult.winning_trades;
            results.portfolioResult.avg_trade_return = pipelineResults.portfolio.total_return / (pipelineResults.portfolio.total_trades > 0 ? pipelineResults.portfolio.total_trades : 1);
            results.portfolioResult.best_trade = 0.0;  // These might not be available from ML pipeline
            results.portfolioResult.worst_trade = 0.0;
        }
        
        results.success = true;
        
        // Calculate detailed metrics using the actual data
        if (results.success && !results.predictions.empty() && !features.labels.empty()) {
            try {
                // Convert predictions to int for classification metrics
                std::vector<int> y_pred_int;
                for (double pred : results.predictions) {
                    y_pred_int.push_back(pred > 0.5 ? 1 : 0);
                }
                
                // Calculate metrics using backend MetricsCalculator
                MLPipeline::MetricsCalculator metricsCalc;
                results.accuracy = metricsCalc.calculateAccuracy(features.labels, y_pred_int);
                results.precision = metricsCalc.calculatePrecision(features.labels, y_pred_int);
                results.recall = metricsCalc.calculateRecall(features.labels, y_pred_int);
                results.f1_score = metricsCalc.calculateF1Score(features.labels, y_pred_int);
                results.confusion_matrix = metricsCalc.calculateConfusionMatrix(features.labels, y_pred_int);
                
                // AUC-ROC if probabilities available
                if (!results.prediction_probabilities.empty()) {
                    std::vector<double> y_true_double(features.labels.begin(), features.labels.end());
                    results.auc_roc = metricsCalc.calculateAUCROC(y_true_double, results.prediction_probabilities);
                }
                
                // Regression metrics for TTBM
                if (config.useTTBM && !features.labels_double.empty()) {
                    results.r2_score = metricsCalc.calculateR2Score(features.labels_double, results.predictions);
                    results.mae = metricsCalc.calculateMAE(features.labels_double, results.predictions);
                    results.rmse = metricsCalc.calculateRMSE(features.labels_double, results.predictions);
                    results.mape = metricsCalc.calculateMAPE(features.labels_double, results.predictions);
                }
            } catch (const std::exception& e) {
                // If metrics calculation fails, log but don't fail the whole operation
                results.errorMessage = QString("Metrics calculation warning: %1").arg(e.what());
            }
        }
        
    } catch (const std::exception& e) {
        results.errorMessage = QString("Model training failed: %1").arg(e.what());
        results.success = false;
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
    // Stub implementation
    Q_UNUSED(modelPath)
    Q_UNUSED(configPath)
    return false; // Not implemented yet
}

bool ModelServiceImpl::loadModel(const QString& modelPath) {
    // Stub implementation
    Q_UNUSED(modelPath)
    return false; // Not implemented yet
}

QStringList ModelServiceImpl::getAvailableModels() {
    // Stub implementation
    return QStringList{"XGBoost", "Random Forest", "Linear Regression"};
}

// PortfolioServiceImpl implementation
MLPipeline::PortfolioResults PortfolioServiceImpl::runSimulation(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const std::vector<double>& predictions,
    bool useTTBM) {
    
    try {
        // Extract returns from the preprocessed rows
        std::vector<double> returns;
        for (const auto& row : rows) {
            // Use the log_return field that's already computed in PreprocessedRow
            returns.push_back(row.log_return);
        }
        
        // Use the actual backend portfolio simulator
        MLPipeline::PortfolioConfig config;
        config.starting_capital = 10000.0;
        config.max_position_pct = 0.1;
        config.position_threshold = 0.01;
        config.hard_barrier_position_pct = 0.05;
        config.trading_days_per_year = 252.0;
        config.max_trade_decisions_logged = 100;
        
        // Run the actual portfolio simulation
        auto simulation = MLPipeline::simulate_portfolio(
            predictions, returns, !useTTBM, config);
        
        // Convert to PortfolioResults format
        MLPipeline::PortfolioResults results;
        results.starting_capital = simulation.starting_capital;
        results.final_value = simulation.final_capital;  // final_capital -> final_value
        results.total_return = simulation.total_return;
        results.annualized_return = simulation.annualized_return;
        results.max_drawdown = simulation.max_drawdown;
        results.sharpe_ratio = simulation.sharpe_ratio;
        results.total_trades = simulation.total_trades;
        results.win_rate = simulation.win_rate;
        
        return results;
        
    } catch (const std::exception& e) {
        // Return a results object with error indication
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
    
    // Determine strategy type from string
    bool useTTBM = strategy.contains("TTBM", Qt::CaseInsensitive);
    
    // Use the actual simulation with the specified strategy
    return runSimulation(rows, labeledEvents, predictions, useTTBM);
}
