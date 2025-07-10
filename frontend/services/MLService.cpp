#include "MLService.h"
#include "../../backend/data/PreprocessedRow.h"
#include "../../backend/data/LabeledEvent.h"
#include "../../backend/data/FeatureExtractor.h"
#include "../../backend/data/PortfolioSimulator.h"
#include "../../backend/ml/MLPipeline.h"
#include "../../backend/ml/MLSplits.h"
#include "../../backend/ml/MetricsCalculator.h"
#include "../../backend/ml/DataUtils.h"
#include "../config/VisualizationConfig.h"
#include <algorithm>
#include <cstdio>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <QStandardPaths>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>

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
        return FeatureExtractor::extractFeaturesForClassification(features, rows, labeledEvents);
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
        return FeatureExtractor::extractFeaturesForRegression(features, rows, labeledEvents);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Feature extraction failed: ") + e.what());
    }
}

QStringList FeatureServiceImpl::getAvailableFeatures() {
    // This would be populated from a feature registry
    return QStringList{
        "price_change", "volume_change", "volatility", "momentum",
        "rsi", "macd", "bollinger_bands", "moving_average_crossover",
        "volume_profile", "order_flow", "bid_ask_spread"
    };
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
        
        // Run portfolio simulation
        try {
            results.portfolioResult = portfolio_service_->runSimulation(
                rows, labeledEvents, results.predictions, config.useTTBM);
        } catch (const std::exception& e) {
            results.errorMessage = QString("Portfolio simulation error: %1").arg(e.what());
            results.success = false;
            return results;
        }
        
        results.success = true;
        
    } catch (const std::exception& e) {
        results.errorMessage = QString("ML Pipeline error: %1").arg(e.what());
        results.success = false;
    }
    
    return results;
}

QFuture<MLResults> MLServiceImpl::runMLPipelineAsync(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config,
    MLProgressCallback callback) {
    
    return QtConcurrent::run([this, rows, labeledEvents, config, callback]() {
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
    config.selectedFeatures = QSet<QString>{"price_change", "volume_change", "volatility"};
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
