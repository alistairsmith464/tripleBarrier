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
#include "../../backend/utils/Exceptions.h"
#include "../../backend/utils/ErrorHandling.h"
#include "../config/VisualizationConfig.h"
#include "../utils/ValidationFramework.h"
#include "../utils/ErrorHandlingStrategy.h"
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
    
    using namespace ValidationFramework;
    ErrorHandlingStrategy::setMode(ErrorHandlingStrategy::Mode::MIXED);
}

void FeatureServiceImpl::validateEventAlignment(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    ValidationFramework::ValidationAccumulator& accumulator) {
    std::set<std::string> rowTimestamps;
    for (const auto& row : rows) {
        rowTimestamps.insert(row.timestamp);
    }
    size_t matchedEvents = 0;
    for (const auto& event : labeledEvents) {
        if (rowTimestamps.count(event.entry_time)) {
            matchedEvents++;
        }
    }
    if (matchedEvents != labeledEvents.size()) {
        std::cerr << "[FeatureServiceImpl] WARNING: Only " << matchedEvents << " out of " << labeledEvents.size() << " labeled events have matching data rows." << std::endl;
    }
    if (rows.size() == labeledEvents.size() && matchedEvents == labeledEvents.size()) {
        accumulator.addResult(ValidationFramework::CoreValidator::validateSizeMatch(rows, labeledEvents, "Data rows", "Labeled events"));
    }
}

FeatureExtractor::FeatureExtractionResult FeatureServiceImpl::extractFeaturesForClassification(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const QSet<QString>& selectedFeatures) {
    using namespace ValidationFramework;
    ValidationAccumulator accumulator;
    accumulator.addResult(DataValidator::validateDataRows(rows));
    accumulator.addResult(DataValidator::validateLabeledEvents(labeledEvents));
    accumulator.addResult(MLValidator::validateFeatureSelection(selectedFeatures));
    validateEventAlignment(rows, labeledEvents, accumulator);
    if (!accumulator.isValid()) {
        throw TripleBarrier::FeatureExtractionException(
            accumulator.getSummary().errorMessage.toStdString(),
            "Classification Feature Extraction"
        );
    }
    std::set<std::string> features;
    for (const QString& feature : selectedFeatures) {
        if (!feature.isEmpty()) {
            features.insert(feature.toStdString());
        }
    }
    if (features.empty()) {
        throw TripleBarrier::FeatureExtractionException(
            "No valid features after conversion",
            "Feature Format Conversion"
        );
    }
    try {
        auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, labeledEvents);
        if (result.features.empty()) {
            throw TripleBarrier::FeatureExtractionException(
                "Feature extraction returned empty feature set",
                "Feature Extraction Output"
            );
        }
        ValidationResult outputValidation = MLValidator::validateModelInputs(result.features, result.labels);
        if (!outputValidation.isValid) {
            throw TripleBarrier::FeatureExtractionException(
                outputValidation.errorMessage.toStdString(),
                "Feature Extraction Output Validation"
            );
        }
        return result;
    } catch (const TripleBarrier::BaseException& e) {
        throw;
    } catch (const std::exception& e) {
        auto converted = TripleBarrier::ExceptionUtils::convertException(e, "Feature Extraction");
        throw *converted;
    }
}

FeatureExtractor::FeatureExtractionResult FeatureServiceImpl::extractFeaturesForRegression(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const QSet<QString>& selectedFeatures) {
    using namespace ValidationFramework;
    ValidationAccumulator accumulator;
    accumulator.addResult(DataValidator::validateDataRows(rows));
    accumulator.addResult(DataValidator::validateLabeledEvents(labeledEvents));
    accumulator.addResult(MLValidator::validateFeatureSelection(selectedFeatures));
    validateEventAlignment(rows, labeledEvents, accumulator);
    if (!accumulator.isValid()) {
        throw TripleBarrier::FeatureExtractionException(
            accumulator.getSummary().errorMessage.toStdString(),
            "Regression Feature Extraction"
        );
    }
    std::set<std::string> features;
    for (const QString& feature : selectedFeatures) {
        if (!feature.isEmpty()) {
            features.insert(feature.toStdString());
        }
    }
    if (features.empty()) {
        throw TripleBarrier::FeatureExtractionException(
            "No valid features after conversion",
            "Feature Format Conversion"
        );
    }
    try {
        auto result = FeatureExtractor::extractFeaturesForRegression(features, rows, labeledEvents);
        if (result.features.empty()) {
            throw TripleBarrier::FeatureExtractionException(
                "Feature extraction returned empty feature set",
                "Feature Extraction Output"
            );
        }
        if (!result.labels_double.empty()) {
            ValidationResult outputValidation = MLValidator::validateModelInputs(result.features, result.labels_double);
            if (!outputValidation.isValid) {
                throw TripleBarrier::FeatureExtractionException(
                    outputValidation.errorMessage.toStdString(),
                    "Regression Feature Extraction Output Validation"
                );
            }
        }
        return result;
    } catch (const TripleBarrier::BaseException& e) {
        throw;
    } catch (const std::exception& e) {
        auto converted = TripleBarrier::ExceptionUtils::convertException(e, "Feature Extraction");
        throw *converted;
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
    using namespace ValidationFramework;
    
    ValidationResult result = MLValidator::validateFeatureSelection(features);
    
    if (!result.isValid) {
        return result.errorMessage;
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
    
    if (!result.warningMessage.isEmpty()) {
        return result.warningMessage;
    }
    
    return QString(); 
}

MLResults MLServiceImpl::runMLPipeline(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config) {
    
    using namespace ValidationFramework;
    
    MLResults results;
    
    ErrorHandlingStrategy::ErrorContext context(
        "ML Pipeline Execution",
        "MLService",
        ErrorHandlingStrategy::Severity::ERROR
    );
    
    try {
        ValidationAccumulator accumulator;
        accumulator.addResult(DataValidator::validateDataRows(rows));
        accumulator.addResult(DataValidator::validateLabeledEvents(labeledEvents));
        accumulator.addResult(MLValidator::validateMLConfig(config));
        
        std::set<std::string> rowTimestamps;
        for (const auto& row : rows) {
            rowTimestamps.insert(row.timestamp);
        }
        size_t matchedEvents = 0;
        for (const auto& event : labeledEvents) {
            if (rowTimestamps.count(event.entry_time)) {
                matchedEvents++;
            }
        }

        if (rows.size() == labeledEvents.size() && matchedEvents == labeledEvents.size()) {
            accumulator.addResult(CoreValidator::validateSizeMatch(rows, labeledEvents, "Data rows", "Labeled events"));
        }
        
        if (!accumulator.isValid()) {
            results.errorMessage = accumulator.getSummary().errorMessage;
            results.success = false;
            return results;
        }
        
        if (accumulator.hasWarnings()) {
            results.warningMessage = accumulator.getSummary().warningMessage;
        }
        
        auto featureExtractor = createValidatedFunction<FeatureExtractor::FeatureExtractionResult>([&]() {
            if (config.useTTBM) {
                return feature_service_->extractFeaturesForRegression(rows, labeledEvents, config.selectedFeatures);
            } else {
                return feature_service_->extractFeaturesForClassification(rows, labeledEvents, config.selectedFeatures);
            }
        }).withContext(ErrorHandlingStrategy::ErrorContext(
            "Feature Extraction",
            "FeatureService",
            ErrorHandlingStrategy::Severity::ERROR
        ));
        
        try {
            results.features = featureExtractor.build().execute();
        } catch (const std::exception& e) {
            results.errorMessage = QString("Feature extraction failed: %1").arg(e.what());
            results.success = false;
            return results;
        }
        
        auto modelTrainer = createValidatedFunction<MLResults>([&]() {
            return model_service_->trainModel(results.features, labeledEvents, config);
        }).withContext(ErrorHandlingStrategy::ErrorContext(
            "Model Training",
            "ModelService",
            ErrorHandlingStrategy::Severity::ERROR
        ));
        
        MLResults model_results;
        try {
            model_results = modelTrainer.build().execute();
        } catch (const std::exception& e) {
            results.errorMessage = QString("Model training failed: %1").arg(e.what());
            results.success = false;
            return results;
        }

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
        results.trade_log = model_results.trade_log;
        
        ValidationAccumulator postValidation;
        postValidation.addResult(CoreValidator::validateNotEmpty(results.predictions, "Predictions"));
        postValidation.addResult(CoreValidator::validateFinite(results.portfolioResult.total_return, "Total Return"));
        postValidation.addResult(CoreValidator::validateNonNegative(results.portfolioResult.total_trades, "Total Trades"));
        
        if (!postValidation.isValid()) {
            results.errorMessage = QString("Post-validation failed: %1").arg(postValidation.getSummary().errorMessage);
            results.success = false;
            return results;
        }
       
        results.success = true;
        
    } catch (const std::exception& e) {
        ErrorHandlingStrategy::handleException(e, context);
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
    using namespace ValidationFramework;
    
    ValidationResult result = MLValidator::validateMLConfig(config);
    
    if (!result.isValid) {
        return result.errorMessage;
    }
    
    if (!result.warningMessage.isEmpty()) {
        return result.warningMessage;
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
    
    using namespace TripleBarrier;
    
    MLResults results;
    
    try { 
        Validation::validateNotEmpty(features.features, "features");
        
        if (config.useTTBM) {
            Validation::validateNotEmpty(features.labels_double, "regression_labels");
            if (features.features.size() != features.labels_double.size()) {
                throw DataValidationException(
                    "Size mismatch: features (" + std::to_string(features.features.size()) + 
                    ") vs regression_labels (" + std::to_string(features.labels_double.size()) + ")"
                );
            }
        } else {
            Validation::validateNotEmpty(features.labels, "classification_labels");
            if (features.features.size() != features.labels.size()) {
                throw DataValidationException(
                    "Size mismatch: features (" + std::to_string(features.features.size()) + 
                    ") vs classification_labels (" + std::to_string(features.labels.size()) + ")"
                );
            }
        }
        
        Validation::validateRange(config.pipelineConfig.test_size, 0.0, 0.8, "test_size");
        Validation::validateRange(config.pipelineConfig.val_size, 0.0, 0.8, "val_size");
        Validation::validatePositive(config.pipelineConfig.learning_rate, "learning_rate");
        Validation::validateRange(config.pipelineConfig.subsample, 0.0, 1.0, "subsample");
        Validation::validateRange(config.pipelineConfig.colsample_bytree, 0.0, 1.0, "colsample_bytree");
        
        if (config.pipelineConfig.n_rounds <= 0) {
            throw HyperparameterException("n_rounds must be positive", "n_rounds");
        }
        if (config.pipelineConfig.max_depth <= 0) {
            throw HyperparameterException("max_depth must be positive", "max_depth");
        }
        if (config.pipelineConfig.nthread <= 0) {
            throw HyperparameterException("nthread must be positive", "nthread");
        }
        
        MLPipeline::BarrierMLStrategy::TrainingConfig training_config;
        training_config.test_size = config.pipelineConfig.test_size;
        training_config.val_size = config.pipelineConfig.val_size;
        training_config.n_rounds = config.pipelineConfig.n_rounds;
        training_config.max_depth = config.pipelineConfig.max_depth;
        training_config.nthread = config.pipelineConfig.nthread;
        training_config.learning_rate = config.pipelineConfig.learning_rate;
        training_config.subsample = config.pipelineConfig.subsample;
        training_config.colsample_bytree = config.pipelineConfig.colsample_bytree;
        training_config.random_seed = config.randomSeed;
        
        std::unique_ptr<MLPipeline::BarrierMLStrategy> strategy;
        try {
            if (config.useTTBM) {
                strategy = std::make_unique<MLPipeline::TTBMStrategy>();
            } else {
                strategy = std::make_unique<MLPipeline::HardBarrierStrategy>();
            }
        } catch (const std::bad_alloc& e) {
            throw ResourceAllocationException("ML Strategy", "Failed to allocate memory for strategy");
        }
        Validation::validateNotNull(strategy.get(), "strategy");

        FeatureExtractor::FeatureExtractionResult mappedFeatures = features;
        if (!config.useTTBM && mappedFeatures.labels.size() > 0) {
            for (auto& label : mappedFeatures.labels) {
                if (label == -1) label = 0;
                else if (label == 0) label = 1;
                else if (label == 1) label = 2;
            }
        }
        
        auto prediction_result = strategy->trainAndPredict(mappedFeatures, mappedFeatures.returns, training_config);

        std::vector<double> mapped_predictions;
        if (!prediction_result.predictions.empty()) {
            mapped_predictions.reserve(prediction_result.predictions.size());
            for (const auto& pred : prediction_result.predictions) {
                mapped_predictions.push_back(static_cast<double>(pred));
            }
        }
        if (!config.useTTBM && !mapped_predictions.empty()) {
            for (auto& pred : mapped_predictions) {
                if (pred == 0) pred = -1;
                else if (pred == 1) pred = 0;
                else if (pred == 2) pred = 1;
            }
        }
        results.predictions = mapped_predictions;
        results.prediction_probabilities = prediction_result.confidence_scores;
        
        const auto& portfolio = prediction_result.portfolio_result;
        results.portfolioResult.starting_capital = portfolio.starting_capital;
        results.portfolioResult.final_value = portfolio.final_capital;
        results.portfolioResult.total_return = portfolio.total_return;
        results.portfolioResult.max_drawdown = portfolio.max_drawdown;
        results.portfolioResult.total_trades = portfolio.total_trades;
        results.portfolioResult.win_rate = portfolio.win_rate;
        results.trade_log = portfolio.trade_log;

        Validation::validateFinite(results.portfolioResult.total_return, "total_return");
        Validation::validateNonNegative(results.portfolioResult.total_trades, "total_trades");
        
        results.modelInfo = QString("Strategy: %1").arg(QString::fromStdString(strategy->getStrategyName()));
        results.success = true;
        
        return results;
    } catch (const BaseException& e) {
        results.success = false;
        results.errorMessage = QString::fromStdString(e.full_message());
        return results;
    } catch (const std::exception& e) {
        auto converted = ExceptionUtils::convertException(e, "Model Training");
        results.success = false;
        results.errorMessage = QString::fromStdString(converted->full_message());
        return results;
    }
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
        config.starting_capital = 1000.0;
        config.position_threshold = 0.25;
        config.hard_barrier_position_pct = 0.25;
        config.max_trade_decisions_logged = 100;
        
        auto simulation = MLPipeline::simulate_portfolio(predictions, returns, config);
        
        MLPipeline::PortfolioResults results;
        results.starting_capital = simulation.starting_capital;
        results.final_value = simulation.final_capital; 
        results.total_return = simulation.total_return;
        results.max_drawdown = simulation.max_drawdown;
        results.total_trades = simulation.total_trades;
        results.win_rate = simulation.win_rate;
        
        return results;
        
    } catch (const std::exception& e) {
        MLPipeline::PortfolioResults results;
        results.starting_capital = 0.0;
        results.final_value = 0.0;
        results.total_return = 0.0;
        results.max_drawdown = 0.0;
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
