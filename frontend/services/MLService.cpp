#include "MLService.h"
#include "../../backend/data/PreprocessedRow.h"
#include "../../backend/data/LabeledEvent.h"
#include "../../backend/data/FeatureExtractor.h"
#include "../../backend/data/PortfolioSimulator.h"
#include "../../backend/ml/MLPipeline.h"
#include "../../backend/ml/MLSplits.h"
#include "../config/VisualizationConfig.h"
#include <algorithm>
#include <cstdio>

FeatureExtractor::FeatureExtractionResult MLServiceImpl::extractFeatures(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const QSet<QString>& selectedFeatures,
    bool useTTBM) {
    
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
        if (useTTBM) {
            return FeatureExtractor::extractFeaturesForRegression(features, rows, labeledEvents);
        } else {
            return FeatureExtractor::extractFeaturesForClassification(features, rows, labeledEvents);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Feature extraction failed: ") + e.what());
    }
}

MLResults MLServiceImpl::runMLPipeline(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config) {
    
    MLResults results;
    
    try {
        // Validate inputs
        if (rows.empty()) {
            results.errorMessage = "No price data available for ML training";
            results.success = false;
            return results;
        }
        
        if (labeledEvents.empty()) {
            results.errorMessage = "No labeled events available for ML training";
            results.success = false;
            return results;
        }
        
        if (config.selectedFeatures.empty()) {
            results.errorMessage = "No features selected for ML training";
            results.success = false;
            return results;
        }
        
        // Extract features
        try {
            results.features = extractFeatures(rows, labeledEvents, config.selectedFeatures, config.useTTBM);
        } catch (const std::exception& e) {
            results.errorMessage = QString("Feature extraction failed: %1").arg(e.what());
            results.success = false;
            return results;
        }
        
        if (results.features.features.empty()) {
            results.errorMessage = "No features extracted for ML training";
            results.success = false;
            return results;
        }
        
        // Validate labeled events
        if (labeledEvents.empty()) {
            results.errorMessage = "No labeled events available for training";
            results.success = false;
            return results;
        }
        
        // Configure splits - validate data types for TTBM mode
        std::vector<int> labels;
        if (config.useTTBM) {
            if (results.features.labels_double.empty()) {
                results.errorMessage = "No TTBM labels available in extracted features";
                results.success = false;
                return results;
            }
            // Convert double labels to int for chronological split (which expects int labels)
            labels.reserve(results.features.labels_double.size());
            for (double label : results.features.labels_double) {
                labels.push_back(static_cast<int>(std::round(label)));
            }
        } else {
            if (results.features.labels.empty()) {
                results.errorMessage = "No classification labels available in extracted features";
                results.success = false;
                return results;
            }
            labels = results.features.labels;
        }
            
        auto splitResult = MLSplitUtils::chronologicalSplit(
            results.features.features, 
            labels, 
            1.0 - config.crossValidationRatio, 
            config.crossValidationRatio, 
            0.0);
            
        // Validate split results
        if (splitResult.X_train.empty() || splitResult.y_train.empty()) {
            results.errorMessage = "Data split produced empty training set";
            results.success = false;
            return results;
        }
        
        if (splitResult.X_val.empty() || splitResult.y_val.empty()) {
            results.errorMessage = "Data split produced empty validation set";
            results.success = false;
            return results;
        }
        
        // Configure ML pipeline
        MLPipeline::PipelineConfig pipelineConfig;
        pipelineConfig.split_type = MLPipeline::Chronological;
        // Configure ML pipeline - use larger validation set for portfolio simulation
        pipelineConfig.train_ratio = 0.6;  // Use 60% for training
        pipelineConfig.val_ratio = 0.4;    // Use 40% for validation/predictions (larger for portfolio sim)
        pipelineConfig.test_ratio = 0.0;   // No separate test set
        pipelineConfig.random_seed = config.randomSeed;
        
        // Configure hyperparameters
        if (config.tuneHyperparameters) {
            // Use broader search ranges for hyperparameter tuning
            pipelineConfig.n_rounds = 50;
            pipelineConfig.max_depth = 6;
            pipelineConfig.nthread = 4;
        } else {
            // Use default values
            pipelineConfig.n_rounds = config.nRounds;
            pipelineConfig.max_depth = config.maxDepth;
            pipelineConfig.nthread = config.nThreads;
        }
        
        // Validate input data before ML pipeline
        if (results.features.features.empty()) {
            results.errorMessage = "No feature data available for ML pipeline";
            results.success = false;
            return results;
        }
        
        // Create returns vector based on labeled events
        std::vector<double> returns;
        for (const auto& event : labeledEvents) {
            double return_value = (event.exit_price - event.entry_price) / event.entry_price;
            returns.push_back(return_value);
        }
        
        // Ensure returns vector has same size as features
        if (returns.size() != results.features.features.size()) {
            returns.resize(results.features.features.size(), 0.0);
        }
        
        // Validate data consistency before ML pipeline
        size_t expectedSize = results.features.features.size();
        size_t labelSize = config.useTTBM ? results.features.labels_double.size() : results.features.labels.size();
        
        if (expectedSize == 0) {
            results.errorMessage = "No feature data available - features vector is empty";
            results.success = false;
            return results;
        }
        
        if (labelSize == 0) {
            results.errorMessage = config.useTTBM ? "No TTBM labels available" : "No classification labels available";
            results.success = false;
            return results;
        }
        
        if (labelSize != expectedSize) {
            results.errorMessage = QString("Data size mismatch: Features=%1, Labels=%2, Returns=%3")
                                  .arg(expectedSize).arg(labelSize).arg(returns.size());
            results.success = false;
            return results;
        }
        
        // Run ML pipeline
        printf("DEBUG: MLService - About to call ML pipeline\n");
        printf("DEBUG: MLService - Features: %zu, Events: %zu, TTBM=%s\n", 
               results.features.features.size(), labeledEvents.size(), config.useTTBM ? "true" : "false");
        printf("DEBUG: MLService - Pipeline config: train=%.1f%%, val=%.1f%%, test=%.1f%%\n",
               pipelineConfig.train_ratio * 100, pipelineConfig.val_ratio * 100, pipelineConfig.test_ratio * 100);
        
        try {
            if (config.useTTBM) {
                // Validate TTBM labels
                if (results.features.labels_double.empty()) {
                    results.errorMessage = "No TTBM labels available for regression";
                    results.success = false;
                    return results;
                }
                
                std::vector<double> ttbmLabels(results.features.labels_double);
                
                // Use appropriate pipeline function based on hyperparameter tuning setting
                if (config.tuneHyperparameters) {
                    auto pipelineResult = MLPipeline::runPipelineRegressionWithTuning(
                        results.features.features, ttbmLabels, returns, pipelineConfig);
                    results.predictions = pipelineResult.predictions;
                    results.modelInfo = "TTBM Regression Model (with hyperparameter tuning)";
                } else {
                    auto pipelineResult = MLPipeline::runPipelineRegression(
                        results.features.features, ttbmLabels, returns, pipelineConfig);
                    results.predictions = pipelineResult.predictions;
                    results.modelInfo = "TTBM Regression Model";
                }
                
                // Debug output
                printf("DEBUG: ML Pipeline generated %zu predictions for %zu features\n", 
                       results.predictions.size(), results.features.features.size());
                results.accuracy = 0.0; // No accuracy for regression
            } else {
                // Validate classification labels
                if (results.features.labels.empty()) {
                    results.errorMessage = "No classification labels available";
                    results.success = false;
                    return results;
                }
                
                // Use appropriate pipeline function based on hyperparameter tuning setting
                if (config.tuneHyperparameters) {
                    auto pipelineResult = MLPipeline::runPipelineWithTuning(
                        results.features.features, results.features.labels, returns, pipelineConfig);
                    results.predictions.assign(pipelineResult.predictions.begin(), pipelineResult.predictions.end());
                    results.modelInfo = "Classification Model (with hyperparameter tuning)";
                } else {
                    auto pipelineResult = MLPipeline::runPipeline(
                        results.features.features, results.features.labels, returns, pipelineConfig);
                    results.predictions.assign(pipelineResult.predictions.begin(), pipelineResult.predictions.end());
                    results.modelInfo = "Classification Model";
                }
                
                // Debug output
                printf("DEBUG: ML Pipeline generated %zu predictions for %zu features\n", 
                       results.predictions.size(), results.features.features.size());
                
                results.accuracy = 0.0; // Calculate from predictions if needed
            }
        } catch (const std::exception& e) {
            results.errorMessage = QString("ML Pipeline execution failed: %1").arg(e.what());
            results.success = false;
            return results;
        }
        
        // Validate predictions before portfolio simulation
        if (results.predictions.empty()) {
            results.errorMessage = QString("ML pipeline produced no predictions. Features: %1, Labels: %2, Returns: %3")
                                  .arg(results.features.features.size())
                                  .arg(config.useTTBM ? results.features.labels_double.size() : results.features.labels.size())
                                  .arg(returns.size());
            results.success = false;
            return results;
        }
        
        // Run portfolio simulation
        try {
            results.portfolioResult = runPortfolioSimulation(rows, labeledEvents, results.predictions, config.useTTBM);
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

PortfolioResults MLServiceImpl::runPortfolioSimulation(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const std::vector<double>& predictions,
    bool useTTBM) {
    
    // Validate inputs
    if (predictions.empty()) {
        throw std::runtime_error("Empty predictions vector provided to portfolio simulation");
    }
    
    if (labeledEvents.empty()) {
        throw std::runtime_error("Empty labeled events provided to portfolio simulation");
    }
    
    // Ensure predictions and events are compatible
    if (predictions.size() > labeledEvents.size()) {
        throw std::runtime_error("More predictions than labeled events available");
    }
    
    try {
        return PortfolioSimulator::runSimulation(predictions, labeledEvents, useTTBM);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Portfolio simulation failed: ") + e.what());
    }
}
