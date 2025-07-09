#include "MLService.h"
#include "../../backend/data/PreprocessedRow.h"
#include "../../backend/data/LabeledEvent.h"
#include "../../backend/data/FeatureExtractor.h"
#include "../../backend/data/PortfolioSimulator.h"
#include "../../backend/ml/MLPipeline.h"
#include "../../backend/ml/MLSplits.h"
#include "../config/VisualizationConfig.h"
#include <algorithm>

FeatureExtractor::FeatureExtractionResult MLServiceImpl::extractFeatures(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const QSet<QString>& selectedFeatures) {
    
    // Convert QSet to std::set
    std::set<std::string> features;
    for (const QString& feature : selectedFeatures) {
        features.insert(feature.toStdString());
    }
    
    return FeatureExtractor::extractFeaturesForClassification(features, rows, labeledEvents);
}

MLResults MLServiceImpl::runMLPipeline(
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    const MLConfig& config) {
    
    MLResults results;
    
    try {
        // Extract features
        results.features = extractFeatures(rows, labeledEvents, config.selectedFeatures);
        
        if (results.features.features.empty()) {
            results.errorMessage = "No features extracted for ML training";
            return results;
        }
        
        // Configure splits
        std::vector<int> labels = config.useTTBM ? 
            std::vector<int>(results.features.labels_double.begin(), results.features.labels_double.end()) :
            results.features.labels;
            
        auto splitResult = MLSplitUtils::chronologicalSplit(
            results.features.features, 
            labels, 
            1.0 - config.crossValidationRatio, 
            config.crossValidationRatio, 
            0.0);
        
        // Configure ML pipeline
        MLPipeline::PipelineConfig pipelineConfig;
        pipelineConfig.split_type = MLPipeline::Chronological;
        pipelineConfig.train_ratio = 1.0 - config.crossValidationRatio;
        pipelineConfig.val_ratio = config.crossValidationRatio;
        pipelineConfig.test_ratio = 0.0;
        pipelineConfig.n_rounds = 20;
        pipelineConfig.max_depth = 3;
        pipelineConfig.nthread = 4;
        
        // Run ML pipeline
        if (config.useTTBM) {
            std::vector<double> ttbmLabels(results.features.labels_double);
            std::vector<double> returns; // Empty returns for now
            auto pipelineResult = MLPipeline::runPipelineRegression(
                results.features.features, ttbmLabels, returns, pipelineConfig);
            results.predictions = pipelineResult.predictions;
            results.accuracy = 0.0; // No accuracy for regression
            results.modelInfo = "TTBM Regression Model";
        } else {
            std::vector<double> returns; // Empty returns for now
            auto pipelineResult = MLPipeline::runPipeline(
                results.features.features, results.features.labels, returns, pipelineConfig);
            results.predictions.assign(pipelineResult.predictions.begin(), pipelineResult.predictions.end());
            results.accuracy = 0.0; // Calculate from predictions if needed
            results.modelInfo = "Classification Model";
        }
        
        // Run portfolio simulation
        results.portfolioResult = runPortfolioSimulation(rows, labeledEvents, results.predictions, config.useTTBM);
        
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
    
    // Use configuration instead of hardcoded values
    double positionMultiplier = useTTBM ? 
        VisualizationConfig::getTTBMPositionMultiplier() :
        VisualizationConfig::getHardBarrierPositionSize();
    
    double threshold = VisualizationConfig::getTradingThreshold();
    
    return PortfolioSimulator::runSimulation(predictions, labeledEvents, useTTBM);
}
