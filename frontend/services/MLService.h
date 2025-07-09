#pragma once

#include <QSet>
#include <QString>
#include <vector>
#include <memory>

// Forward declarations
struct PreprocessedRow;
struct LabeledEvent;

// Re-use the existing FeatureExtractor result structure
#include "../backend/data/FeatureExtractor.h"
#include "../backend/data/PortfolioSimulator.h"

// ML configuration data transfer object
struct MLConfig {
    QSet<QString> selectedFeatures;
    bool useTTBM = false;
    double crossValidationRatio = 0.2;
    int randomSeed = 42;
};

// ML results data transfer object
struct MLResults {
    FeatureExtractor::FeatureExtractionResult features;
    std::vector<double> predictions;
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    QString modelInfo;
    PortfolioResults portfolioResult;
    bool success = false;
    QString errorMessage;
};

// Service interface for ML operations
class MLService {
public:
    virtual ~MLService() = default;
    
    // Extract features for ML
    virtual FeatureExtractor::FeatureExtractionResult extractFeatures(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const QSet<QString>& selectedFeatures) = 0;
    
    // Run ML pipeline
    virtual MLResults runMLPipeline(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config) = 0;
    
    // Run portfolio simulation
    virtual PortfolioResults runPortfolioSimulation(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const std::vector<double>& predictions,
        bool useTTBM) = 0;
};

// Concrete implementation
class MLServiceImpl : public MLService {
public:
    FeatureExtractor::FeatureExtractionResult extractFeatures(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const QSet<QString>& selectedFeatures) override;
    
    MLResults runMLPipeline(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config) override;
    
    PortfolioResults runPortfolioSimulation(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const std::vector<double>& predictions,
        bool useTTBM) override;
};
