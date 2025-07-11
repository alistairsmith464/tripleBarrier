#pragma once

#include <QSet>
#include <QString>
#include <QObject>
#include <vector>
#include <memory>
#include <functional>
#include <future>

struct PreprocessedRow;
struct LabeledEvent;

#include "../backend/data/FeatureExtractor.h"
#include "../backend/ml/PortfolioSimulator.h"
#include "../backend/ml/MLPipeline.h"

struct MLConfig {
    QSet<QString> selectedFeatures;
    bool useTTBM = false;
    double crossValidationRatio = 0.2;
    int randomSeed = 42;
    bool tuneHyperparameters = false;
    
    MLPipeline::UnifiedPipelineConfig pipelineConfig;
    
    bool preprocessFeatures = true;
    bool normalizeFeatures = false;
    bool removeOutliers = false;
    double outlierThreshold = 3.0;
    
    bool enableProgressCallbacks = false;
};

struct MLResults {
    FeatureExtractor::FeatureExtractionResult features;
    
    std::vector<double> predictions;
    std::vector<double> prediction_probabilities;
    
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
    double auc_roc = 0.0;
    std::vector<std::vector<int>> confusion_matrix;
    
    double r2_score = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    double mape = 0.0;
    
    QString modelInfo;
    QString modelVersion;
    QString trainingTime;
    std::vector<QString> featureImportance;
    
    MLPipeline::PortfolioResults portfolioResult;
    
    bool success = false;
    QString errorMessage;
    QString warningMessage;
    
    struct DataQuality {
        size_t total_samples;
        size_t valid_samples;
        size_t removed_outliers;
        double feature_completeness;
    } dataQuality;
};

struct MLProgress {
    enum Stage {
        FEATURE_EXTRACTION,
        DATA_PREPROCESSING,
        MODEL_TRAINING,
        HYPERPARAMETER_TUNING,
        MODEL_EVALUATION,
        PORTFOLIO_SIMULATION,
        COMPLETE
    };
    
    Stage current_stage;
    double progress_percentage;
    QString status_message;
    QString estimated_time_remaining;
};

using MLProgressCallback = std::function<void(const MLProgress&)>;

class FeatureService {
public:
    virtual ~FeatureService() = default;
    
    virtual FeatureExtractor::FeatureExtractionResult extractFeaturesForClassification(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const QSet<QString>& selectedFeatures) = 0;
    
    virtual FeatureExtractor::FeatureExtractionResult extractFeaturesForRegression(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const QSet<QString>& selectedFeatures) = 0;
        
    virtual QStringList getAvailableFeatures() = 0;
    virtual QString validateFeatureSelection(const QSet<QString>& features) = 0;
};

class ModelService {
public:
    virtual ~ModelService() = default;
    
    virtual MLResults trainModel(
        const FeatureExtractor::FeatureExtractionResult& features,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config) = 0;
    
    virtual std::future<MLResults> trainModelAsync(
        const FeatureExtractor::FeatureExtractionResult& features,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config,
        MLProgressCallback callback = nullptr) = 0;
};

class PortfolioService {
public:
    virtual ~PortfolioService() = default;
    
    virtual MLPipeline::PortfolioResults runSimulation(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const std::vector<double>& predictions,
        bool useTTBM) = 0;
        
    virtual MLPipeline::PortfolioResults runBacktest(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const std::vector<double>& predictions,
        const QString& strategy) = 0;
};

class MLService : public QObject {
    Q_OBJECT
    
public:
    virtual ~MLService() = default;
    
    virtual MLResults runMLPipeline(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config) = 0;
    
    virtual std::future<MLResults> runMLPipelineAsync(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config,
        MLProgressCallback callback = nullptr) = 0;
    
    virtual FeatureService* getFeatureService() = 0;
    virtual ModelService* getModelService() = 0;
    virtual PortfolioService* getPortfolioService() = 0;
    
    virtual QString validateConfiguration(const MLConfig& config) = 0;
    virtual MLConfig getDefaultConfiguration() = 0;

signals:
    void progressUpdate(const MLProgress& progress);
    void operationComplete(const MLResults& results);
    void errorOccurred(const QString& error);
};

class FeatureServiceImpl : public FeatureService {
public:
    FeatureExtractor::FeatureExtractionResult extractFeaturesForClassification(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const QSet<QString>& selectedFeatures) override;
    
    FeatureExtractor::FeatureExtractionResult extractFeaturesForRegression(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const QSet<QString>& selectedFeatures) override;
        
    QStringList getAvailableFeatures() override;
    QString validateFeatureSelection(const QSet<QString>& features) override;
};

class ModelServiceImpl : public ModelService {
public:
    MLResults trainModel(
        const FeatureExtractor::FeatureExtractionResult& features,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config) override;
    
    std::future<MLResults> trainModelAsync(
        const FeatureExtractor::FeatureExtractionResult& features,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config,
        MLProgressCallback callback = nullptr) override;
};

class PortfolioServiceImpl : public PortfolioService {
public:
    MLPipeline::PortfolioResults runSimulation(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const std::vector<double>& predictions,
        bool useTTBM) override;
        
    MLPipeline::PortfolioResults runBacktest(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const std::vector<double>& predictions,
        const QString& strategy) override;
};

class MLServiceImpl : public MLService {
public:
    MLServiceImpl();
    
    MLResults runMLPipeline(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config) override;
    
    std::future<MLResults> runMLPipelineAsync(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents,
        const MLConfig& config,
        MLProgressCallback callback = nullptr) override;
    
    FeatureService* getFeatureService() override { return feature_service_.get(); }
    ModelService* getModelService() override { return model_service_.get(); }
    PortfolioService* getPortfolioService() override { return portfolio_service_.get(); }
    
    QString validateConfiguration(const MLConfig& config) override;
    MLConfig getDefaultConfiguration() override;

private:
    std::unique_ptr<FeatureService> feature_service_;
    std::unique_ptr<ModelService> model_service_;
    std::unique_ptr<PortfolioService> portfolio_service_;
    
    MLResults calculateDetailedMetrics(
        const std::vector<int>& y_true,
        const std::vector<int>& y_pred,
        const std::vector<double>& y_prob = {});
        
    MLResults calculateRegressionMetrics(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred);
};
