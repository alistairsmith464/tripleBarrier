#pragma once

#include <vector>
#include <QString>
#include "../../backend/data/DataRow.h"
#include "../../backend/data/PreprocessedRow.h"
#include "../../backend/data/LabeledEvent.h"
#include "../../backend/data/BarrierConfig.h"
#include "../../backend/data/DataPreprocessor.h"

// Service interface for data operations
class DataService {
public:
    virtual ~DataService() = default;
    
    // Data loading
    virtual std::vector<DataRow> loadCSVData(const QString& filePath) = 0;
    
    // Data processing
    virtual std::vector<PreprocessedRow> preprocessData(
        const std::vector<DataRow>& rawData,
        const DataPreprocessor::Params& params) = 0;
    
    // Event labeling
    virtual std::vector<LabeledEvent> labelEvents(
        const std::vector<PreprocessedRow>& processedData,
        const std::vector<size_t>& eventIndices,
        const BarrierConfig& config) = 0;
    
    // Combined labeling (convenience method)
    virtual std::vector<LabeledEvent> generateLabeledEvents(
        const std::vector<PreprocessedRow>& processedData,
        const BarrierConfig& config) = 0;
    
    // Event selection
    virtual std::vector<size_t> selectEventIndices(
        const std::vector<PreprocessedRow>& processedData) = 0;
};

// Concrete implementation
class DataServiceImpl : public DataService {
public:
    std::vector<DataRow> loadCSVData(const QString& filePath) override;
    
    std::vector<PreprocessedRow> preprocessData(
        const std::vector<DataRow>& rawData,
        const DataPreprocessor::Params& params) override;
    
    std::vector<LabeledEvent> labelEvents(
        const std::vector<PreprocessedRow>& processedData,
        const std::vector<size_t>& eventIndices,
        const BarrierConfig& config) override;
    
    std::vector<LabeledEvent> generateLabeledEvents(
        const std::vector<PreprocessedRow>& processedData,
        const BarrierConfig& config) override;
    
    std::vector<size_t> selectEventIndices(
        const std::vector<PreprocessedRow>& processedData) override;
};
