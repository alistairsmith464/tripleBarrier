#include "DataService.h"
#include "../../backend/data/CSVDataSource.h"
#include "../../backend/data/DataPreprocessor.h"
#include "../../backend/data/HardBarrierLabeler.h"
#include "../../backend/data/TTBMLabeler.h"
#include "../../backend/data/BarrierConfig.h"

std::vector<DataRow> DataServiceImpl::loadCSVData(const QString& filePath) {
    CSVDataSource source;
    return source.loadData(filePath.toStdString());
}

std::vector<PreprocessedRow> DataServiceImpl::preprocessData(
    const std::vector<DataRow>& rawData,
    const DataPreprocessor::Params& params) {
    return DataPreprocessor::preprocess(rawData, params);
}

std::vector<LabeledEvent> DataServiceImpl::labelEvents(
    const std::vector<PreprocessedRow>& processedData,
    const std::vector<size_t>& eventIndices,
    const BarrierConfig& config) {
    
    if (config.labeling_type == BarrierConfig::TTBM) {
        TTBMLabeler labeler(config.ttbm_decay_type, config.ttbm_lambda, config.ttbm_alpha, config.ttbm_beta);
        return labeler.label(
            processedData,
            eventIndices,
            config.profit_multiple,
            config.stop_multiple,
            config.vertical_window
        );
    } else {
        HardBarrierLabeler labeler;
        return labeler.label(
            processedData,
            eventIndices,
            config.profit_multiple,
            config.stop_multiple,
            config.vertical_window
        );
    }
}

std::vector<LabeledEvent> DataServiceImpl::generateLabeledEvents(
    const std::vector<PreprocessedRow>& processedData,
    const BarrierConfig& config) {
    
    // Generate event indices first
    std::vector<size_t> eventIndices = selectEventIndices(processedData);
    
    // Then label the events
    return labelEvents(processedData, eventIndices, config);
}

std::vector<size_t> DataServiceImpl::selectEventIndices(
        const std::vector<PreprocessedRow>& processedData) {
    std::vector<size_t> event_indices;
    for (size_t i = 0; i < processedData.size(); ++i) {
        if (processedData[i].is_event) event_indices.push_back(i);
    }
    return event_indices;
}
