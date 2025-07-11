#include "ValidationFramework.h"
#include "TypeConversionAdapter.h"
#include "UnifiedErrorHandling.h"
#include "../../backend/data/PreprocessedRow.h"
#include "../../backend/data/LabeledEvent.h"
#include "../../backend/data/BarrierConfig.h"
#include "../../backend/ml/MLPipeline.h"
#include "../services/MLService.h"
#include <QFile>
#include <QFileInfo>
#include <cmath>
#include <algorithm>

namespace ValidationFramework {

ValidationConfig Validator::config_;

ValidationResult DataValidator::validateDataRows(const std::vector<PreprocessedRow>& rows) {
    using namespace UnifiedErrorHandling;
    
    ErrorHandler::ErrorContext context(
        ErrorHandler::ErrorCategory::Validation,
        ErrorHandler::ErrorSeverity::Error,
        "DataValidator",
        "validateDataRows"
    );
    
    return ErrorHandler::safeExecute([&]() -> ValidationResult {
        ValidationAccumulator accumulator;
        
        if (rows.empty()) {
            return ValidationResult::error(
                "Data rows cannot be empty",
                {"Load valid data", "Check data source"},
                "Data rows",
                "Empty Check"
            );
        }
        
        if (rows.size() < 10) {
            accumulator.addResult(ValidationResult::warning(
                QString("Very few data rows: %1 (minimum recommended: 10)").arg(rows.size()),
                {"Load more data", "Check data quality"},
                "Data rows",
                "Size Check"
            ));
        }
        
        size_t invalidRows = 0;
        size_t nanCount = 0;
        size_t infCount = 0;
        
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            
            if (std::isnan(row.log_return)) {
                nanCount++;
            }
            if (std::isinf(row.log_return)) {
                infCount++;
            }
            
            if (row.volatility < 0.0 || !std::isfinite(row.volatility)) {
                invalidRows++;
            }
            
            if (row.timestamp.empty()) {
                invalidRows++;
            }
        }
        
        if (nanCount > 0) {
            double nanPercent = (nanCount * 100.0) / rows.size();
            if (nanPercent > 5.0) {
                accumulator.addResult(ValidationResult::error(
                    QString("Too many NaN values: %1 out of %2 rows (%3%)")
                        .arg(nanCount).arg(rows.size()).arg(nanPercent, 0, 'f', 1),
                    {"Clean the data before processing", "Check data source quality"},
                    "Data Quality",
                    "NaN Check"
                ));
            } else if (nanPercent > 1.0) {
                accumulator.addResult(ValidationResult::warning(
                    QString("Some NaN values found: %1 out of %2 rows (%3%)")
                        .arg(nanCount).arg(rows.size()).arg(nanPercent, 0, 'f', 1),
                    {"Consider cleaning the data", "Monitor data quality"},
                    "Data Quality",
                    "NaN Check"
                ));
            }
        }
        
        if (infCount > 0) {
            accumulator.addResult(ValidationResult::error(
                QString("Infinite values found: %1 out of %2 rows").arg(infCount).arg(rows.size()),
                {"Remove infinite values", "Check data calculation logic"},
                "Data Quality",
                "Infinity Check"
            ));
        }
        
        if (invalidRows > 0) {
            double invalidPercent = (invalidRows * 100.0) / rows.size();
            if (invalidPercent > 10.0) {
                accumulator.addResult(ValidationResult::error(
                    QString("Too many invalid rows: %1 out of %2 (%3%)")
                        .arg(invalidRows).arg(rows.size()).arg(invalidPercent, 0, 'f', 1),
                    {"Fix data quality issues", "Check data preprocessing"},
                    "Data Quality",
                    "Invalid Rows Check"
                ));
            }
        }
        
        return accumulator.getSummary();
    }, context);
}

ValidationResult DataValidator::validateLabeledEvents(const std::vector<LabeledEvent>& events) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(CoreValidator::validateNotEmpty(events, "Labeled events"));
    if (!accumulator.isValid()) {
        return accumulator.getSummary();
    }
    
    accumulator.addResult(CoreValidator::validateMinSize(events.size(), 5, "Labeled events"));
    
    std::map<int, int> labelCounts;
    size_t invalidLabels = 0;
    
    for (const auto& event : events) {
        if (event.label < -1 || event.label > 1) {
            invalidLabels++;
        } else {
            labelCounts[event.label]++;
        }
        
        if (!std::isfinite(event.ttbm_label)) {
            invalidLabels++;
        }
    }
    
    if (invalidLabels > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("Invalid labels found: %1 out of %2 events").arg(invalidLabels).arg(events.size()),
            {"Check labeling logic", "Ensure labels are in range [-1, 1]"},
            "Label Quality",
            "Invalid Labels Check"
        ));
    }
    
    if (labelCounts.size() > 1) {
        int maxCount = 0;
        int minCount = INT_MAX;
        for (const auto& pair : labelCounts) {
            maxCount = std::max(maxCount, pair.second);
            minCount = std::min(minCount, pair.second);
        }
        
        if (maxCount > 0 && minCount > 0) {
            double imbalanceRatio = static_cast<double>(maxCount) / minCount;
            if (imbalanceRatio > 10.0) {
                accumulator.addResult(ValidationResult::warning(
                    QString("Severe label imbalance: ratio %1:1").arg(imbalanceRatio, 0, 'f', 1),
                    {"Consider balancing techniques", "Check barrier configuration"},
                    "Label Balance",
                    "Imbalance Check"
                ));
            } else if (imbalanceRatio > 3.0) {
                accumulator.addResult(ValidationResult::warning(
                    QString("Label imbalance detected: ratio %1:1").arg(imbalanceRatio, 0, 'f', 1),
                    {"Monitor model performance", "Consider rebalancing"},
                    "Label Balance",
                    "Imbalance Check"
                ));
            }
        }
    }
    
    return accumulator.getSummary();
}

ValidationResult DataValidator::validateDataConsistency(const std::vector<PreprocessedRow>& rows,
                                                       const std::vector<LabeledEvent>& events) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(validateDataRows(rows));
    accumulator.addResult(validateLabeledEvents(events));
    
    if (!accumulator.isValid()) {
        return accumulator.getSummary();
    }
    
    if (!rows.empty() && !events.empty()) {
        bool hasValidTimestamps = true;
        for (const auto& row : rows) {
            if (row.timestamp.empty()) {
                hasValidTimestamps = false;
                break;
            }
        }
        
        for (const auto& event : events) {
            if (event.entry_time.empty()) {
                hasValidTimestamps = false;
                break;
            }
        }
        
        if (!hasValidTimestamps) {
            accumulator.addResult(ValidationResult::warning(
                "Some entries have missing timestamps",
                {"Check data alignment", "Verify timestamp format"},
                "Time Consistency",
                "Timestamp Validation Check"
            ));
        }
    }
    
    return accumulator.getSummary();
}

ValidationResult MLValidator::validateMLConfig(const MLConfig& config) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(validateFeatureSelection(config.selectedFeatures));
    accumulator.addResult(validatePipelineConfig(config.pipelineConfig));
    
    accumulator.addResult(CoreValidator::validateRange(
        config.crossValidationRatio, 0.0, 0.5, "Cross-validation ratio"));
    
    accumulator.addResult(CoreValidator::validateRange(
        config.outlierThreshold, 1.0, 10.0, "Outlier threshold"));
    
    if (config.randomSeed < 0) {
        accumulator.addResult(ValidationResult::warning(
            "Negative random seed may cause issues",
            {"Use positive values for reproducibility"},
            "Random Seed",
            "Seed Check"
        ));
    }
    
    return accumulator.getSummary();
}

ValidationResult MLValidator::validatePipelineConfig(const MLPipeline::UnifiedPipelineConfig& config) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(CoreValidator::validateRange(
        config.test_size, 0.05, 0.5, "Test size"));
    
    accumulator.addResult(CoreValidator::validateRange(
        config.val_size, 0.0, 0.5, "Validation size"));
    
    if (config.test_size + config.val_size >= 1.0) {
        accumulator.addResult(ValidationResult::error(
            QString("Test size (%1) + Validation size (%2) must be less than 1.0")
                .arg(config.test_size).arg(config.val_size),
            {"Reduce test size or validation size", "Common split: 0.2 test, 0.2 validation"},
            "Data Split",
            "Split Sum Check"
        ));
    }
    
    accumulator.addResult(CoreValidator::validateRange(
        config.n_rounds, 1, 10000, "Number of rounds"));
    
    accumulator.addResult(CoreValidator::validateRange(
        config.max_depth, 1, 20, "Max depth"));
    
    accumulator.addResult(CoreValidator::validateRange(
        config.learning_rate, 0.001, 1.0, "Learning rate"));
    
    accumulator.addResult(CoreValidator::validateRange(
        config.subsample, 0.1, 1.0, "Subsample"));
    
    accumulator.addResult(CoreValidator::validateRange(
        config.colsample_bytree, 0.1, 1.0, "Column sample by tree"));
    
    accumulator.addResult(CoreValidator::validatePositive(
        config.nthread, "Number of threads"));
    
    if (config.n_rounds > 1000) {
        accumulator.addResult(ValidationResult::warning(
            "Large number of rounds may cause overfitting",
            {"Consider early stopping", "Monitor validation performance"},
            "Training Parameters",
            "Rounds Check"
        ));
    }
    
    if (config.learning_rate > 0.3) {
        accumulator.addResult(ValidationResult::warning(
            "High learning rate may cause instability",
            {"Try values between 0.01 and 0.3", "Monitor training loss"},
            "Learning Rate",
            "Rate Check"
        ));
    }
    
    return accumulator.getSummary();
}

ValidationResult MLValidator::validateFeatureSelection(const QSet<QString>& features) {
    ValidationAccumulator accumulator;
    
    if (TypeConversion::TypeAdapter::isEmpty(features)) {
        accumulator.addResult(ValidationResult::error(
            "Feature selection cannot be empty",
            {"Add at least one feature"},
            "Feature selection",
            "Empty Container Check"
        ));
        return accumulator.getSummary();
    }
    
    if (features.size() > 50) {
        accumulator.addResult(ValidationResult::warning(
            QString("Large number of features selected: %1").arg(features.size()),
            {"Consider feature selection techniques", "Start with 5-15 most relevant features"},
            "Feature Count",
            "Feature Selection Check"
        ));
    }
    
    QStringList invalidFeatures;
    try {
        auto stdFeatures = TypeConversion::TypeAdapter::toStdSet(features);
        
        for (const auto& feature : stdFeatures) {
            if (feature.empty()) {
                invalidFeatures.append("(empty)");
            }
        }
    } catch (const std::exception& e) {
        accumulator.addResult(ValidationResult::error(
            QString("Failed to validate feature selection: %1").arg(e.what()),
            {"Check feature format", "Ensure valid feature names"},
            "Feature Validation",
            "Type Conversion Error"
        ));
        return accumulator.getSummary();
    }
    
    if (!invalidFeatures.isEmpty()) {
        accumulator.addResult(ValidationResult::error(
            QString("Invalid features found: %1").arg(invalidFeatures.join(", ")),
            {"Remove invalid features", "Check feature names"},
            "Feature Validation",
            "Invalid Features Check"
        ));
    }
    
    return accumulator.getSummary();
}

ValidationResult MLValidator::validateModelInputs(const std::vector<std::vector<float>>& X,
                                                 const std::vector<float>& y) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(CoreValidator::validateNotEmpty(X, "Feature matrix"));
    accumulator.addResult(CoreValidator::validateNotEmpty(y, "Target vector"));
    
    if (!accumulator.isValid()) {
        return accumulator.getSummary();
    }
    
    accumulator.addResult(CoreValidator::validateSizeMatch(X, y, "Features", "Targets"));
    
    if (!X.empty()) {
        size_t expectedFeatures = X[0].size();
        for (size_t i = 1; i < X.size(); ++i) {
            if (X[i].size() != expectedFeatures) {
                accumulator.addResult(ValidationResult::error(
                    QString("Inconsistent feature count: row %1 has %2 features, expected %3")
                        .arg(i).arg(X[i].size()).arg(expectedFeatures),
                    {"Ensure all samples have the same number of features"},
                    "Feature Matrix",
                    "Feature Count Check"
                ));
                break;
            }
        }
    }
    
    size_t nanCount = 0;
    size_t infCount = 0;
    for (const auto& row : X) {
        for (float value : row) {
            if (std::isnan(value)) nanCount++;
            if (std::isinf(value)) infCount++;
        }
    }
    
    for (float value : y) {
        if (std::isnan(value)) nanCount++;
        if (std::isinf(value)) infCount++;
    }
    
    if (nanCount > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("NaN values found: %1 total").arg(nanCount),
            {"Remove or impute NaN values", "Check data preprocessing"},
            "Data Quality",
            "NaN Check"
        ));
    }
    
    if (infCount > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("Infinite values found: %1 total").arg(infCount),
            {"Remove or cap infinite values", "Check feature scaling"},
            "Data Quality",
            "Infinity Check"
        ));
    }
    
    return accumulator.getSummary();
}

ValidationResult MLValidator::validateModelInputs(const std::vector<std::vector<float>>& X,
                                                 const std::vector<double>& y) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(CoreValidator::validateNotEmpty(X, "Feature matrix"));
    accumulator.addResult(CoreValidator::validateNotEmpty(y, "Target vector"));
    
    if (!accumulator.isValid()) {
        return accumulator.getSummary();
    }
    
    accumulator.addResult(CoreValidator::validateSizeMatch(X, y, "Features", "Targets"));
    
    if (!X.empty()) {
        size_t expectedFeatures = X[0].size();
        for (size_t i = 1; i < X.size(); ++i) {
            if (X[i].size() != expectedFeatures) {
                accumulator.addResult(ValidationResult::error(
                    QString("Inconsistent feature count: row %1 has %2 features, expected %3")
                        .arg(i).arg(X[i].size()).arg(expectedFeatures),
                    {"Ensure all samples have the same number of features"},
                    "Feature Matrix",
                    "Feature Count Check"
                ));
                break;
            }
        }
    }
    
    size_t nanCount = 0;
    size_t infCount = 0;
    for (const auto& row : X) {
        for (float value : row) {
            if (std::isnan(value)) nanCount++;
            if (std::isinf(value)) infCount++;
        }
    }
    
    for (double value : y) {
        if (std::isnan(value)) nanCount++;
        if (std::isinf(value)) infCount++;
    }
    
    if (nanCount > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("NaN values found: %1 total").arg(nanCount),
            {"Remove or impute NaN values", "Check data preprocessing"},
            "Data Quality",
            "NaN Check"
        ));
    }
    
    if (infCount > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("Infinite values found: %1 total").arg(infCount),
            {"Remove or cap infinite values", "Check feature scaling"},
            "Data Quality",
            "Infinity Check"
        ));
    }

    return accumulator.getSummary();
}

ValidationResult MLValidator::validateModelInputs(const std::vector<std::map<std::string, double>>& X,
                                                 const std::vector<int>& y) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(CoreValidator::validateNotEmpty(X, "Feature matrix"));
    accumulator.addResult(CoreValidator::validateNotEmpty(y, "Target vector"));
    
    if (!accumulator.isValid()) {
        return accumulator.getSummary();
    }
    
    if (X.size() != y.size()) {
        accumulator.addResult(ValidationResult::error(
            QString("Feature matrix size (%1) doesn't match target vector size (%2)")
                .arg(X.size()).arg(y.size()),
            {"Ensure feature matrix and target vector have the same number of samples"},
            "Size Mismatch",
            "Sample Count Check"
        ));
        return accumulator.getSummary();
    }
    
    if (!X.empty()) {
        std::set<std::string> expectedFeatures;
        for (const auto& pair : X[0]) {
            expectedFeatures.insert(pair.first);
        }
        
        for (size_t i = 1; i < X.size(); ++i) {
            std::set<std::string> currentFeatures;
            for (const auto& pair : X[i]) {
                currentFeatures.insert(pair.first);
            }
            
            if (currentFeatures != expectedFeatures) {
                accumulator.addResult(ValidationResult::error(
                    QString("Inconsistent feature set at sample %1").arg(i),
                    {"Ensure all samples have the same feature keys"},
                    "Feature Matrix",
                    "Feature Set Check"
                ));
                break;
            }
        }
    }
    
    size_t nanCount = 0;
    size_t infCount = 0;
    for (const auto& sample : X) {
        for (const auto& pair : sample) {
            if (std::isnan(pair.second)) nanCount++;
            if (std::isinf(pair.second)) infCount++;
        }
    }
    
    if (nanCount > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("NaN values found: %1 total").arg(nanCount),
            {"Remove or impute NaN values", "Check data preprocessing"},
            "Data Quality",
            "NaN Check"
        ));
    }
    
    if (infCount > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("Infinite values found: %1 total").arg(infCount),
            {"Remove or cap infinite values", "Check feature scaling"},
            "Data Quality",
            "Infinity Check"
        ));
    }
    
    return accumulator.getSummary();
}

ValidationResult MLValidator::validateModelInputs(const std::vector<std::map<std::string, double>>& X,
                                                 const std::vector<double>& y) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(CoreValidator::validateNotEmpty(X, "Feature matrix"));
    accumulator.addResult(CoreValidator::validateNotEmpty(y, "Target vector"));
    
    if (!accumulator.isValid()) {
        return accumulator.getSummary();
    }
    
    if (X.size() != y.size()) {
        accumulator.addResult(ValidationResult::error(
            QString("Feature matrix size (%1) doesn't match target vector size (%2)")
                .arg(X.size()).arg(y.size()),
            {"Ensure feature matrix and target vector have the same number of samples"},
            "Size Mismatch",
            "Sample Count Check"
        ));
        return accumulator.getSummary();
    }
    
    if (!X.empty()) {
        std::set<std::string> expectedFeatures;
        for (const auto& pair : X[0]) {
            expectedFeatures.insert(pair.first);
        }
        
        for (size_t i = 1; i < X.size(); ++i) {
            std::set<std::string> currentFeatures;
            for (const auto& pair : X[i]) {
                currentFeatures.insert(pair.first);
            }
            
            if (currentFeatures != expectedFeatures) {
                accumulator.addResult(ValidationResult::error(
                    QString("Inconsistent feature set at sample %1").arg(i),
                    {"Ensure all samples have the same feature keys"},
                    "Feature Matrix",
                    "Feature Set Check"
                ));
                break;
            }
        }
    }
    
    size_t nanCount = 0;
    size_t infCount = 0;
    for (const auto& sample : X) {
        for (const auto& pair : sample) {
            if (std::isnan(pair.second)) nanCount++;
            if (std::isinf(pair.second)) infCount++;
        }
    }
    
    for (double value : y) {
        if (std::isnan(value)) nanCount++;
        if (std::isinf(value)) infCount++;
    }
    
    if (nanCount > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("NaN values found: %1 total").arg(nanCount),
            {"Remove or impute NaN values", "Check data preprocessing"},
            "Data Quality",
            "NaN Check"
        ));
    }
    
    if (infCount > 0) {
        accumulator.addResult(ValidationResult::error(
            QString("Infinite values found: %1 total").arg(infCount),
            {"Remove or cap infinite values", "Check feature scaling"},
            "Data Quality",
            "Infinity Check"
        ));
    }

    return accumulator.getSummary();
}

ValidationResult BarrierValidator::validateBarrierConfig(const BarrierConfig& config) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(validateBarrierParameters(
        config.profit_multiple, config.stop_multiple, config.vertical_window));
    
    if (config.use_cusum) {
        accumulator.addResult(CoreValidator::validatePositive(
            config.cusum_threshold, "CUSUM threshold"));
    }
    
    if (config.profit_multiple < config.stop_multiple) {
        accumulator.addResult(ValidationResult::warning(
            "Profit multiple is smaller than stop multiple",
            {"Consider making profit multiple at least equal to stop multiple"},
            "Risk-Reward Ratio",
            "Ratio Check"
        ));
    }
    
    return accumulator.getSummary();
}

ValidationResult BarrierValidator::validateBarrierParameters(double profitMultiple, 
                                                           double stopMultiple, 
                                                           int verticalWindow) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(CoreValidator::validatePositive(profitMultiple, "Profit multiple"));
    accumulator.addResult(CoreValidator::validatePositive(stopMultiple, "Stop multiple"));
    accumulator.addResult(CoreValidator::validatePositive(verticalWindow, "Vertical window"));
    
    if (profitMultiple > 10.0) {
        accumulator.addResult(ValidationResult::warning(
            "Very high profit multiple may result in few profitable trades",
            {"Consider values between 1.5 and 3.0"},
            "Profit Multiple",
            "Range Check"
        ));
    }
    
    if (verticalWindow > 1000) {
        accumulator.addResult(ValidationResult::warning(
            "Very large vertical window may affect performance",
            {"Consider values between 20 and 200"},
            "Vertical Window",
            "Range Check"
        ));
    }
    
    return accumulator.getSummary();
}

ValidationResult Validator::validateData(const std::vector<PreprocessedRow>& rows,
                                        const std::vector<LabeledEvent>& events) {
    return DataValidator::validateDataConsistency(rows, events);
}

ValidationResult Validator::validateML(const MLConfig& config) {
    return MLValidator::validateMLConfig(config);
}

ValidationResult Validator::validateBarrier(const BarrierConfig& config) {
    return BarrierValidator::validateBarrierConfig(config);
}

ValidationResult Validator::validateAll(const std::vector<PreprocessedRow>& rows,
                                       const std::vector<LabeledEvent>& events,
                                       const MLConfig& mlConfig,
                                       const BarrierConfig& barrierConfig) {
    ValidationAccumulator accumulator;
    
    accumulator.addResult(validateData(rows, events));
    accumulator.addResult(validateML(mlConfig));
    accumulator.addResult(validateBarrier(barrierConfig));
    
    return accumulator.getSummary();
}

} // namespace ValidationFramework
