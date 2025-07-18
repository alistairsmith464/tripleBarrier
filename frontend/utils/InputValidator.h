#pragma once
#include <QString>
#include <QSet>
#include <vector>
#include "../../backend/data/BarrierConfig.h"
#include "../../backend/ml/MLPipeline.h"

struct ValidationResult {
    bool isValid;
    QString errorMessage;
    QString warningMessage;
    QStringList suggestions;
    
    ValidationResult(bool valid = true) : isValid(valid) {}
    
    static ValidationResult success() {
        return ValidationResult(true);
    }
    
    static ValidationResult error(const QString& message, const QStringList& suggestions = {}) {
        ValidationResult result(false);
        result.errorMessage = message;
        result.suggestions = suggestions;
        return result;
    }
    
    static ValidationResult warning(const QString& message, const QStringList& suggestions = {}) {
        ValidationResult result(true);
        result.warningMessage = message;
        result.suggestions = suggestions;
        return result;
    }
};

class InputValidator {
public:
    static ValidationResult validateBarrierConfig(const BarrierConfig& config);
    
    static ValidationResult validateFeatureSelection(const QSet<QString>& features);
    
    static ValidationResult validateFilePath(const QString& filePath, bool mustExist = true);
    
    static ValidationResult validateRange(double value, double min, double max, 
                                        const QString& fieldName);
    
    static ValidationResult validateMLConfig(const MLPipeline::UnifiedPipelineConfig& config);
    
    static ValidationResult validateDataSize(size_t dataSize, size_t minRequired = 10);
    
    static ValidationResult validateWindowSize(int windowSize, int dataSize);
    
private:
    static QStringList getAvailableFeatures();
    static bool isValidFeature(const QString& feature);
};
