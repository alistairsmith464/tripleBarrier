#pragma once

#include <QString>
#include <QStringList>
#include <QSet>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <map>
#include <set>
#include "../../backend/utils/Exceptions.h"

// Forward declarations
struct PreprocessedRow;
struct LabeledEvent;
struct MLConfig;
struct BarrierConfig;
namespace MLPipeline { struct UnifiedPipelineConfig; }

namespace ValidationFramework {

// Enhanced validation result with detailed error information
struct ValidationResult {
    bool isValid;
    QString errorMessage;
    QString warningMessage;
    QStringList suggestions;
    QString fieldName;
    QString context;
    int errorCode;
    
    ValidationResult(bool valid = true) 
        : isValid(valid), errorCode(0) {}
    
    static ValidationResult success() {
        return ValidationResult(true);
    }
    
    static ValidationResult error(const QString& message, 
                                 const QStringList& suggestions = {},
                                 const QString& fieldName = "",
                                 const QString& context = "",
                                 int errorCode = 0) {
        ValidationResult result(false);
        result.errorMessage = message;
        result.suggestions = suggestions;
        result.fieldName = fieldName;
        result.context = context;
        result.errorCode = errorCode;
        return result;
    }
    
    static ValidationResult warning(const QString& message, 
                                   const QStringList& suggestions = {},
                                   const QString& fieldName = "",
                                   const QString& context = "") {
        ValidationResult result(true);
        result.warningMessage = message;
        result.suggestions = suggestions;
        result.fieldName = fieldName;
        result.context = context;
        return result;
    }
    
    // Convert to backend exception
    TripleBarrier::BaseException toException() const {
        return TripleBarrier::DataValidationException(
            errorMessage.toStdString(),
            fieldName.toStdString()
        );
    }
};

// Validation result accumulator for batch operations
class ValidationAccumulator {
public:
    void addResult(const ValidationResult& result) {
        results_.push_back(result);
        if (!result.isValid) {
            hasErrors_ = true;
        }
        if (!result.warningMessage.isEmpty()) {
            hasWarnings_ = true;
        }
    }
    
    bool hasErrors() const { return hasErrors_; }
    bool hasWarnings() const { return hasWarnings_; }
    bool isValid() const { return !hasErrors_; }
    
    QStringList getErrors() const {
        QStringList errors;
        for (const auto& result : results_) {
            if (!result.isValid) {
                errors.append(result.errorMessage);
            }
        }
        return errors;
    }
    
    QStringList getWarnings() const {
        QStringList warnings;
        for (const auto& result : results_) {
            if (!result.warningMessage.isEmpty()) {
                warnings.append(result.warningMessage);
            }
        }
        return warnings;
    }
    
    ValidationResult getSummary() const {
        if (hasErrors_) {
            return ValidationResult::error(
                QString("Validation failed with %1 errors").arg(getErrors().size()),
                getErrors()
            );
        } else if (hasWarnings_) {
            return ValidationResult::warning(
                QString("Validation completed with %1 warnings").arg(getWarnings().size()),
                getWarnings()
            );
        } else {
            return ValidationResult::success();
        }
    }
    
private:
    std::vector<ValidationResult> results_;
    bool hasErrors_ = false;
    bool hasWarnings_ = false;
};

// Core validation functions
class CoreValidator {
public:
    // Basic type validations
    static ValidationResult validateNotEmpty(const QString& value, const QString& fieldName) {
        if (value.isEmpty()) {
            return ValidationResult::error(
                QString("%1 cannot be empty").arg(fieldName),
                {"Enter a valid value"},
                fieldName,
                "Empty Value Check"
            );
        }
        return ValidationResult::success();
    }
    
    template<typename Container>
    static ValidationResult validateNotEmpty(const Container& container, const QString& fieldName) {
        if (container.empty()) {
            return ValidationResult::error(
                QString("%1 cannot be empty").arg(fieldName),
                {"Add at least one item"},
                fieldName,
                "Empty Container Check"
            );
        }
        return ValidationResult::success();
    }
    
    static ValidationResult validateRange(double value, double min, double max, const QString& fieldName) {
        if (std::isnan(value)) {
            return ValidationResult::error(
                QString("%1 is not a valid number").arg(fieldName),
                {"Enter a numeric value"},
                fieldName,
                "NaN Check"
            );
        }
        
        if (value < min || value > max) {
            return ValidationResult::error(
                QString("%1 must be between %2 and %3 (got %4)").arg(fieldName).arg(min).arg(max).arg(value),
                {QString("Enter a value between %1 and %2").arg(min).arg(max)},
                fieldName,
                "Range Check"
            );
        }
        
        return ValidationResult::success();
    }
    
    static ValidationResult validatePositive(double value, const QString& fieldName) {
        if (value <= 0.0) {
            return ValidationResult::error(
                QString("%1 must be positive (got %2)").arg(fieldName).arg(value),
                {"Enter a positive value"},
                fieldName,
                "Positive Check"
            );
        }
        return ValidationResult::success();
    }
    
    static ValidationResult validateNonNegative(double value, const QString& fieldName) {
        if (value < 0.0) {
            return ValidationResult::error(
                QString("%1 must be non-negative (got %2)").arg(fieldName).arg(value),
                {"Enter a non-negative value"},
                fieldName,
                "Non-negative Check"
            );
        }
        return ValidationResult::success();
    }
    
    static ValidationResult validateFinite(double value, const QString& fieldName) {
        if (!std::isfinite(value)) {
            return ValidationResult::error(
                QString("%1 must be finite (got %2)").arg(fieldName).arg(value),
                {"Enter a finite value (not NaN or infinity)"},
                fieldName,
                "Finite Check"
            );
        }
        return ValidationResult::success();
    }
    
    template<typename Container1, typename Container2>
    static ValidationResult validateSizeMatch(const Container1& c1, const Container2& c2, 
                                             const QString& name1, const QString& name2) {
        if (c1.size() != c2.size()) {
            return ValidationResult::error(
                QString("Size mismatch: %1 has %2 elements, %3 has %4 elements")
                    .arg(name1).arg(c1.size()).arg(name2).arg(c2.size()),
                {"Ensure both containers have the same number of elements"},
                QString("%1 vs %2").arg(name1, name2),
                "Size Match Check"
            );
        }
        return ValidationResult::success();
    }
    
    static ValidationResult validateMinSize(size_t size, size_t minSize, const QString& fieldName) {
        if (size < minSize) {
            return ValidationResult::error(
                QString("%1 has %2 elements, minimum required is %3")
                    .arg(fieldName).arg(size).arg(minSize),
                {QString("Provide at least %1 elements").arg(minSize)},
                fieldName,
                "Minimum Size Check"
            );
        }
        return ValidationResult::success();
    }
};

// Specialized validators for different domains
class DataValidator {
public:
    static ValidationResult validateDataRows(const std::vector<PreprocessedRow>& rows);
    static ValidationResult validateLabeledEvents(const std::vector<LabeledEvent>& events);
    static ValidationResult validateDataConsistency(const std::vector<PreprocessedRow>& rows,
                                                   const std::vector<LabeledEvent>& events);
};

class MLValidator {
public:
    static ValidationResult validateMLConfig(const MLConfig& config);
    static ValidationResult validatePipelineConfig(const MLPipeline::UnifiedPipelineConfig& config);
    static ValidationResult validateFeatureSelection(const QSet<QString>& features);
    static ValidationResult validateModelInputs(const std::vector<std::vector<float>>& X,
                                              const std::vector<float>& y);
    static ValidationResult validateModelInputs(const std::vector<std::vector<float>>& X,
                                              const std::vector<double>& y);
    // Overloads for feature extraction results
    static ValidationResult validateModelInputs(const std::vector<std::map<std::string, double>>& X,
                                              const std::vector<int>& y);
    static ValidationResult validateModelInputs(const std::vector<std::map<std::string, double>>& X,
                                              const std::vector<double>& y);
};

class BarrierValidator {
public:
    static ValidationResult validateBarrierConfig(const BarrierConfig& config);
    static ValidationResult validateBarrierParameters(double profitMultiple, double stopMultiple, 
                                                     int verticalWindow);
};

// Validation chain for complex validations
class ValidationChain {
public:
    ValidationChain& add(std::function<ValidationResult()> validator) {
        validators_.push_back(validator);
        return *this;
    }
    
    ValidationResult execute(bool stopOnFirstError = false) {
        ValidationAccumulator accumulator;
        
        for (const auto& validator : validators_) {
            ValidationResult result = validator();
            accumulator.addResult(result);
            
            if (stopOnFirstError && !result.isValid) {
                break;
            }
        }
        
        return accumulator.getSummary();
    }
    
private:
    std::vector<std::function<ValidationResult()>> validators_;
};

// Exception-safe validation wrapper
template<typename T>
class SafeValidator {
public:
    static ValidationResult validate(std::function<T()> operation, const QString& operationName) {
        try {
            operation();
            return ValidationResult::success();
        } catch (const TripleBarrier::BaseException& e) {
            return ValidationResult::error(
                QString::fromStdString(e.what()),
                {},
                operationName,
                QString::fromStdString(e.context()),
                e.error_code()
            );
        } catch (const std::exception& e) {
            return ValidationResult::error(
                QString("Validation error: %1").arg(e.what()),
                {},
                operationName,
                "C++ Exception"
            );
        }
    }
};

// Validation configuration
struct ValidationConfig {
    bool enableWarnings = true;
    bool stopOnFirstError = false;
    bool enableSuggestions = true;
    bool enableContext = true;
    
    static ValidationConfig strict() {
        ValidationConfig config;
        config.stopOnFirstError = true;
        return config;
    }
    
    static ValidationConfig permissive() {
        ValidationConfig config;
        config.enableWarnings = false;
        return config;
    }
};

// Main validation facade
class Validator {
public:
    static void setConfig(const ValidationConfig& config) {
        config_ = config;
    }
    
    static const ValidationConfig& getConfig() {
        return config_;
    }
    
    // Convenience methods that use the configured behavior
    static ValidationResult validateData(const std::vector<PreprocessedRow>& rows,
                                        const std::vector<LabeledEvent>& events);
    
    static ValidationResult validateML(const MLConfig& config);
    
    static ValidationResult validateBarrier(const BarrierConfig& config);
    
    // Batch validation with accumulation
    static ValidationResult validateAll(const std::vector<PreprocessedRow>& rows,
                                       const std::vector<LabeledEvent>& events,
                                       const MLConfig& mlConfig,
                                       const BarrierConfig& barrierConfig);

private:
    static ValidationConfig config_;
};

} // namespace ValidationFramework
