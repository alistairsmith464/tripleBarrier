#pragma once

#ifndef UNIFIED_ERROR_HANDLING_H
#define UNIFIED_ERROR_HANDLING_H

// Standard library includes
#include <exception>
#include <string>
#include <functional>
#include <memory>
#include <stdexcept>
#include <optional>

// Qt includes
#include <QString>
#include <QStringList>

// Backend includes
#include "../../backend/utils/Exceptions.h"
#include "../../backend/utils/ErrorHandling.h"

// Frontend includes
#include "ValidationFramework.h"

namespace UnifiedErrorHandling {

class ErrorHandler {
public:
    enum class ErrorCategory {
        Validation,     // User input validation errors
        System,         // System-level errors (file I/O, network, etc.)
        Business,       // Business logic errors
        Configuration,  // Configuration errors
        Runtime        // Runtime errors (memory, type conversion, etc.)
    };
    
    enum class ErrorSeverity {
        Info,
        Warning,
        Error,
        Critical
    };
    
    struct ErrorContext {
        ErrorCategory category;
        ErrorSeverity severity;
        QString component;
        QString operation;
        QString additionalInfo;
        
        ErrorContext(ErrorCategory cat, ErrorSeverity sev, 
                    const QString& comp = "", const QString& op = "", 
                    const QString& info = "")
            : category(cat), severity(sev), component(comp), operation(op), additionalInfo(info) {}
    };

    static ValidationFramework::ValidationResult fromException(
        const std::exception& ex, const ErrorContext& context) {
        
        QString errorMessage = QString::fromStdString(ex.what());
        QStringList suggestions;
        
        switch (context.category) {
            case ErrorCategory::Validation:
                suggestions << "Check input values" << "Verify data format";
                break;
            case ErrorCategory::System:
                suggestions << "Check system resources" << "Verify file permissions";
                break;
            case ErrorCategory::Business:
                suggestions << "Review business logic" << "Check configuration";
                break;
            case ErrorCategory::Configuration:
                suggestions << "Verify configuration settings" << "Check parameter values";
                break;
            case ErrorCategory::Runtime:
                suggestions << "Check memory usage" << "Verify data types";
                break;
        }
        
        QString fieldName = context.component;
        QString contextStr = context.operation;
        
        if (context.severity == ErrorSeverity::Critical || 
            context.severity == ErrorSeverity::Error) {
            return ValidationFramework::ValidationResult::error(
                errorMessage, suggestions, fieldName, contextStr);
        } else {
            return ValidationFramework::ValidationResult::warning(
                errorMessage, suggestions, fieldName, contextStr);
        }
    }
    
    static ValidationFramework::ValidationResult fromTripleBarrierException(
        const TripleBarrier::BaseException& ex, const ErrorContext& context) {
        
        QString errorMessage = QString::fromStdString(ex.what());
        QStringList suggestions;
        
        if (auto dataEx = dynamic_cast<const TripleBarrier::DataException*>(&ex)) {
            suggestions << "Check data quality" << "Verify data format" << "Review data source";
        } else if (auto mlEx = dynamic_cast<const TripleBarrier::MLException*>(&ex)) {
            suggestions << "Check model parameters" << "Verify training data" << "Review feature selection";
        } else if (auto configEx = dynamic_cast<const TripleBarrier::ConfigException*>(&ex)) {
            suggestions << "Check configuration values" << "Verify parameter ranges" << "Review defaults";
        } else if (auto resourceEx = dynamic_cast<const TripleBarrier::ResourceException*>(&ex)) {
            suggestions << "Check system resources" << "Verify memory usage" << "Review allocation";
        }
        
        QString fieldName = context.component;
        QString contextStr = QString("%1: %2").arg(context.operation, context.additionalInfo);
        
        return ValidationFramework::ValidationResult::error(
            errorMessage, suggestions, fieldName, contextStr);
    }
    
    static std::unique_ptr<TripleBarrier::BaseException> toTripleBarrierException(
        const ValidationFramework::ValidationResult& result, const ErrorContext& context) {
        
        if (result.isValid) {
            return nullptr;
        }
        
        std::string message = result.errorMessage.toStdString();
        std::string component = context.component.toStdString();
        
        switch (context.category) {
            case ErrorCategory::Validation:
                return std::make_unique<TripleBarrier::DataValidationException>(message, component);
            case ErrorCategory::System:
                return std::make_unique<TripleBarrier::ResourceException>(message, component);
            case ErrorCategory::Business:
                return std::make_unique<TripleBarrier::DataProcessingException>(message, component);
            case ErrorCategory::Configuration:
                return std::make_unique<TripleBarrier::InvalidConfigException>(message, component);
            case ErrorCategory::Runtime:
                return std::make_unique<TripleBarrier::BaseException>(message, component);
        }
        
        return std::make_unique<TripleBarrier::BaseException>(message, component);
    }
    
    template<typename Func>
    static ValidationFramework::ValidationResult safeExecute(
        Func&& func, const ErrorContext& context) {
        
        try {
            return func();
        } catch (const TripleBarrier::BaseException& ex) {
            return fromTripleBarrierException(ex, context);
        } catch (const std::exception& ex) {
            return fromException(ex, context);
        } catch (...) {
            return ValidationFramework::ValidationResult::error(
                "Unknown error occurred",
                {"Check system state", "Review recent operations"},
                context.component,
                context.operation
            );
        }
    }
    
    template<typename ValidationFunc>
    static ValidationFramework::ValidationResult validateSafely(
        ValidationFunc&& validator, const ErrorContext& context) {
        
        return safeExecute(std::forward<ValidationFunc>(validator), context);
    }
    
    /**
     * @brief Check if error should be propagated as exception
     */
    static bool shouldPropagateAsException(const ErrorContext& context) {
        return context.severity == ErrorSeverity::Critical ||
               context.category == ErrorCategory::System ||
               context.category == ErrorCategory::Runtime;
    }
    
    static void logError(const ValidationFramework::ValidationResult& result, 
                        const ErrorContext& context) {
        if (!result.isValid) {
            QString logMessage = QString("[%1] %2: %3")
                .arg(severityToString(context.severity))
                .arg(context.component)
                .arg(result.errorMessage);
            
            fprintf(stderr, "%s\n", logMessage.toStdString().c_str());
        }
    }
    
    static QString categoryToString(ErrorCategory category) {
        switch (category) {
            case ErrorCategory::Validation: return "Validation";
            case ErrorCategory::System: return "System";
            case ErrorCategory::Business: return "Business";
            case ErrorCategory::Configuration: return "Configuration";
            case ErrorCategory::Runtime: return "Runtime";
        }
        return "Unknown";
    }
    
    static QString severityToString(ErrorSeverity severity) {
        switch (severity) {
            case ErrorSeverity::Info: return "Info";
            case ErrorSeverity::Warning: return "Warning";
            case ErrorSeverity::Error: return "Error";
            case ErrorSeverity::Critical: return "Critical";
        }
        return "Unknown";
    }
};

class ValidationScope {
public:
    ValidationScope(const ErrorHandler::ErrorContext& context) 
        : context_(context), hasError_(false) {}
    
    ~ValidationScope() {
        if (hasError_ && result_.has_value()) {
            ErrorHandler::logError(result_.value(), context_);
        }
    }
    
    template<typename ValidationFunc>
    ValidationFramework::ValidationResult validate(ValidationFunc&& validator) {
        result_ = ErrorHandler::validateSafely(std::forward<ValidationFunc>(validator), context_);
        hasError_ = !result_->isValid;
        return result_.value();
    }
    
    bool hasError() const { return hasError_; }
    
private:
    ErrorHandler::ErrorContext context_;
    std::optional<ValidationFramework::ValidationResult> result_;
    bool hasError_;
};

} // namespace UnifiedErrorHandling

#endif // UNIFIED_ERROR_HANDLING_H
