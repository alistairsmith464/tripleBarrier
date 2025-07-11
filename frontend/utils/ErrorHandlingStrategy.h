#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <QString>
#include <QWidget>
#include "ValidationFramework.h"
#include "../../backend/utils/Exceptions.h"
#include "../../backend/utils/ErrorHandling.h"

namespace ValidationFramework {

// Unified error handling strategy
class ErrorHandlingStrategy {
public:
    enum class Mode {
        EXCEPTIONS,      // Always throw exceptions
        RETURN_CODES,    // Return error codes/status
        MIXED           // Use exceptions for critical errors, return codes for validation
    };
    
    enum class Severity {
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };
    
    struct ErrorContext {
        QString operation;
        QString component;
        QString userMessage;
        QString technicalMessage;
        QString suggestions;
        Severity severity;
        int errorCode;
        
        ErrorContext(const QString& op = "", 
                    const QString& comp = "",
                    Severity sev = Severity::ERROR,
                    int code = 0)
            : operation(op), component(comp), severity(sev), errorCode(code) {}
    };
    
    static void setMode(Mode mode) { mode_ = mode; }
    static Mode getMode() { return mode_; }
    
    // Handle validation result according to current mode
    static void handleValidationResult(const ValidationResult& result, 
                                     const ErrorContext& context);
    
    // Handle exception according to current mode
    static void handleException(const std::exception& e, 
                              const ErrorContext& context);
    
    // Create user-friendly error messages
    static QString createUserMessage(const ValidationResult& result, 
                                   const ErrorContext& context);
    
    // Create technical error messages
    static QString createTechnicalMessage(const ValidationResult& result, 
                                        const ErrorContext& context);
    
    // Convert between different error representations
    static TripleBarrier::BaseException toException(const ValidationResult& result, 
                                                  const ErrorContext& context);
    
    static ValidationResult fromException(const TripleBarrier::BaseException& exception);
    
private:
    static Mode mode_;
};

// Validation-aware function wrapper
template<typename T>
class ValidatedFunction {
public:
    using FunctionType = std::function<T()>;
    using ValidationFunction = std::function<ValidationResult()>;
    
    ValidatedFunction(FunctionType func, 
                     ValidationFunction preValidation = nullptr,
                     ValidationFunction postValidation = nullptr,
                     ErrorHandlingStrategy::ErrorContext context = {})
        : function_(func), preValidation_(preValidation), 
          postValidation_(postValidation), context_(context) {}
    
    T execute() {
        // Pre-validation
        if (preValidation_) {
            ValidationResult preResult = preValidation_();
            if (!preResult.isValid) {
                ErrorHandlingStrategy::handleValidationResult(preResult, context_);
                // If we reach here, error handling strategy allowed continuation
            }
        }
        
        // Execute main function
        T result;
        try {
            result = function_();
        } catch (const std::exception& e) {
            ErrorHandlingStrategy::handleException(e, context_);
            throw; // Re-throw after handling
        }
        
        // Post-validation
        if (postValidation_) {
            ValidationResult postResult = postValidation_();
            if (!postResult.isValid) {
                ErrorHandlingStrategy::handleValidationResult(postResult, context_);
                // If we reach here, error handling strategy allowed continuation
            }
        }
        
        return result;
    }
    
private:
    FunctionType function_;
    ValidationFunction preValidation_;
    ValidationFunction postValidation_;
    ErrorHandlingStrategy::ErrorContext context_;
};

// Builder for creating validated functions
template<typename T>
class ValidatedFunctionBuilder {
public:
    using FunctionType = std::function<T()>;
    using ValidationFunction = std::function<ValidationResult()>;
    
    ValidatedFunctionBuilder(FunctionType func) : function_(func) {}
    
    ValidatedFunctionBuilder& withPreValidation(ValidationFunction validation) {
        preValidation_ = validation;
        return *this;
    }
    
    ValidatedFunctionBuilder& withPostValidation(ValidationFunction validation) {
        postValidation_ = validation;
        return *this;
    }
    
    ValidatedFunctionBuilder& withContext(const ErrorHandlingStrategy::ErrorContext& context) {
        context_ = context;
        return *this;
    }
    
    ValidatedFunction<T> build() {
        return ValidatedFunction<T>(function_, preValidation_, postValidation_, context_);
    }
    
private:
    FunctionType function_;
    ValidationFunction preValidation_;
    ValidationFunction postValidation_;
    ErrorHandlingStrategy::ErrorContext context_;
};

// Helper functions for creating validated functions
template<typename T>
ValidatedFunctionBuilder<T> createValidatedFunction(std::function<T()> func) {
    return ValidatedFunctionBuilder<T>(func);
}

// Error aggregation for batch operations
class ErrorAggregator {
public:
    struct AggregatedError {
        std::vector<ValidationResult> validationErrors;
        std::vector<std::pair<std::exception_ptr, ErrorHandlingStrategy::ErrorContext>> exceptions;
        
        bool hasErrors() const {
            return !validationErrors.empty() || !exceptions.empty();
        }
        
        size_t totalErrors() const {
            size_t count = 0;
            for (const auto& result : validationErrors) {
                if (!result.isValid) count++;
            }
            count += exceptions.size();
            return count;
        }
        
        QString getSummary() const {
            if (!hasErrors()) return "No errors";
            
            QString summary = QString("Total errors: %1").arg(totalErrors());
            
            if (!validationErrors.empty()) {
                summary += QString("\nValidation errors: %1").arg(validationErrors.size());
            }
            
            if (!exceptions.empty()) {
                summary += QString("\nExceptions: %1").arg(exceptions.size());
            }
            
            return summary;
        }
    };
    
    void addValidationResult(const ValidationResult& result) {
        if (!result.isValid) {
            aggregatedError_.validationErrors.push_back(result);
        }
    }
    
    void addException(const std::exception& e, 
                     const ErrorHandlingStrategy::ErrorContext& context) {
        aggregatedError_.exceptions.push_back({std::current_exception(), context});
    }
    
    const AggregatedError& getAggregatedError() const {
        return aggregatedError_;
    }
    
    bool hasErrors() const {
        return aggregatedError_.hasErrors();
    }
    
    void clear() {
        aggregatedError_.validationErrors.clear();
        aggregatedError_.exceptions.clear();
    }
    
private:
    AggregatedError aggregatedError_;
};

// Pre-condition and post-condition validation macros
#define VALIDATE_PRE(condition, message) \
    do { \
        if (!(condition)) { \
            auto result = ValidationResult::error(message, {}, "Pre-condition", "Pre-condition Check"); \
            ErrorHandlingStrategy::handleValidationResult(result, \
                ErrorHandlingStrategy::ErrorContext(__func__, __FILE__, \
                    ErrorHandlingStrategy::Severity::CRITICAL)); \
        } \
    } while(0)

#define VALIDATE_POST(condition, message) \
    do { \
        if (!(condition)) { \
            auto result = ValidationResult::error(message, {}, "Post-condition", "Post-condition Check"); \
            ErrorHandlingStrategy::handleValidationResult(result, \
                ErrorHandlingStrategy::ErrorContext(__func__, __FILE__, \
                    ErrorHandlingStrategy::Severity::CRITICAL)); \
        } \
    } while(0)

// Safe execution wrapper that handles all error types
template<typename T>
class SafeExecutor {
public:
    using ExecutionFunction = std::function<T()>;
    using ErrorHandler = std::function<void(const std::exception&, const ErrorHandlingStrategy::ErrorContext&)>;
    
    SafeExecutor(ExecutionFunction func, 
                const ErrorHandlingStrategy::ErrorContext& context)
        : function_(func), context_(context) {}
    
    SafeExecutor& withErrorHandler(ErrorHandler handler) {
        errorHandler_ = handler;
        return *this;
    }
    
    SafeExecutor& withFallback(T fallbackValue) {
        fallbackValue_ = fallbackValue;
        hasFallback_ = true;
        return *this;
    }
    
    T execute() {
        try {
            return function_();
        } catch (const TripleBarrier::BaseException& e) {
            if (errorHandler_) {
                errorHandler_(e, context_);
            } else {
                ErrorHandlingStrategy::handleException(e, context_);
            }
            
            if (hasFallback_) {
                return fallbackValue_;
            }
            throw;
        } catch (const std::exception& e) {
            if (errorHandler_) {
                errorHandler_(e, context_);
            } else {
                ErrorHandlingStrategy::handleException(e, context_);
            }
            
            if (hasFallback_) {
                return fallbackValue_;
            }
            throw;
        }
    }
    
private:
    ExecutionFunction function_;
    ErrorHandlingStrategy::ErrorContext context_;
    ErrorHandler errorHandler_;
    T fallbackValue_;
    bool hasFallback_ = false;
};

// Helper function for creating safe executors
template<typename T>
SafeExecutor<T> createSafeExecutor(std::function<T()> func, 
                                  const ErrorHandlingStrategy::ErrorContext& context) {
    return SafeExecutor<T>(func, context);
}

} // namespace ValidationFramework
