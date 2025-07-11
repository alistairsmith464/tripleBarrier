#include "ErrorHandlingStrategy.h"
#include "ErrorHandler.h"
#include <QDebug>
#include <QMessageBox>

namespace ValidationFramework {

// Static member initialization
ErrorHandlingStrategy::Mode ErrorHandlingStrategy::mode_ = Mode::MIXED;

void ErrorHandlingStrategy::handleValidationResult(const ValidationResult& result, 
                                                  const ErrorContext& context) {
    if (result.isValid) {
        // Handle warnings if present
        if (!result.warningMessage.isEmpty()) {
            QString userMsg = createUserMessage(result, context);
            qDebug() << "Warning:" << userMsg;
        }
        return;
    }
    
    // Handle errors based on current mode
    switch (mode_) {
        case Mode::EXCEPTIONS:
            throw toException(result, context);
            
        case Mode::RETURN_CODES:
            // Log error but don't throw
            qDebug() << "Error:" << createTechnicalMessage(result, context);
            break;
            
        case Mode::MIXED:
            // Throw for critical errors, log for others
            if (context.severity == Severity::CRITICAL) {
                throw toException(result, context);
            } else {
                qDebug() << "Error:" << createTechnicalMessage(result, context);
            }
            break;
    }
}

void ErrorHandlingStrategy::handleException(const std::exception& e, 
                                          const ErrorContext& context) {
    // Convert exception to validation result for consistent handling
    ValidationResult result = ValidationResult::error(
        QString::fromStdString(e.what()),
        {},
        context.operation,
        context.component
    );
    
    // Log the exception
    qDebug() << "Exception in" << context.operation << ":" << e.what();
    
    // Handle based on severity
    switch (context.severity) {
        case Severity::CRITICAL:
            // Always propagate critical exceptions
            throw;
            
        case Severity::ERROR:
            // Handle based on current mode
            if (mode_ == Mode::EXCEPTIONS || mode_ == Mode::MIXED) {
                throw;
            }
            break;
            
        case Severity::WARNING:
        case Severity::INFO:
            // Log but don't propagate
            break;
    }
}

QString ErrorHandlingStrategy::createUserMessage(const ValidationResult& result, 
                                                const ErrorContext& context) {
    QString message;
    
    if (!result.isValid) {
        message = result.errorMessage;
    } else if (!result.warningMessage.isEmpty()) {
        message = result.warningMessage;
    }
    
    // Add context information
    if (!context.operation.isEmpty()) {
        message = QString("%1 in %2").arg(message, context.operation);
    }
    
    // Add suggestions if available
    if (!result.suggestions.isEmpty()) {
        message += QString("\n\nSuggestions:\n• %1").arg(result.suggestions.join("\n• "));
    }
    
    return message;
}

QString ErrorHandlingStrategy::createTechnicalMessage(const ValidationResult& result, 
                                                    const ErrorContext& context) {
    QString message;
    
    if (!result.isValid) {
        message = QString("ERROR: %1").arg(result.errorMessage);
    } else if (!result.warningMessage.isEmpty()) {
        message = QString("WARNING: %1").arg(result.warningMessage);
    }
    
    // Add technical context
    if (!context.component.isEmpty()) {
        message += QString(" [Component: %1]").arg(context.component);
    }
    
    if (!context.operation.isEmpty()) {
        message += QString(" [Operation: %1]").arg(context.operation);
    }
    
    if (!result.fieldName.isEmpty()) {
        message += QString(" [Field: %1]").arg(result.fieldName);
    }
    
    if (!result.context.isEmpty()) {
        message += QString(" [Context: %1]").arg(result.context);
    }
    
    if (result.errorCode != 0) {
        message += QString(" [Code: %1]").arg(result.errorCode);
    }
    
    return message;
}

TripleBarrier::BaseException ErrorHandlingStrategy::toException(const ValidationResult& result, 
                                                              const ErrorContext& context) {
    std::string message = result.errorMessage.toStdString();
    std::string contextStr = context.operation.toStdString();
    
    if (!context.component.isEmpty()) {
        contextStr += " [" + context.component.toStdString() + "]";
    }
    
    // Create specific exception type based on content
    if (result.fieldName.contains("feature", Qt::CaseInsensitive) ||
        result.context.contains("feature", Qt::CaseInsensitive)) {
        return TripleBarrier::FeatureExtractionException(message, contextStr);
    } else if (result.fieldName.contains("model", Qt::CaseInsensitive) ||
               result.context.contains("model", Qt::CaseInsensitive)) {
        return TripleBarrier::ModelTrainingException(message, contextStr);
    } else if (result.fieldName.contains("config", Qt::CaseInsensitive) ||
               result.context.contains("config", Qt::CaseInsensitive)) {
        return TripleBarrier::ConfigException(message, contextStr);
    } else if (result.context.contains("validation", Qt::CaseInsensitive)) {
        return TripleBarrier::DataValidationException(message, contextStr);
    } else {
        return TripleBarrier::BaseException(message, contextStr, result.errorCode);
    }
}

ValidationResult ErrorHandlingStrategy::fromException(const TripleBarrier::BaseException& exception) {
    return ValidationResult::error(
        QString::fromStdString(exception.what()),
        {},
        QString::fromStdString(exception.context()),
        "Exception Conversion",
        exception.error_code()
    );
}

} // namespace ValidationFramework
