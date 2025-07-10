#include "ErrorHandler.h"
#include <QMessageBox>
#include <QDateTime>
#include <QFile>
#include <QTextStream>
#include <QDebug>

// Static member definitions
std::map<ErrorHandler::ErrorType, ErrorHandler::RecoveryCallback> ErrorHandler::s_recoveryCallbacks;
QString ErrorHandler::s_logFilePath;
bool ErrorHandler::s_loggingEnabled = false;

void ErrorHandler::handleError(const ErrorInfo& errorInfo, QWidget* parent) {
    // Log the error if logging is enabled
    if (s_loggingEnabled) {
        logError(errorInfo);
    }
    
    // Show error dialog to user
    showErrorDialog(errorInfo, parent);
    
    // Attempt recovery if available
    attemptRecovery(errorInfo);
}

void ErrorHandler::handleException(const std::exception& ex, ErrorType type, QWidget* parent) {
    QString message = formatError(ex);
    ErrorInfo errorInfo(type, Severity::Error, message);
    handleError(errorInfo, parent);
}

QString ErrorHandler::formatError(const std::exception& ex) {
    return QString("Exception: %1").arg(ex.what());
}

QString ErrorHandler::getErrorTypeString(ErrorType type) {
    switch (type) {
        case ErrorType::DataLoad: return "Data Loading";
        case ErrorType::Processing: return "Data Processing";
        case ErrorType::ML: return "Machine Learning";
        case ErrorType::UI: return "User Interface";
        case ErrorType::Configuration: return "Configuration";
        case ErrorType::Network: return "Network";
        case ErrorType::FileSystem: return "File System";
        default: return "Unknown";
    }
}

QString ErrorHandler::getSeverityString(Severity severity) {
    switch (severity) {
        case Severity::Info: return "Info";
        case Severity::Warning: return "Warning";
        case Severity::Error: return "Error";
        case Severity::Critical: return "Critical";
        default: return "Unknown";
    }
}

void ErrorHandler::setRecoveryCallback(ErrorType type, RecoveryCallback callback) {
    s_recoveryCallbacks[type] = callback;
}

void ErrorHandler::enableLogging(const QString& logFilePath) {
    s_logFilePath = logFilePath;
    s_loggingEnabled = true;
}

void ErrorHandler::logError(const ErrorInfo& errorInfo) {
    if (!s_loggingEnabled || s_logFilePath.isEmpty()) {
        return;
    }
    
    QFile logFile(s_logFilePath);
    if (logFile.open(QIODevice::WriteOnly | QIODevice::Append)) {
        QTextStream stream(&logFile);
        stream << QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
               << " [" << getSeverityString(errorInfo.severity) << "]"
               << " [" << getErrorTypeString(errorInfo.type) << "]"
               << " " << errorInfo.message;
        if (!errorInfo.details.isEmpty()) {
            stream << " - Details: " << errorInfo.details;
        }
        stream << "\n";
    }
}

void ErrorHandler::showErrorDialog(const ErrorInfo& errorInfo, QWidget* parent) {
    QString title = QString("%1 - %2").arg(getSeverityString(errorInfo.severity), 
                                          getErrorTypeString(errorInfo.type));
    
    QString message = errorInfo.message;
    if (!errorInfo.details.isEmpty()) {
        message += QString("\n\nDetails: %1").arg(errorInfo.details);
    }
    if (!errorInfo.suggestion.isEmpty()) {
        message += QString("\n\nSuggestion: %1").arg(errorInfo.suggestion);
    }
    
    QMessageBox::Icon icon;
    switch (errorInfo.severity) {
        case Severity::Info:
            icon = QMessageBox::Information;
            break;
        case Severity::Warning:
            icon = QMessageBox::Warning;
            break;
        case Severity::Error:
        case Severity::Critical:
            icon = QMessageBox::Critical;
            break;
    }
    
    QMessageBox msgBox(icon, title, message, QMessageBox::Ok, parent);
    msgBox.exec();
}

void ErrorHandler::attemptRecovery(const ErrorInfo& errorInfo) {
    auto it = s_recoveryCallbacks.find(errorInfo.type);
    if (it != s_recoveryCallbacks.end() && it->second) {
        try {
            bool recovered = it->second();
            if (recovered) {
                qDebug() << "Recovery successful for error type:" << getErrorTypeString(errorInfo.type);
            } else {
                qDebug() << "Recovery failed for error type:" << getErrorTypeString(errorInfo.type);
            }
        } catch (const std::exception& ex) {
            qDebug() << "Recovery callback threw exception:" << ex.what();
        }
    }
}
