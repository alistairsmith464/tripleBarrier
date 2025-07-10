#pragma once
#include <QString>
#include <QWidget>
#include <exception>
#include <functional>
#include <map>

class ErrorHandler {
public:
    enum class ErrorType { 
        DataLoad, 
        Processing, 
        ML, 
        UI, 
        Configuration,
        Network,
        FileSystem
    };
    
    enum class Severity {
        Info,
        Warning, 
        Error,
        Critical
    };
    
    struct ErrorInfo {
        ErrorType type;
        Severity severity;
        QString message;
        QString details;
        QString suggestion;
        
        ErrorInfo(ErrorType t, Severity s, const QString& msg, 
                 const QString& det = "", const QString& sug = "")
            : type(t), severity(s), message(msg), details(det), suggestion(sug) {}
    };
    
    // Main error handling methods
    static void handleError(const ErrorInfo& errorInfo, QWidget* parent = nullptr);
    static void handleException(const std::exception& ex, ErrorType type, QWidget* parent = nullptr);
    
    // Utility methods
    static QString formatError(const std::exception& ex);
    static QString getErrorTypeString(ErrorType type);
    static QString getSeverityString(Severity severity);
    
    // Error recovery callbacks
    using RecoveryCallback = std::function<bool()>;
    static void setRecoveryCallback(ErrorType type, RecoveryCallback callback);
    
    // Logging
    static void enableLogging(const QString& logFilePath);
    static void logError(const ErrorInfo& errorInfo);
    
private:
    static void showErrorDialog(const ErrorInfo& errorInfo, QWidget* parent);
    static void attemptRecovery(const ErrorInfo& errorInfo);
    
    static std::map<ErrorType, RecoveryCallback> s_recoveryCallbacks;
    static QString s_logFilePath;
    static bool s_loggingEnabled;
};
