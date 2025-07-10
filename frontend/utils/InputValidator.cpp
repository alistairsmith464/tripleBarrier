#include "InputValidator.h"
#include <QFile>
#include <QFileInfo>
#include <cmath>

ValidationResult InputValidator::validateBarrierConfig(const BarrierConfig& config) {
    // Validate profit multiple
    if (config.profit_multiple <= 0) {
        return ValidationResult::error(
            "Profit multiple must be positive",
            {"Try values between 1.0 and 5.0", "Common values: 2.0, 2.5, 3.0"}
        );
    }
    
    if (config.profit_multiple > 10.0) {
        return ValidationResult::warning(
            "Profit multiple is very high - this may result in few profitable trades",
            {"Consider values between 1.5 and 3.0 for better balance"}
        );
    }
    
    // Validate stop multiple
    if (config.stop_multiple <= 0) {
        return ValidationResult::error(
            "Stop multiple must be positive",
            {"Try values between 0.5 and 2.0", "Common values: 1.0, 1.5"}
        );
    }
    
    // Validate vertical window
    if (config.vertical_window <= 0) {
        return ValidationResult::error(
            "Vertical window must be positive",
            {"Try values between 10 and 100", "Common values: 20, 50"}
        );
    }
    
    if (config.vertical_window > 1000) {
        return ValidationResult::warning(
            "Vertical window is very large - this may affect performance",
            {"Consider values between 20 and 200"}
        );
    }
    
    // Validate CUSUM threshold if enabled
    if (config.use_cusum && config.cusum_threshold <= 0) {
        return ValidationResult::error(
            "CUSUM threshold must be positive when CUSUM is enabled",
            {"Try values between 2.0 and 10.0", "Higher values = fewer events"}
        );
    }
    
    // Validate ratio between profit and stop
    if (config.profit_multiple < config.stop_multiple) {
        return ValidationResult::warning(
            "Profit multiple is smaller than stop multiple - this may result in poor risk/reward ratio",
            {"Consider making profit multiple at least equal to stop multiple"}
        );
    }
    
    return ValidationResult::success();
}

ValidationResult InputValidator::validateFeatureSelection(const QSet<QString>& features) {
    if (features.isEmpty()) {
        return ValidationResult::error(
            "No features selected",
            {"Select at least one feature for analysis", "Recommended: Start with 3-5 features"}
        );
    }
    
    if (features.size() > 50) {
        return ValidationResult::warning(
            "Large number of features selected - this may lead to overfitting",
            {"Consider feature selection techniques", "Start with 5-15 most relevant features"}
        );
    }
    
    // Check for valid features
    QStringList availableFeatures = getAvailableFeatures();
    QStringList invalidFeatures;
    
    for (const QString& feature : features) {
        if (!isValidFeature(feature)) {
            invalidFeatures.append(feature);
        }
    }
    
    if (!invalidFeatures.isEmpty()) {
        return ValidationResult::error(
            QString("Invalid features selected: %1").arg(invalidFeatures.join(", ")),
            {"Check feature names", "Use only available features from the list"}
        );
    }
    
    return ValidationResult::success();
}

ValidationResult InputValidator::validateFilePath(const QString& filePath, bool mustExist) {
    if (filePath.isEmpty()) {
        return ValidationResult::error(
            "File path is empty",
            {"Select a valid file", "Browse for files using the file dialog"}
        );
    }
    
    QFileInfo fileInfo(filePath);
    
    if (mustExist && !fileInfo.exists()) {
        return ValidationResult::error(
            QString("File does not exist: %1").arg(filePath),
            {"Check the file path", "Browse for an existing file"}
        );
    }
    
    if (mustExist && !fileInfo.isFile()) {
        return ValidationResult::error(
            QString("Path is not a file: %1").arg(filePath),
            {"Select a file, not a directory"}
        );
    }
    
    if (mustExist && !fileInfo.isReadable()) {
        return ValidationResult::error(
            QString("File is not readable: %1").arg(filePath),
            {"Check file permissions", "Select a different file"}
        );
    }
    
    // Check file extension for CSV files
    if (fileInfo.suffix().toLower() != "csv") {
        return ValidationResult::warning(
            "File does not have .csv extension",
            {"Ensure the file is in CSV format", "Check file content"}
        );
    }
    
    return ValidationResult::success();
}

ValidationResult InputValidator::validateRange(double value, double min, double max, 
                                              const QString& fieldName) {
    if (std::isnan(value)) {
        return ValidationResult::error(
            QString("%1 is not a valid number").arg(fieldName),
            {"Enter a numeric value"}
        );
    }
    
    if (value < min) {
        return ValidationResult::error(
            QString("%1 is below minimum value (%2)").arg(fieldName).arg(min),
            {QString("Enter a value >= %1").arg(min)}
        );
    }
    
    if (value > max) {
        return ValidationResult::error(
            QString("%1 is above maximum value (%2)").arg(fieldName).arg(max),
            {QString("Enter a value <= %1").arg(max)}
        );
    }
    
    return ValidationResult::success();
}

ValidationResult InputValidator::validateMLConfig(const MLPipeline::UnifiedPipelineConfig& config) {
    // Validate test size
    auto testSizeResult = validateRange(config.test_size, 0.05, 0.5, "Test size");
    if (!testSizeResult.isValid) {
        return testSizeResult;
    }
    
    // Validate validation size
    auto valSizeResult = validateRange(config.val_size, 0.0, 0.5, "Validation size");
    if (!valSizeResult.isValid) {
        return valSizeResult;
    }
    
    // Check total size
    if (config.test_size + config.val_size >= 1.0) {
        return ValidationResult::error(
            "Test size + Validation size must be less than 1.0",
            {"Reduce test size or validation size", "Common split: 0.2 test, 0.2 validation"}
        );
    }
    
    // Validate n_rounds
    if (config.n_rounds <= 0) {
        return ValidationResult::error(
            "Number of rounds must be positive",
            {"Try values between 50 and 500", "Common values: 100, 200"}
        );
    }
    
    if (config.n_rounds > 1000) {
        return ValidationResult::warning(
            "Large number of rounds - training may take a long time",
            {"Consider values between 100 and 500"}
        );
    }
    
    // Validate max_depth
    auto depthResult = validateRange(config.max_depth, 1, 20, "Max depth");
    if (!depthResult.isValid) {
        return depthResult;
    }
    
    // Validate learning rate
    auto lrResult = validateRange(config.learning_rate, 0.001, 1.0, "Learning rate");
    if (!lrResult.isValid) {
        return lrResult;
    }
    
    return ValidationResult::success();
}

ValidationResult InputValidator::validateDataSize(size_t dataSize, size_t minRequired) {
    if (dataSize < minRequired) {
        return ValidationResult::error(
            QString("Insufficient data: %1 samples (minimum %2 required)")
                .arg(dataSize).arg(minRequired),
            {"Load more data", "Use a larger dataset"}
        );
    }
    
    if (dataSize < 100) {
        return ValidationResult::warning(
            QString("Small dataset: %1 samples").arg(dataSize),
            {"Consider using more data for better results", "Results may not be reliable"}
        );
    }
    
    return ValidationResult::success();
}

ValidationResult InputValidator::validateWindowSize(int windowSize, int dataSize) {
    if (windowSize <= 0) {
        return ValidationResult::error(
            "Window size must be positive",
            {"Try values between 10 and 100"}
        );
    }
    
    if (windowSize >= dataSize) {
        return ValidationResult::error(
            QString("Window size (%1) must be smaller than data size (%2)")
                .arg(windowSize).arg(dataSize),
            {"Reduce window size", "Use more data"}
        );
    }
    
    if (windowSize > dataSize / 2) {
        return ValidationResult::warning(
            "Window size is large relative to data size",
            {"Consider reducing window size", "Use more data if available"}
        );
    }
    
    return ValidationResult::success();
}

QStringList InputValidator::getAvailableFeatures() {
    // Return list of known valid features
    return {
        "returns", "volatility", "volume", "price", "sma_10", "sma_20", "sma_50",
        "ema_10", "ema_20", "rsi", "macd", "bollinger_upper", "bollinger_lower",
        "momentum", "roc", "williams_r", "stoch_k", "stoch_d"
    };
}

bool InputValidator::isValidFeature(const QString& feature) {
    QStringList availableFeatures = getAvailableFeatures();
    return availableFeatures.contains(feature);
}
