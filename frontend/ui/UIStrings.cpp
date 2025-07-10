#include "UIStrings.h"

QString UIStringHelper::uploadSuccessMessage(const QString& filePath) {
    return QString("Successfully loaded data from: %1").arg(filePath);
}

QString UIStringHelper::loadErrorMessage(const QString& error) {
    return QString("Failed to load data: %1").arg(error);
}

QString UIStringHelper::featureCountMessage(int count) {
    return QString("%1 feature%2 selected").arg(count).arg(count == 1 ? "" : "s");
}

QString UIStringHelper::dataPointsMessage(int count) {
    return QString("%1 data point%2 loaded").arg(count).arg(count == 1 ? "" : "s");
}

QString UIStringHelper::accuracyMessage(double accuracy) {
    return QString("Model accuracy: %1%").arg(QString::number(accuracy * 100, 'f', 2));
}

QString UIStringHelper::processingTimeMessage(int seconds) {
    if (seconds < 60) {
        return QString("Processing completed in %1 second%2").arg(seconds).arg(seconds == 1 ? "" : "s");
    } else {
        int minutes = seconds / 60;
        int remainingSeconds = seconds % 60;
        return QString("Processing completed in %1m %2s").arg(minutes).arg(remainingSeconds);
    }
}
