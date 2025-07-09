#pragma once

#include <QString>
#include <QDateTime>
#include <vector>

class DateParsingUtils {
public:
    // Parse timestamp with multiple format attempts
    static QDateTime parseTimestamp(const QString& timestamp);
    
    // Get all supported date formats
    static std::vector<QString> getSupportedFormats();
    
private:
    static const std::vector<QString> DATE_FORMATS;
};
