#pragma once

#include <QString>
#include <QDateTime>
#include <vector>

class DateParsingUtils {
public:
    static QDateTime parseTimestamp(const QString& timestamp);

    static std::vector<QString> getSupportedFormats();
    
private:
    static const std::vector<QString> DATE_FORMATS;
};
