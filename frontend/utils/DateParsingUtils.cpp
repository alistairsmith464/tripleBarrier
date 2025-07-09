#include "DateParsingUtils.h"
#include <QDateTime>

const std::vector<QString> DateParsingUtils::DATE_FORMATS = {
    "yyyy-MM-ddTHH:mm:ss",  // ISO format as string
    "yyyy-MM-dd HH:mm:ss",
    "yyyy/MM/dd HH:mm:ss",
    "dd/MM/yyyy HH:mm:ss",
    "MM/dd/yyyy HH:mm:ss",
    "M/d/yyyy H:mm:ss"
};

QDateTime DateParsingUtils::parseTimestamp(const QString& timestamp) {
    // Try ISO date first
    QDateTime dt = QDateTime::fromString(timestamp, Qt::ISODate);
    if (dt.isValid()) {
        return dt;
    }
    
    // Try other formats
    for (const QString& format : DATE_FORMATS) {
        dt = QDateTime::fromString(timestamp, format);
        if (dt.isValid()) {
            return dt;
        }
    }
    
    return QDateTime(); // Invalid
}

std::vector<QString> DateParsingUtils::getSupportedFormats() {
    return DATE_FORMATS;
}
