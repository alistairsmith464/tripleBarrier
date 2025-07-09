#include "CSVExportUtils.h"
#include <QFile>
#include <QTextStream>
#include <QMessageBox>

namespace CSVExportUtils {

bool exportLabeledEventsToCSV(const QString& fileName,
                              const std::vector<PreprocessedRow>& rows,
                              const std::vector<LabeledEvent>& labeledEvents,
                              QString* errorMsg)
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (errorMsg) *errorMsg = "Could not open file for writing.";
        return false;
    }
    QTextStream out(&file);
    out << "timestamp,price,volatility,label,exit_time,entry_price,exit_price\n";
    for (const auto& e : labeledEvents) {
        auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
        double price = (it != rows.end()) ? it->price : e.entry_price;
        double vol = (it != rows.end()) ? it->volatility : 0.0;
        out << QString::fromStdString(e.entry_time) << ","
            << price << ","
            << vol << ","
            << e.label << ","
            << QString::fromStdString(e.exit_time) << ","
            << e.entry_price << ","
            << e.exit_price << "\n";
    }
    file.close();
    return true;
}

} 
