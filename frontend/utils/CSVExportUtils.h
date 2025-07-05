#pragma once
#include <QString>
#include <vector>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"

namespace CSVExportUtils {
    // Exports labeled events and their corresponding rows to a CSV file.
    // Returns true on success, false on failure.
    bool exportLabeledEventsToCSV(const QString& fileName,
                                  const std::vector<PreprocessedRow>& rows,
                                  const std::vector<LabeledEvent>& labeledEvents,
                                  QString* errorMsg = nullptr);
}
