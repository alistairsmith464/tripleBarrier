#pragma once
#include <vector>
#include <QtCharts/QChartView>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"

namespace LabeledEventPlotter {
    void plot(QChartView* chartView, const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents);
}
