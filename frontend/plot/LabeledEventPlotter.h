#pragma once
#include <vector>
#include <QtCharts/QChartView>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"
#include <optional>

enum class PlotMode {
    TimeSeries,
    Histogram,
    TTBM_TimeSeries,  
    TTBM_Distribution  
};

namespace LabeledEventPlotter {
    void plot(QChartView* chartView, const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents, PlotMode mode = PlotMode::TimeSeries);
}
