#pragma once
#include <vector>
#include <QtCharts/QChartView>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"
#include <optional>

// Plot mode for event visualization
enum class PlotMode {
    TimeSeries,
    Histogram
};

namespace LabeledEventPlotter {
    // If soft labels are present, will use them for gradient coloring
    void plot(QChartView* chartView, const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents, PlotMode mode = PlotMode::TimeSeries);
}
