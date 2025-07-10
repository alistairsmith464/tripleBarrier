#include "LabeledEventPlotter.h"
#include "PlotStrategy.h"
#include <QtCharts/QChart>
#include <QMessageBox>
#include <memory>

namespace LabeledEventPlotter {

void plot(QChartView* chartView, const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents, PlotMode mode) {
    QChart *chart = new QChart();
    
    std::unique_ptr<PlotStrategy> strategy;
    
    switch (mode) {
        case PlotMode::Histogram:
            strategy = std::make_unique<HistogramPlotStrategy>();
            break;
        case PlotMode::TTBM_Distribution:
            strategy = std::make_unique<TTBMDistributionPlotStrategy>();
            break;
        case PlotMode::TTBM_TimeSeries:
            strategy = std::make_unique<TTBMTimeSeriesPlotStrategy>();
            break;
        case PlotMode::TimeSeries:
        default:
            strategy = std::make_unique<TimeSeriesPlotStrategy>();
            break;
    }
    
    strategy->createPlot(chart, rows, labeledEvents);
    chartView->setChart(chart);
    
    bool hasValidData = !rows.empty() && !labeledEvents.empty();
    if (!hasValidData) {
        QMessageBox::warning(chartView, "Chart Error", "No valid data found. Check your CSV format.");
    }
}

}
