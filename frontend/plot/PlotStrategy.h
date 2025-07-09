#pragma once

#include <QtCharts/QChart>
#include <vector>

// Forward declarations
struct PreprocessedRow;
struct LabeledEvent;
class QChartView;

// Strategy interface for different plot types
class PlotStrategy {
public:
    virtual ~PlotStrategy() = default;
    virtual void createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                           const std::vector<LabeledEvent>& labeledEvents) = 0;
};

// Concrete strategies
class HistogramPlotStrategy : public PlotStrategy {
public:
    void createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                   const std::vector<LabeledEvent>& labeledEvents) override;
};

class TTBMDistributionPlotStrategy : public PlotStrategy {
public:
    void createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                   const std::vector<LabeledEvent>& labeledEvents) override;
};

class TimeSeriesPlotStrategy : public PlotStrategy {
public:
    void createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                   const std::vector<LabeledEvent>& labeledEvents) override;
};

class TTBMTimeSeriesPlotStrategy : public PlotStrategy {
public:
    void createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                   const std::vector<LabeledEvent>& labeledEvents) override;
};
