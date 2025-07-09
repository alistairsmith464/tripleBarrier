#include "PlotStrategy.h"
#include "../config/VisualizationConfig.h"
#include "../utils/DateParsingUtils.h"
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QValueAxis>
#include <QtCharts/QDateTimeAxis>
#include <QDateTime>
#include <algorithm>
#include <cmath>

namespace {
    QColor ttbmLabelToColor(double ttbm_label) {
        double normalized = (ttbm_label + 1.0) / 2.0;
        normalized = std::max(0.0, std::min(1.0, normalized));
        
        if (normalized < 0.5) {
            double factor = normalized * 2.0; 
            int red = 255;
            int green = static_cast<int>(255 * factor);
            int blue = static_cast<int>(255 * factor);
            return QColor(red, green, blue);
        } else {
            double factor = (normalized - 0.5) * 2.0;
            int red = static_cast<int>(255 * (1.0 - factor));
            int green = 255;
            int blue = static_cast<int>(255 * (1.0 - factor));
            return QColor(red, green, blue);
        }
    }

    int getMarkerSize(double time_ratio) {
        double speed_factor = 1.0 - time_ratio;
        int minSize = VisualizationConfig::getMinMarkerSize();
        int maxSize = VisualizationConfig::getMaxMarkerSize();
        return static_cast<int>(minSize + (maxSize - minSize) * speed_factor);
    }
}

void HistogramPlotStrategy::createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                                     const std::vector<LabeledEvent>& labeledEvents) {
    int count_pos = 0, count_neg = 0, count_zero = 0;
    for (const auto& e : labeledEvents) {
        if (e.label == 1) count_pos++;
        else if (e.label == -1) count_neg++;
        else count_zero++;
    }
    
    QBarSet *set = new QBarSet("Event Count");
    *set << count_neg << count_zero << count_pos;
    
    QBarSeries *series = new QBarSeries();
    series->append(set);
    chart->addSeries(series);
    
    QStringList categories;
    categories << "-1 (Stop)" << "0 (Vertical)" << "+1 (Profit)";
    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    axisX->append(categories);
    axisX->setTitleText("Label");
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    
    QValueAxis *axisY = new QValueAxis();
    axisY->setTitleText("Count");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    
    chart->setTitle("Histogram of Event Label Distribution");
}

void TTBMDistributionPlotStrategy::createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                                            const std::vector<LabeledEvent>& labeledEvents) {
    const int numBins = VisualizationConfig::getTTBMBinCount();
    QVector<double> binCounts(numBins, 0);
    QStringList binLabels;
    
    for (int i = 0; i < numBins; ++i) {
        double binCenter = -1.0 + (2.0 * i + 1.0) / numBins;
        binLabels << QString::number(binCenter, 'f', 2);
    }
    
    for (const auto& e : labeledEvents) {
        double ttbm = e.ttbm_label;
        int binIndex = static_cast<int>((ttbm + 1.0) / 2.0 * numBins);
        binIndex = std::max(0, std::min(numBins - 1, binIndex));
        binCounts[binIndex]++;
    }
    
    QBarSet *set = new QBarSet("TTBM Label Count");
    for (int i = 0; i < numBins; ++i) {
        *set << binCounts[i];
    }
    
    QBarSeries *series = new QBarSeries();
    series->append(set);
    chart->addSeries(series);
    
    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    axisX->append(binLabels);
    axisX->setTitleText("TTBM Label");
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    
    QValueAxis *axisY = new QValueAxis();
    axisY->setTitleText("Count");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    
    chart->setTitle("Distribution of TTBM Labels (Time-Decay Adjusted)");
}

void TimeSeriesPlotStrategy::createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                                      const std::vector<LabeledEvent>& labeledEvents) {
    chart->setTitle("Price Series with Triple Barrier Labels");
    
    QLineSeries *priceSeries = new QLineSeries();
    priceSeries->setName("Price");
    QVector<QDateTime> xDates;
    
    for (size_t i = 0; i < rows.size(); ++i) {
        QString ts = QString::fromStdString(rows[i].timestamp);
        QDateTime dt = DateParsingUtils::parseTimestamp(ts);
        if (dt.isValid()) {
            xDates.append(dt);
            priceSeries->append(dt.toMSecsSinceEpoch(), rows[i].price);
        }
    }
    chart->addSeries(priceSeries);
    
    // Create scatter series for different labels
    QScatterSeries *profitSeries = new QScatterSeries();
    profitSeries->setName("Profit Hit (+1)");
    profitSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    profitSeries->setColor(VisualizationConfig::getProfitColor());
    profitSeries->setPen(QPen(Qt::black, 1));
    
    QScatterSeries *stopSeries = new QScatterSeries();
    stopSeries->setName("Stop Hit (-1)");
    stopSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    stopSeries->setColor(VisualizationConfig::getStopColor());
    stopSeries->setPen(QPen(Qt::black, 1));
    
    QScatterSeries *vertSeries = new QScatterSeries();
    vertSeries->setName("Vertical Barrier (0)");
    vertSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    vertSeries->setColor(VisualizationConfig::getVerticalBarrierColor());
    vertSeries->setPen(QPen(Qt::black, 1));
    
    for (const auto& e : labeledEvents) {
        auto it = std::find_if(rows.begin(), rows.end(), 
            [&](const PreprocessedRow& r) { return r.timestamp == e.exit_time; });
        if (it == rows.end()) continue;
        
        int idx = int(std::distance(rows.begin(), it));
        if (idx >= xDates.size()) continue;
        
        QDateTime dt = xDates[idx];
        if (!dt.isValid()) continue;
        
        if (e.label == +1) profitSeries->append(dt.toMSecsSinceEpoch(), e.exit_price);
        else if (e.label == -1) stopSeries->append(dt.toMSecsSinceEpoch(), e.exit_price);
        else vertSeries->append(dt.toMSecsSinceEpoch(), e.exit_price);
    }
    
    chart->addSeries(profitSeries);
    chart->addSeries(stopSeries);
    chart->addSeries(vertSeries);
    
    // Setup axes
    auto *axisX = new QDateTimeAxis;
    axisX->setFormat(VisualizationConfig::getDateTimeFormat());
    axisX->setTitleText("Timestamp");
    chart->addAxis(axisX, Qt::AlignBottom);
    
    QValueAxis *axisY = new QValueAxis;
    axisY->setTitleText("Price");
    chart->addAxis(axisY, Qt::AlignLeft);
    
    for (auto series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    }
}

void TTBMTimeSeriesPlotStrategy::createPlot(QChart* chart, const std::vector<PreprocessedRow>& rows, 
                                          const std::vector<LabeledEvent>& labeledEvents) {
    chart->setTitle("Price Series with TTBM Labels");
    
    QLineSeries *priceSeries = new QLineSeries();
    priceSeries->setName("Price");
    QVector<QDateTime> xDates;
    
    for (size_t i = 0; i < rows.size(); ++i) {
        QString ts = QString::fromStdString(rows[i].timestamp);
        QDateTime dt = DateParsingUtils::parseTimestamp(ts);
        if (dt.isValid()) {
            xDates.append(dt);
            priceSeries->append(dt.toMSecsSinceEpoch(), rows[i].price);
        }
    }
    chart->addSeries(priceSeries);
    
    const int numColorSeries = VisualizationConfig::getTTBMColorSeriesCount();
    QVector<QScatterSeries*> colorSeries(numColorSeries);
    
    for (int i = 0; i < numColorSeries; ++i) {
        colorSeries[i] = new QScatterSeries();
        double labelRange = -1.0 + (2.0 * i) / (numColorSeries - 1);
        colorSeries[i]->setName(QString("TTBM %1").arg(labelRange, 0, 'f', 2));
        colorSeries[i]->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        colorSeries[i]->setColor(ttbmLabelToColor(labelRange));
        colorSeries[i]->setMarkerSize(10);
        colorSeries[i]->setPen(QPen(Qt::black, 1));
    }
    
    for (const auto& e : labeledEvents) {
        auto it = std::find_if(rows.begin(), rows.end(), 
            [&](const PreprocessedRow& r) { return r.timestamp == e.exit_time; });
        if (it == rows.end()) continue;
        
        int idx = int(std::distance(rows.begin(), it));
        if (idx >= xDates.size()) continue;
        
        QDateTime dt = xDates[idx];
        if (!dt.isValid()) continue;
        
        int seriesIndex = static_cast<int>((e.ttbm_label + 1.0) / 2.0 * (numColorSeries - 1));
        seriesIndex = std::max(0, std::min(numColorSeries - 1, seriesIndex));
        
        int markerSize = getMarkerSize(e.time_elapsed_ratio);
        colorSeries[seriesIndex]->setMarkerSize(markerSize);
        
        colorSeries[seriesIndex]->append(dt.toMSecsSinceEpoch(), e.exit_price);
    }
    
    for (int i = 0; i < numColorSeries; ++i) {
        if (colorSeries[i]->count() > 0) {
            chart->addSeries(colorSeries[i]);
        }
    }
    
    // Setup axes
    auto *axisX = new QDateTimeAxis;
    axisX->setFormat(VisualizationConfig::getDateTimeFormat());
    axisX->setTitleText("Timestamp");
    chart->addAxis(axisX, Qt::AlignBottom);
    
    QValueAxis *axisY = new QValueAxis;
    axisY->setTitleText("Price");
    chart->addAxis(axisY, Qt::AlignLeft);
    
    for (auto series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    }
}
