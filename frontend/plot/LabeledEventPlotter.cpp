#include "LabeledEventPlotter.h"
#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QDateTimeAxis>
#include <QtCharts/QValueAxis>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QLegendMarker>
#include <QDateTime>
#include <QMessageBox>
#include <QDebug>
#include <cmath>

namespace {
QColor hardLabelToColor(int label) {
    switch (label) {
        case 1: return Qt::green;   // Profit
        case -1: return Qt::red;    // Stop
        default: return Qt::white;  // Vertical/Neutral
    }
}

// Convert TTBM continuous label to color with simple gradient (no time scaling)
QColor ttbmLabelToColor(double ttbm_label) {
    // Normalize ttbm_label from [-1, +1] to [0, 1]
    double normalized = (ttbm_label + 1.0) / 2.0;
    normalized = std::max(0.0, std::min(1.0, normalized));
    
    // Create simple gradient from red (-1) through white (0) to green (+1)
    if (normalized < 0.5) {
        // Red to White gradient for negative values
        double factor = normalized * 2.0;  // Scale to [0, 1]
        int red = 255;
        int green = static_cast<int>(255 * factor);
        int blue = static_cast<int>(255 * factor);
        return QColor(red, green, blue);
    } else {
        // White to Green gradient for positive values
        double factor = (normalized - 0.5) * 2.0;  // Scale to [0, 1]
        int red = static_cast<int>(255 * (1.0 - factor));
        int green = 255;
        int blue = static_cast<int>(255 * (1.0 - factor));
        return QColor(red, green, blue);
    }
}

// Get marker size based on time-to-barrier (faster hits = larger markers)
int getMarkerSize(double time_ratio) {
    // Scale marker size from 6 to 15 based on speed (inverse of time_ratio)
    double speed_factor = 1.0 - time_ratio; // 1.0 = instant, 0.0 = max time
    return static_cast<int>(6 + 9 * speed_factor);
}
}

namespace LabeledEventPlotter {

void plot(QChartView* chartView, const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents, PlotMode mode) {
    QChart *chart = new QChart();
    
    if (mode == PlotMode::Histogram) {
        // Histogram of hard labels
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
        chartView->setChart(chart);
        return;
    }
    
    if (mode == PlotMode::TTBM_Distribution) {
        // Histogram of TTBM labels with enhanced binning
        const int numBins = 30; // More bins for better resolution
        QVector<double> binCounts(numBins, 0);
        QStringList binLabels;
        
        // Create bins from -1 to +1 with better labeling
        for (int i = 0; i < numBins; ++i) {
            double binCenter = -1.0 + (2.0 * i + 1.0) / numBins;
            binLabels << QString::number(binCenter, 'f', 2);
        }
        
        // Count TTBM labels in bins
        for (const auto& e : labeledEvents) {
            double ttbm = e.ttbm_label;
            int binIndex = static_cast<int>((ttbm + 1.0) / 2.0 * numBins);
            binIndex = std::max(0, std::min(numBins - 1, binIndex));
            binCounts[binIndex]++;
        }
        
        QBarSet *set = new QBarSet("TTBM Label Count");
        // Color bars based on their position (label value)
        for (int i = 0; i < numBins; ++i) {
            *set << binCounts[i];
        }
        
        QBarSeries *series = new QBarSeries();
        series->append(set);
        chart->addSeries(series);
        
        // Create gradient coloring for the bars (this is limited in Qt, but we can try)
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
        chartView->setChart(chart);
        return;
    }
    // Time series plotting (regular or TTBM)
    QString title = (mode == PlotMode::TTBM_TimeSeries) ? 
                   "Price Series with TTBM Labels" : 
                   "Price Series with Triple Barrier Labels";
    chart->setTitle(title);
    
    QLineSeries *priceSeries = new QLineSeries();
    priceSeries->setName("Price");
    QVector<QDateTime> xDates;
    bool anyValid = false;
    for (size_t i = 0; i < rows.size(); ++i) {
        QString ts = QString::fromStdString(rows[i].timestamp);
        QDateTime dt = QDateTime::fromString(ts, Qt::ISODate);
        if (!dt.isValid()) dt = QDateTime::fromString(ts, "yyyy-MM-dd HH:mm:ss");
        if (!dt.isValid()) dt = QDateTime::fromString(ts, "yyyy/MM/dd HH:mm:ss");
        if (!dt.isValid()) dt = QDateTime::fromString(ts, "dd/MM/yyyy HH:mm:ss");
        if (!dt.isValid()) dt = QDateTime::fromString(ts, "MM/dd/yyyy HH:mm:ss");
        if (!dt.isValid()) dt = QDateTime::fromString(ts, "M/d/yyyy H:mm:ss");
        if (dt.isValid()) {
            xDates.append(dt);
            priceSeries->append(dt.toMSecsSinceEpoch(), rows[i].price);
            anyValid = true;
        }
    }
    chart->addSeries(priceSeries);

    if (mode == PlotMode::TTBM_TimeSeries) {
        // TTBM visualization with continuous coloring and size scaling
        // Create more color series for better gradient effect
        const int numColorSeries = 15; // More series for smoother gradient
        QVector<QScatterSeries*> colorSeries(numColorSeries);
        
        for (int i = 0; i < numColorSeries; ++i) {
            colorSeries[i] = new QScatterSeries();
            double labelRange = -1.0 + (2.0 * i) / (numColorSeries - 1);
            colorSeries[i]->setName(QString("TTBM %1").arg(labelRange, 0, 'f', 2));
            colorSeries[i]->setMarkerShape(QScatterSeries::MarkerShapeCircle);
            colorSeries[i]->setColor(ttbmLabelToColor(labelRange)); // Simplified color function
            colorSeries[i]->setMarkerSize(10); // Default size
            colorSeries[i]->setPen(QPen(Qt::black, 1)); // Thin black border
        }
        
        for (const auto& e : labeledEvents) {
            auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.exit_time; });
            if (it == rows.end()) continue;
            int idx = int(std::distance(rows.begin(), it));
            if (idx >= xDates.size()) continue;
            QDateTime dt = xDates[idx];
            if (!dt.isValid()) continue;
            
            // Determine which color series this event belongs to
            int seriesIndex = static_cast<int>((e.ttbm_label + 1.0) / 2.0 * (numColorSeries - 1));
            seriesIndex = std::max(0, std::min(numColorSeries - 1, seriesIndex));
            
            // Set marker size based on how quickly the barrier was hit (keep size scaling)
            int markerSize = getMarkerSize(e.time_to_barrier_ratio);
            colorSeries[seriesIndex]->setMarkerSize(markerSize);
            
            colorSeries[seriesIndex]->append(dt.toMSecsSinceEpoch(), e.exit_price);
        }
        
        // Add only non-empty series to the chart
        for (int i = 0; i < numColorSeries; ++i) {
            if (colorSeries[i]->count() > 0) {
                chart->addSeries(colorSeries[i]);
            }
        }
        
    } else {
        // Traditional hard label visualization
        QScatterSeries *profitSeries = new QScatterSeries();
        profitSeries->setName("Profit Hit (+1)");
        profitSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        profitSeries->setColor(Qt::green);
        profitSeries->setPen(QPen(Qt::black, 1)); // Thin black border
        QScatterSeries *stopSeries = new QScatterSeries();
        stopSeries->setName("Stop Hit (-1)");
        stopSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        stopSeries->setColor(Qt::red);
        stopSeries->setPen(QPen(Qt::black, 1)); // Thin black border
        QScatterSeries *vertSeries = new QScatterSeries();
        vertSeries->setName("Vertical Barrier (0)");
        vertSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        vertSeries->setColor(Qt::blue);
        vertSeries->setPen(QPen(Qt::black, 1)); // Thin black border
        
        for (const auto& e : labeledEvents) {
            auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.exit_time; });
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
    }

    auto *axisX = new QDateTimeAxis;
    axisX->setFormat("yyyy-MM-dd HH:mm");
    axisX->setTitleText("Timestamp");
    chart->addAxis(axisX, Qt::AlignBottom);
    priceSeries->attachAxis(axisX);
    
    // Attach axes for all series
    for (auto series : chart->series()) {
        if (series != priceSeries)
            series->attachAxis(axisX);
    }
    
    QValueAxis *axisY = new QValueAxis;
    axisY->setTitleText("Price");
    chart->addAxis(axisY, Qt::AlignLeft);
    priceSeries->attachAxis(axisY);
    for (auto series : chart->series()) {
        if (series != priceSeries)
            series->attachAxis(axisY);
    }
    chartView->setChart(chart);
    if (!anyValid) {
        QMessageBox::warning(chartView, "Chart Error", "No valid timestamps found in data. Check your CSV timestamp format.");
    }
}

} // namespace LabeledEventPlotter
