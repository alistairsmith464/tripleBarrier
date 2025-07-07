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
QColor softLabelToColor(double soft_label) {
    double v = std::max(-1.0, std::min(1.0, soft_label));
    if (v < 0) {
        int r = 255;
        int g = int(255 * (1 + v));
        int b = int(255 * (1 + v));
        return QColor(r, g, b);
    } else {
        int r = int(255 * (1 - v));
        int g = 255;
        int b = int(255 * (1 - v));
        return QColor(r, g, b);
    }
}
}

namespace LabeledEventPlotter {

void plot(QChartView* chartView, const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents, PlotMode mode) {
    QChart *chart = new QChart();
    if (mode == PlotMode::Histogram) {
        // Detect if soft labels are in use
        bool useSoft = false;
        for (const auto& e : labeledEvents) {
            if (std::abs(e.soft_label) > 1e-6 || e.soft_label != 0) {
                useSoft = true;
                break;
            }
        }
        if (useSoft) {
            // Histogram of soft_label values
            const int numBins = 10;
            std::vector<int> bins(numBins, 0);
            for (const auto& e : labeledEvents) {
                double v = std::max(-1.0, std::min(1.0, e.soft_label));
                int bin = int((v + 1.0) / 2.0 * numBins);
                if (bin < 0) bin = 0;
                if (bin >= numBins) bin = numBins - 1;
                bins[bin]++;
            }
            QBarSet *set = new QBarSet("Soft Label Count");
            for (int c : bins) *set << c;
            QBarSeries *series = new QBarSeries();
            series->append(set);
            chart->addSeries(series);
            QStringList categories;
            for (int i = 0; i < numBins; ++i) {
                double left = -1.0 + (2.0 * i) / numBins;
                double right = -1.0 + (2.0 * (i + 1)) / numBins;
                categories << QString("[%1, %2)").arg(left, 0, 'f', 2).arg(right, 0, 'f', 2);
            }
            QBarCategoryAxis *axisX = new QBarCategoryAxis();
            axisX->append(categories);
            axisX->setTitleText("Soft Label");
            chart->addAxis(axisX, Qt::AlignBottom);
            series->attachAxis(axisX);
            QValueAxis *axisY = new QValueAxis();
            axisY->setTitleText("Count");
            chart->addAxis(axisY, Qt::AlignLeft);
            series->attachAxis(axisY);
            chart->setTitle("Histogram of Soft Label Distribution");
        } else {
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
        }
        chartView->setChart(chart);
        return;
    }
    chart->setTitle("Price Series with Triple Barrier Labels");
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

    // Detect if soft labels are in use
    bool useSoft = false;
    for (const auto& e : labeledEvents) {
        if (std::abs(e.soft_label) > 1e-6 || e.soft_label != 0) {
            useSoft = true;
            break;
        }
    }

    if (useSoft) {
        // Add a dummy series for the legend
        QScatterSeries *legendSeries = new QScatterSeries();
        legendSeries->setName("Soft Label (gradient)");
        legendSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        legendSeries->setColor(Qt::gray); // Neutral color for legend
        legendSeries->setMarkerSize(12.0);
        legendSeries->append(0, 0); // Dummy point, will not be visible
        chart->addSeries(legendSeries);
        legendSeries->setVisible(false); // Hide dummy point from plot, but keep in legend
        chart->legend()->markers(legendSeries).first()->setVisible(true);
        // Add each event as a single-point series, hide from legend
        for (const auto& e : labeledEvents) {
            auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
            if (it == rows.end()) continue;
            int idx = int(std::distance(rows.begin(), it));
            if (idx >= xDates.size()) continue;
            QDateTime dt = xDates[idx];
            if (!dt.isValid()) continue;
            auto pt = QPointF(dt.toMSecsSinceEpoch(), e.entry_price);
            QScatterSeries *pointSeries = new QScatterSeries();
            pointSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
            pointSeries->setMarkerSize(12.0);
            pointSeries->setColor(softLabelToColor(e.soft_label));
            pointSeries->append(pt);
            pointSeries->setName(""); // Hide from legend
            chart->addSeries(pointSeries);
            // Hide legend marker for this series
            auto markers = chart->legend()->markers(pointSeries);
            if (!markers.isEmpty()) markers.first()->setVisible(false);
        }
    } else {
        QScatterSeries *profitSeries = new QScatterSeries();
        profitSeries->setName("Profit Hit (+1)");
        profitSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        profitSeries->setColor(Qt::green);
        QScatterSeries *stopSeries = new QScatterSeries();
        stopSeries->setName("Stop Hit (-1)");
        stopSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        stopSeries->setColor(Qt::red);
        QScatterSeries *vertSeries = new QScatterSeries();
        vertSeries->setName("Vertical Barrier (0)");
        vertSeries->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        vertSeries->setColor(Qt::blue);
        for (const auto& e : labeledEvents) {
            auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
            if (it == rows.end()) continue;
            int idx = int(std::distance(rows.begin(), it));
            if (idx >= xDates.size()) continue;
            QDateTime dt = xDates[idx];
            if (!dt.isValid()) continue;
            if (e.label == +1) profitSeries->append(dt.toMSecsSinceEpoch(), e.entry_price);
            else if (e.label == -1) stopSeries->append(dt.toMSecsSinceEpoch(), e.entry_price);
            else vertSeries->append(dt.toMSecsSinceEpoch(), e.entry_price);
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
    if (!useSoft) {
        // Attach axes for hard label series
        for (auto series : chart->series()) {
            if (series != priceSeries)
                series->attachAxis(axisX);
        }
    } else {
        // Attach axes for all series (each point is a series)
        for (auto series : chart->series()) {
            if (series != priceSeries)
                series->attachAxis(axisX);
        }
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
