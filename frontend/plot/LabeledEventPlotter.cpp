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
        default: return Qt::blue;   // Vertical/Neutral
    }
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

    // Plot hard labels only
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
