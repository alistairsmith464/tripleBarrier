#include "MainWindow.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <QSizePolicy>
#include <QFont>
#include <QApplication>
#include "../backend/data/DataPreprocessor.h"
#include "BarrierConfigDialog.h"
#include <QInputDialog>
#include "TripleBarrierLabeler.h"
#include "LabeledEvent.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setupUI();
    setWindowTitle("Triple Barrier - File Upload");
    setMinimumSize(600, 400);
    resize(800, 500);
}

MainWindow::~MainWindow() {}

void MainWindow::setupUI()
{
    // Create central widget
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // Create main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(24);
    mainLayout->setContentsMargins(40, 36, 40, 36);
    centralWidget->setStyleSheet("background: #f4f8fb;");

    // Title
    m_titleLabel = new QLabel("Triple Barrier Data Uploader", this);
    QFont titleFont = m_titleLabel->font();
    titleFont.setPointSize(22);
    titleFont.setBold(true);
    m_titleLabel->setFont(titleFont);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    m_titleLabel->setStyleSheet("color: #22313f; margin-bottom: 8px; letter-spacing: 1px;");

    // Status label
    m_statusLabel = new QLabel("Click 'Upload Data' to begin", this);
    m_statusLabel->setAlignment(Qt::AlignCenter);
    m_statusLabel->setStyleSheet("color: #e74c3c; font-size: 15px; font-weight: bold; margin-bottom: 8px;");

    // Progress bar (initially hidden)
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    m_progressBar->setStyleSheet(
        "QProgressBar { border: 1px solid #b2bec3; border-radius: 5px; background: #eaf0f6; height: 18px; }"
        "QProgressBar::chunk { background: #3498db; border-radius: 5px; }"
    );

    // Buttons layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(24);

    m_uploadDataButton = new QPushButton("Upload Data", this);
    m_uploadDataButton->setMinimumHeight(44);
    m_uploadDataButton->setMinimumWidth(220);
    m_uploadDataButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #3498db;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 17px;"
        "    font-weight: bold;"
        "    letter-spacing: 1px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #21618c;"
        "}"
    );

    m_uploadMenu = new QMenu(this);
    m_csvAction = new QAction("Upload CSV", this);
    m_uploadMenu->addAction(m_csvAction);
    m_uploadDataButton->setMenu(m_uploadMenu);

    m_clearButton = new QPushButton("Clear", this);
    m_clearButton->setMinimumHeight(44);
    m_clearButton->setMinimumWidth(220);
    m_clearButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #b2bec3;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 17px;"
        "    font-weight: bold;"
        "    letter-spacing: 1px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #636e72;"
        "}"
    );

    buttonLayout->addWidget(m_uploadDataButton);
    buttonLayout->addWidget(m_clearButton);

    // File info display
    m_fileInfoDisplay = new QTextEdit(this);
    m_fileInfoDisplay->setReadOnly(true);
    m_fileInfoDisplay->setPlaceholderText("Data summary will appear here after upload...");
    m_fileInfoDisplay->setStyleSheet(
        "QTextEdit {"
        "    border: 2px solid #bdc3c7;"
        "    border-radius: 8px;"
        "    padding: 16px;"
        "    background-color: #f8f9fa;"
        "    font-family: 'Fira Mono', 'Consolas', 'Courier New', monospace;"
        "    font-size: 15px;"
        "    color: #22313f;"
        "}"
    );

    // Add all components to main layout
    mainLayout->addWidget(m_titleLabel);
    mainLayout->addWidget(m_statusLabel);
    mainLayout->addWidget(m_progressBar);
    mainLayout->addLayout(buttonLayout);
    mainLayout->addWidget(m_fileInfoDisplay, 1);

    // Chart for graphical display
    m_chartView = new QtCharts::QChartView(this);
    m_chartView->setMinimumHeight(300);
    mainLayout->addWidget(m_chartView);

    // Connect signals
    connect(m_csvAction, &QAction::triggered, this, &MainWindow::onSelectCSVFile);
    connect(m_clearButton, &QPushButton::clicked, this, &MainWindow::onClearButtonClicked);
}

void MainWindow::onUploadDataButtonClicked() {
    // Not used, menu is attached directly to button
}

void MainWindow::onSelectCSVFile() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select CSV File",
        "",
        "CSV Files (*.csv);;All Files (*.*)"
    );
    if (fileName.isEmpty()) {
        return;
    }
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);
    m_statusLabel->setText("Loading CSV data...");
    m_uploadDataButton->setEnabled(false);
    QApplication::processEvents();
    try {
        CSVDataSource src;
        std::vector<DataRow> rows = src.loadData(fileName.toStdString());
        showUploadSuccess(fileName, rows);
        // Prompt for barrier config
        BarrierConfigDialog dialog(this);
        if (dialog.exec() == QDialog::Accepted) {
            BarrierConfig cfg = dialog.getConfig();
            try {
                cfg.validate();
                DataPreprocessor::Params params;
                params.volatility_window = 20;
                params.event_interval = 10;
                params.barrier_multiple = cfg.profit_multiple;
                params.vertical_barrier = cfg.vertical_window;
                auto processed = DataPreprocessor::preprocess(rows, params);
                // Find event indices
                std::vector<size_t> event_indices;
                for (size_t i = 0; i < processed.size(); ++i) {
                    if (processed[i].is_event) event_indices.push_back(i);
                }
                auto labeled = TripleBarrierLabeler::label(
                    processed,
                    event_indices,
                    cfg.profit_multiple,
                    cfg.stop_multiple,
                    cfg.vertical_window
                );
                // Show plot
                plotLabeledEvents(processed, labeled);
            } catch (const std::exception& ex) {
                showUploadError(QString("Barrier config error: %1").arg(ex.what()));
            }
        }
    } catch (const std::exception& ex) {
        showUploadError(QString("Failed to load CSV: %1").arg(ex.what()));
    }
    m_progressBar->setVisible(false);
    m_uploadDataButton->setEnabled(true);
}

void MainWindow::onClearButtonClicked()
{
    m_fileInfoDisplay->clear();
    m_statusLabel->setText("Select a file to upload");
    m_statusLabel->setStyleSheet("color: #7f8c8d; font-size: 12px;");
}

void MainWindow::showUploadSuccess(const QString& filePath, const std::vector<DataRow>& rows) {
    m_statusLabel->setText("✓ Data loaded successfully!");
    m_statusLabel->setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;");
    QMessageBox::information(this, "Upload Successful", QString("Loaded %1 rows from %2").arg(rows.size()).arg(filePath));
}

void MainWindow::showUploadError(const QString& error) {
    m_statusLabel->setText("✗ Data load failed");
    m_statusLabel->setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;");
    QMessageBox::critical(this, "Load Failed", error);
}

void MainWindow::plotLabeledEvents(const std::vector<ProcessedRow>& processed, const std::vector<LabeledEvent>& labeled) {
    // Clear previous chart data
    m_chartView->chart()->removeAllSeries();
    m_chartView->chart()->axes(Qt::Horizontal).first()->setVisible(false);
    m_chartView->chart()->axes(Qt::Vertical).first()->setVisible(false);

    if (labeled.empty()) {
        return;
    }

    // Prepare data for plotting
    QVector<QPointF> points;
    for (const auto& row : processed) {
        points.append(QPointF(row.timestamp.toMSecsSinceEpoch(), row.close));
    }

    // Create line series for price data
    QtCharts::QLineSeries* priceSeries = new QtCharts::QLineSeries();
    priceSeries->setName("Price");
    priceSeries->setPen(QPen(QColor(52, 152, 219), 2));
    priceSeries->append(points);

    // Add series to chart
    m_chartView->chart()->addSeries(priceSeries);

    // Create scatter series for labeled events
    QtCharts::QScatterSeries* eventSeries = new QtCharts::QScatterSeries();
    eventSeries->setName("Labeled Events");
    eventSeries->setMarkerSize(10);
    eventSeries->setColor(QColor(231, 76, 60));

    // Add points to scatter series
    for (const auto& event : labeled) {
        qint64 x = QDateTime::fromString(QString::fromStdString(event.entry_time), "yyyy-MM-dd hh:mm:ss").toMSecsSinceEpoch();
        qint64 y = event.entry_price;
        eventSeries->append(x, y);
    }

    // Add scatter series to chart
    m_chartView->chart()->addSeries(eventSeries);

    // Configure axes
    auto* xAxis = new QtCharts::QValueAxis;
    xAxis->setLabelFormat("%.0f");
    xAxis->setTitleText("Time");
    m_chartView->chart()->addAxis(xAxis, Qt::AlignBottom);
    priceSeries->attachAxis(xAxis);
    eventSeries->attachAxis(xAxis);

    auto* yAxis = new QtCharts::QValueAxis;
    yAxis->setLabelFormat("%.2f");
    yAxis->setTitleText("Price");
    m_chartView->chart()->addAxis(yAxis, Qt::AlignLeft);
    priceSeries->attachAxis(yAxis);
    eventSeries->attachAxis(yAxis);

    // Set chart title
    m_chartView->chart()->setTitle("Labeled Events on Price Chart");
    m_chartView->chart()->setAnimationOptions(QtCharts::QChart::SeriesAnimations);
    m_chartView->setRenderHint(QPainter::Antialiasing);
}

void MainWindow::plotLabeledEvents(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents) {
    using namespace QtCharts;
    QChart *chart = new QChart();
    chart->setTitle("Price Series with Triple Barrier Labels");
    // Price line
    QLineSeries *priceSeries = new QLineSeries();
    priceSeries->setName("Price");
    for (size_t i = 0; i < rows.size(); ++i) {
        priceSeries->append(i, rows[i].price);
    }
    chart->addSeries(priceSeries);
    // Markers for labeled events
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
        // Find entry index
        auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
        if (it == rows.end()) continue;
        int idx = int(std::distance(rows.begin(), it));
        if (e.label == +1) profitSeries->append(idx, e.entry_price);
        else if (e.label == -1) stopSeries->append(idx, e.entry_price);
        else vertSeries->append(idx, e.entry_price);
    }
    chart->addSeries(profitSeries);
    chart->addSeries(stopSeries);
    chart->addSeries(vertSeries);
    // Axes
    chart->createDefaultAxes();
    m_chartView->setChart(chart);
}
