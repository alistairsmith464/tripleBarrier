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
#include "../backend/data/LabeledEvent.h"
#include <QtCharts/QValueAxis>
#include "FeatureSelectionDialog.h"
#include "../backend/data/FeatureCalculator.h"
#include <QTableWidget>
#include <QHeaderView>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"
#include <vector>
#include "../backend/data/HardBarrierLabeler.h"
#include "../backend/data/ProbabilisticBarrierLabeler.h"

// For ML feature calculation preview
std::vector<PreprocessedRow> g_lastRows;
std::vector<LabeledEvent> g_lastLabeledEvents;

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

    // Add all components to main layout
    mainLayout->addWidget(m_titleLabel);
    mainLayout->addWidget(m_statusLabel);
    mainLayout->addWidget(m_progressBar);
    mainLayout->addLayout(buttonLayout);

    // Chart for graphical display
    m_chartView = new QChartView(this);
    m_chartView->setMinimumHeight(300);
    mainLayout->addWidget(m_chartView);

    // Machine Learning button
    QPushButton* mlButton = new QPushButton("Next: Machine Learning", this);
    mlButton->setMinimumHeight(44);
    mlButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #2ecc71;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 17px;"
        "    font-weight: bold;"
        "    letter-spacing: 1px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #27ae60;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #219653;"
        "}"
    );
    connect(mlButton, &QPushButton::clicked, this, &MainWindow::onMLButtonClicked);
    mainLayout->addWidget(mlButton);

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
        showUploadSuccess(fileName);
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
                params.use_cusum = cfg.use_cusum;
                params.cusum_threshold = cfg.cusum_threshold;
                auto processed = DataPreprocessor::preprocess(rows, params);
                // Find event indices
                std::vector<size_t> event_indices;
                for (size_t i = 0; i < processed.size(); ++i) {
                    if (processed[i].is_event) event_indices.push_back(i);
                }
                std::vector<LabeledEvent> labeled;
                if (cfg.labeling_type == BarrierConfig::Hard) {
                    HardBarrierLabeler labeler;
                    labeled = labeler.label(
                        processed,
                        event_indices,
                        cfg.profit_multiple,
                        cfg.stop_multiple,
                        cfg.vertical_window
                    );
                } else {
                    ProbabilisticBarrierLabeler labeler;
                    labeled = labeler.label(
                        processed,
                        event_indices,
                        cfg.profit_multiple,
                        cfg.stop_multiple,
                        cfg.vertical_window
                    );
                }
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
    m_statusLabel->setText("Select a file to upload");
    m_statusLabel->setStyleSheet("color: #7f8c8d; font-size: 12px;");
}

void MainWindow::showUploadSuccess(const QString& filePath) {
    m_statusLabel->setText("✓ Data loaded successfully!");
    m_statusLabel->setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;");
    QMessageBox::information(this, "Upload Successful", QString("Loaded from %1").arg(filePath));
}

void MainWindow::showUploadError(const QString& error) {
    m_statusLabel->setText("✗ Data load failed");
    m_statusLabel->setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;");
    QMessageBox::critical(this, "Load Failed", error);
}

void MainWindow::plotLabeledEvents(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents) {
    QChart *chart = new QChart();
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
        if (!dt.isValid()) dt = QDateTime::fromString(ts, "M/d/yyyy H:mm:ss"); // Add this for single-digit months/days
        if (dt.isValid()) {
            xDates.append(dt);
            priceSeries->append(dt.toMSecsSinceEpoch(), rows[i].price);
            anyValid = true;
        }
    }
    qDebug() << "Price series count:" << priceSeries->count();
    chart->addSeries(priceSeries);
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
    qDebug() << "Profit series count:" << profitSeries->count();
    qDebug() << "Stop series count:" << stopSeries->count();
    qDebug() << "Vert series count:" << vertSeries->count();
    chart->addSeries(profitSeries);
    chart->addSeries(stopSeries);
    chart->addSeries(vertSeries);
    auto *axisX = new QDateTimeAxis;
    axisX->setFormat("yyyy-MM-dd HH:mm");
    axisX->setTitleText("Timestamp");
    chart->addAxis(axisX, Qt::AlignBottom);
    priceSeries->attachAxis(axisX);
    profitSeries->attachAxis(axisX);
    stopSeries->attachAxis(axisX);
    vertSeries->attachAxis(axisX);
    QValueAxis *axisY = new QValueAxis;
    axisY->setTitleText("Price");
    chart->addAxis(axisY, Qt::AlignLeft);
    priceSeries->attachAxis(axisY);
    profitSeries->attachAxis(axisY);
    stopSeries->attachAxis(axisY);
    vertSeries->attachAxis(axisY);
    m_chartView->setChart(chart);
    // Save for ML step
    g_lastRows = rows;
    g_lastLabeledEvents = labeledEvents;
    if (!anyValid) {
        QMessageBox::warning(this, "Chart Error", "No valid timestamps found in data. Check your CSV timestamp format.");
    }
}

void MainWindow::onMLButtonClicked() {
    FeatureSelectionDialog dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        QSet<QString> selected = dlg.selectedFeatures();
        // Map UI feature names to backend feature IDs
        QMap<QString, std::string> featureMap = {
            {"Close-to-close return for the previous day", FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D},
            {"Return over the past 5 days", FeatureCalculator::RETURN_5D},
            {"Return over the past 10 days", FeatureCalculator::RETURN_10D},
            {"Rolling standard deviation of daily returns over the last 5 days", FeatureCalculator::ROLLING_STD_5D},
            {"EWMA volatility over 10 days", FeatureCalculator::EWMA_VOL_10D},
            {"5-day simple moving average (SMA)", FeatureCalculator::SMA_5D},
            {"10-day SMA", FeatureCalculator::SMA_10D},
            {"20-day SMA", FeatureCalculator::SMA_20D},
            {"Distance between current close price and 5-day SMA", FeatureCalculator::DIST_TO_SMA_5D},
            {"Rate of Change (ROC) over 5 days", FeatureCalculator::ROC_5D},
            {"Relative Strength Index (RSI) over 14 days", FeatureCalculator::RSI_14D},
            {"5-day high minus 5-day low (price range)", FeatureCalculator::PRICE_RANGE_5D},
            {"Current close price relative to 5-day high", FeatureCalculator::CLOSE_OVER_HIGH_5D},
            {"Slope of linear regression of close prices over 10 days", FeatureCalculator::SLOPE_LR_10D},
            {"Day of the week", FeatureCalculator::DAY_OF_WEEK},
            {"Days since last event", FeatureCalculator::DAYS_SINCE_LAST_EVENT}
        };
        std::set<std::string> backendFeatures;
        for (const QString& feat : selected) {
            if (featureMap.contains(feat)) backendFeatures.insert(featureMap[feat]);
        }
        // Gather data from last plot (assume last processed data is available)
        // For demo, use static variables (should be class members in real code)
        extern std::vector<PreprocessedRow> g_lastRows;
        extern std::vector<LabeledEvent> g_lastLabeledEvents;
        if (g_lastRows.empty() || g_lastLabeledEvents.empty()) {
            QMessageBox::warning(this, "Feature Error", "No labeled events available. Please upload and label data first.");
            return;
        }
        // Prepare data vectors
        std::vector<double> prices;
        std::vector<std::string> timestamps;
        std::vector<int> eventIndices;
        for (size_t i = 0; i < g_lastRows.size(); ++i) {
            prices.push_back(g_lastRows[i].price);
            timestamps.push_back(g_lastRows[i].timestamp);
        }
        for (const auto& e : g_lastLabeledEvents) {
            auto it = std::find_if(g_lastRows.begin(), g_lastRows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
            if (it != g_lastRows.end()) eventIndices.push_back(int(std::distance(g_lastRows.begin(), it)));
        }
        // Calculate features for each event
        std::vector<std::map<std::string, double>> allFeatures;
        for (size_t i = 0; i < eventIndices.size(); ++i) {
            allFeatures.push_back(FeatureCalculator::calculateFeatures(prices, timestamps, eventIndices, int(i), backendFeatures));
        }
        // Show preview table
        QDialog previewDlg(this);
        previewDlg.setWindowTitle("Feature Preview");
        QVBoxLayout* vbox = new QVBoxLayout(&previewDlg);
        QTableWidget* table = new QTableWidget(int(eventIndices.size()), int(backendFeatures.size()), &previewDlg);
        QStringList headers;
        for (const QString& feat : selected) headers << feat;
        table->setHorizontalHeaderLabels(headers);
        int col = 0;
        for (const QString& feat : selected) {
            std::string backendId = featureMap[feat];
            for (int row = 0; row < int(eventIndices.size()); ++row) {
                double val = allFeatures[row].count(backendId) ? allFeatures[row][backendId] : NAN;
                table->setItem(row, col, new QTableWidgetItem(QString::number(val)));
            }
            ++col;
        }
        table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        vbox->addWidget(table);
        QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok, &previewDlg);
        connect(box, &QDialogButtonBox::accepted, &previewDlg, &QDialog::accept);
        vbox->addWidget(box);
        previewDlg.exec();
    }
}
