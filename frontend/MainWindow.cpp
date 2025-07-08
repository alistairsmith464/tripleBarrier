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
#include "utils/CSVExportUtils.h"
#include "plot/LabeledEventPlotter.h"
#include "feature/FeaturePreviewDialog.h"
#include "utils/DialogUtils.h"
#include "utils/FileDialogUtils.h"
#include "utils/UserInputUtils.h"
#include "utils/LabelingUtils.h"

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
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    m_ui.setup(centralWidget);
    m_titleLabel = m_ui.titleLabel;
    m_statusLabel = m_ui.statusLabel;
    m_progressBar = m_ui.progressBar;
    m_uploadDataButton = m_ui.uploadDataButton;
    m_clearButton = m_ui.clearButton;
    m_uploadMenu = m_ui.uploadMenu;
    m_csvAction = m_ui.csvAction;
    m_exportCSVAction = m_ui.exportCSVAction;
    m_chartView = m_ui.chartView;
    connect(m_csvAction, &QAction::triggered, this, &MainWindow::onSelectCSVFile);
    connect(m_clearButton, &QPushButton::clicked, this, &MainWindow::onClearButtonClicked);
    connect(m_ui.mlButton, &QPushButton::clicked, this, &MainWindow::onMLButtonClicked);

    // Add plot mode toggle
    QHBoxLayout *plotModeLayout = new QHBoxLayout();
    QLabel *plotModeLabel = new QLabel("Plot Mode:", this);
    m_plotModeComboBox = new QComboBox(this);
    m_plotModeComboBox->addItem("Time Series");
    m_plotModeComboBox->addItem("Histogram");
    plotModeLayout->addWidget(plotModeLabel);
    plotModeLayout->addWidget(m_plotModeComboBox);
    plotModeLayout->addStretch();
    m_ui.mainLayout->insertLayout(1, plotModeLayout); // Insert near top
    connect(m_plotModeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        m_plotMode = (idx == 0) ? PlotMode::TimeSeries : PlotMode::Histogram;
        if (!g_lastRows.empty() && !g_lastLabeledEvents.empty())
            plotLabeledEvents(g_lastRows, g_lastLabeledEvents);
    });
}

void MainWindow::onUploadDataButtonClicked() {
    // Not used, menu is attached directly to button
}

void MainWindow::onSelectCSVFile() {
    QString fileName = FileDialogUtils::getOpenCSVFile(this);
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
        BarrierConfig cfg;
        DataPreprocessor::Params params;
        if (UserInputUtils::getLabelingConfig(this, cfg, params)) {
            try {
                cfg.validate();
                auto processed = DataPreprocessor::preprocess(rows, params);
                std::vector<size_t> event_indices;
                for (size_t i = 0; i < processed.size(); ++i) {
                    if (processed[i].is_event) event_indices.push_back(i);
                }
                std::vector<LabeledEvent> labeled = LabelingUtils::labelEvents(processed, event_indices, cfg);
                plotLabeledEvents(processed, labeled);
            } catch (const std::exception& ex) {
                showUploadError(QString("Barrier config error: %1").arg(ex.what()));
            }
        } else {
            m_statusLabel->setText("Labeling cancelled.");
            m_progressBar->setVisible(false);
            m_uploadDataButton->setEnabled(true);
            return;
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
    DialogUtils::showInfo(this, "Upload Successful", QString("Loaded from %1").arg(filePath));
}

void MainWindow::showUploadError(const QString& error) {
    m_statusLabel->setText("✗ Data load failed");
    m_statusLabel->setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;");
    DialogUtils::showError(this, "Load Failed", error);
}

void MainWindow::plotLabeledEvents(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents) {
    LabeledEventPlotter::plot(m_chartView, rows, labeledEvents, m_plotMode);
    g_lastRows = rows;
    g_lastLabeledEvents = labeledEvents;
}

void MainWindow::onMLButtonClicked() {
    FeatureSelectionDialog dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        QSet<QString> selected = dlg.selectedFeatures();
        if (g_lastRows.empty() || g_lastLabeledEvents.empty()) {
            DialogUtils::showWarning(this, "Feature Error", "No labeled events available. Please upload and label data first.");
            return;
        }
        FeaturePreviewDialog previewDlg(selected, g_lastRows, g_lastLabeledEvents, this);
        previewDlg.exec();
    }
}

void MainWindow::onExportCSVClicked() {
    // Export to CSV is now handled in FeaturePreviewDialog only
}
