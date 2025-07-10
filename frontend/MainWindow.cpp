#include "MainWindow.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <QSizePolicy>
#include <QFont>
#include <QApplication>
#include "BarrierConfigDialog.h"
#include <QInputDialog>
#include "../backend/data/LabeledEvent.h"
#include "../backend/data/DataRow.h"
#include "../backend/data/CSVDataSource.h"
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/DataPreprocessor.h"
#include "../backend/data/BarrierConfig.h"
#include <QtCharts/QValueAxis>
#include "FeatureSelectionDialog.h"
#include <QTableWidget>
#include <QHeaderView>
#include <vector>
#include "plot/LabeledEventPlotter.h"
#include "feature/FeaturePreviewDialog.h"
#include "utils/DialogUtils.h"
#include "utils/FileDialogUtils.h"
#include "utils/UserInputUtils.h"
#include "utils/InputValidator.h"
#include "services/DataService.h"
#include "services/MLService.h"
#include "utils/ErrorHandler.h"
#include "utils/AsyncTaskManager.h"
#include "config/ApplicationConfig.h"
#include "ui/UIStrings.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      m_dataService(std::make_unique<DataServiceImpl>()),
      m_mlService(std::make_unique<MLServiceImpl>())
{
    setupErrorHandling();
    loadApplicationConfig();
    setupUI();
    
    setWindowTitle(UIStrings::APP_TITLE);
    
    // Apply saved window size
    auto& config = ApplicationConfig::instance();
    resize(config.defaultWindowSize());
    setMinimumSize(600, 400);
    
    // Setup application shutdown handling
    connect(qApp, &QApplication::aboutToQuit, this, &MainWindow::onApplicationShutdown);
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
    m_chartView = m_ui.chartView;
    connect(m_csvAction, &QAction::triggered, this, &MainWindow::onSelectCSVFile);
    connect(m_clearButton, &QPushButton::clicked, this, &MainWindow::onClearButtonClicked);
    connect(m_ui.mlButton, &QPushButton::clicked, this, &MainWindow::onMLButtonClicked);

    QHBoxLayout *plotModeLayout = new QHBoxLayout();
    QLabel *plotModeLabel = new QLabel(UIStrings::PLOT_MODE, this);
    m_plotModeComboBox = new QComboBox(this);
    m_plotModeComboBox->addItem(UIStrings::TIME_SERIES);
    m_plotModeComboBox->addItem(UIStrings::HISTOGRAM);
    m_plotModeComboBox->addItem(UIStrings::TTBM_TIME_SERIES);
    m_plotModeComboBox->addItem(UIStrings::TTBM_DISTRIBUTION);
    plotModeLayout->addWidget(plotModeLabel);
    plotModeLayout->addWidget(m_plotModeComboBox);
    plotModeLayout->addStretch();
    m_ui.mainLayout->insertLayout(1, plotModeLayout);
    connect(m_plotModeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        switch(idx) {
            case 0: m_plotMode = PlotMode::TimeSeries; break;
            case 1: m_plotMode = PlotMode::Histogram; break;
            case 2: m_plotMode = PlotMode::TTBM_TimeSeries; break;
            case 3: m_plotMode = PlotMode::TTBM_Distribution; break;
            default: m_plotMode = PlotMode::TimeSeries; break;
        }
        if (!m_lastRows.empty() && !m_lastLabeledEvents.empty())
            plotLabeledEvents(m_lastRows, m_lastLabeledEvents);
    });
}

void MainWindow::onSelectCSVFile() {
    QString fileName = FileDialogUtils::getOpenCSVFile(this);
    if (fileName.isEmpty()) {
        return;
    }
    
    // Validate file path
    ValidationResult fileValidation = InputValidator::validateFilePath(fileName, true);
    if (!fileValidation.isValid) {
        ErrorHandler::ErrorInfo errorInfo(
            ErrorHandler::ErrorType::FileSystem,
            ErrorHandler::Severity::Error,
            UIStrings::INVALID_FILE_SELECTION,
            fileValidation.errorMessage
        );
        ErrorHandler::handleError(errorInfo, this);
        return;
    }
    
    // Update last used path
    auto& config = ApplicationConfig::instance();
    config.setLastUsedDataPath(fileName);
    config.addRecentFile(fileName);
    
    // Disable UI during processing
    m_uploadDataButton->setEnabled(false);
    m_statusLabel->setText(UIStrings::LOADING_CSV);
    
    // Try a step-by-step synchronous approach to isolate the crash
    try {
        // Step 1: Load CSV
        CSVDataSource src;
        std::vector<DataRow> rows = src.loadData(fileName.toStdString());
        
        if (rows.empty()) {
            throw std::runtime_error("No data loaded from CSV file");
        }
        
        m_statusLabel->setText(QString("Loaded %1 rows successfully").arg(rows.size()));
        qApp->processEvents();
        
        showBarrierConfigurationDialog(rows);
        
    } catch (const std::exception& e) {
        ErrorHandler::ErrorInfo errorInfo(
            ErrorHandler::ErrorType::DataLoad,
            ErrorHandler::Severity::Error,
            "CSV Loading Failed",
            QString::fromStdString(e.what())
        );
        ErrorHandler::handleError(errorInfo, this);
        m_uploadDataButton->setEnabled(true);
    }
}

void MainWindow::showBarrierConfigurationDialog(const std::vector<DataRow>& rows) {
    BarrierConfigDialog dialog(this);
    if (dialog.exec() == QDialog::Accepted) {
        BarrierConfig cfg = dialog.getConfig();
        
        DataPreprocessor::Params params;
        params.volatility_window = dialog.volatilityWindow();
        params.barrier_multiple = cfg.profit_multiple;
        params.vertical_barrier = cfg.vertical_window;
        params.use_cusum = cfg.use_cusum;
        params.cusum_threshold = cfg.cusum_threshold;
        params.barrier_config = cfg;
        
        processDataWithUserConfig(rows, cfg, params);
    } else {
        m_statusLabel->setText("Configuration cancelled by user");
        m_uploadDataButton->setEnabled(true);
    }
}

void MainWindow::processDataWithConfig(const std::vector<DataRow>& rows, 
                                     const BarrierConfig& cfg, 
                                     const DataPreprocessor::Params& params) {
    try {
        // Try processing with the user-provided configuration
        auto processed = DataPreprocessor::preprocess(rows, params);
        
        if (processed.empty()) {
            throw std::runtime_error("Data preprocessing returned empty result");
        }
        
        // Try generating labeled events
        DataServiceImpl dataService;
        auto labeled = dataService.generateLabeledEvents(processed, cfg);
        
        // If we get here, display the results
        plotLabeledEvents(processed, labeled);
        showUploadSuccess(QString::fromStdString("Processing completed with %1 events").arg(labeled.size()));
        
    } catch (const std::exception& e) {
        ErrorHandler::ErrorInfo errorInfo(
            ErrorHandler::ErrorType::DataLoad,
            ErrorHandler::Severity::Error,
            "Data Processing Failed",
            QString::fromStdString(e.what())
        );
        ErrorHandler::handleError(errorInfo, this);
    }
}

void MainWindow::onClearButtonClicked()
{
    m_statusLabel->setText(UIStrings::READY);
    m_statusLabel->setStyleSheet("color: #7f8c8d; font-size: 12px;");
}

void MainWindow::showUploadSuccess(const QString& filePath) {
    m_statusLabel->setText(UIStrings::LOAD_SUCCESS);
    m_statusLabel->setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;");
    DialogUtils::showInfo(this, UIStrings::INFO_TITLE, UIStringHelper::uploadSuccessMessage(filePath));
}

void MainWindow::showUploadError(const QString& error) {
    m_statusLabel->setText(UIStrings::LOAD_FAILED);
    m_statusLabel->setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;");
    DialogUtils::showError(this, UIStrings::ERROR_TITLE, UIStringHelper::loadErrorMessage(error));
}

void MainWindow::plotLabeledEvents(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents) {
    LabeledEventPlotter::plot(m_chartView, rows, labeledEvents, m_plotMode);
    m_lastRows = rows;
    m_lastLabeledEvents = labeledEvents;
}

void MainWindow::onMLButtonClicked() {
    FeatureSelectionDialog dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        QSet<QString> selected = dlg.selectedFeatures();
        if (m_lastRows.empty() || m_lastLabeledEvents.empty()) {
            DialogUtils::showWarning(this, UIStrings::WARNING_TITLE, UIStrings::NO_DATA_ERROR);
            return;
        }
        FeaturePreviewDialog previewDlg(selected, m_lastRows, m_lastLabeledEvents, this);
        previewDlg.exec();
    }
}

void MainWindow::setupErrorHandling() {
    auto& config = ApplicationConfig::instance();
    
    // Enable error logging if configured
    if (config.enableLogging()) {
        ErrorHandler::enableLogging(config.logFilePath());
    }
    
    // Setup recovery callbacks
    ErrorHandler::setRecoveryCallback(ErrorHandler::ErrorType::DataLoad, [this]() {
        // Try to reload the last used file
        auto& appConfig = ApplicationConfig::instance();
        QString lastPath = appConfig.lastUsedDataPath();
        if (!lastPath.isEmpty()) {
            onSelectCSVFile();
            return true;
        }
        return false;
    });
    
    ErrorHandler::setRecoveryCallback(ErrorHandler::ErrorType::UI, [this]() {
        // Reset UI to default state
        m_statusLabel->setText(UIStrings::READY);
        m_progressBar->setVisible(false);
        m_uploadDataButton->setEnabled(true);
        return true;
    });
}

void MainWindow::loadApplicationConfig() {
    auto& config = ApplicationConfig::instance();
    config.load();
    
    // Apply configuration to UI
    // This will be called during setupUI
}

void MainWindow::saveApplicationConfig() {
    auto& config = ApplicationConfig::instance();
    
    // Save current window size
    config.setDefaultWindowSize(size());
    
    // Save last used data path if available
    if (!m_lastRows.empty()) {
        // Could save the path here if we track it
    }
    
    config.save();
}

void MainWindow::onApplicationShutdown() {
    // Cancel any running tasks
    AsyncTaskManager::instance().cancelAllTasks();
    
    // Save configuration
    saveApplicationConfig();
}

void MainWindow::processDataWithUserConfig(const std::vector<DataRow>& rows, 
                                          const BarrierConfig& cfg, 
                                          const DataPreprocessor::Params& params) {
    try {
        m_statusLabel->setText("Processing data...");
        qApp->processEvents();
        
        if (rows.empty()) {
            throw std::runtime_error("No input data rows");
        }
        
        // Validate the config
        BarrierConfig testCfg = cfg;
        testCfg.validate();
        
        // Preprocess the data
        auto processed = DataPreprocessor::preprocess(rows, params);
        if (processed.empty()) {
            throw std::runtime_error("Data preprocessing returned empty result");
        }
        
        // Generate labeled events
        DataServiceImpl dataService;
        auto labeled = dataService.generateLabeledEvents(processed, cfg);
        
        // Display results
        plotLabeledEvents(processed, labeled);
        m_statusLabel->setText(QString("Success! Processed %1 rows, found %2 events").arg(processed.size()).arg(labeled.size()));
        m_uploadDataButton->setEnabled(true);
        
    } catch (const std::exception& e) {
        ErrorHandler::ErrorInfo errorInfo(
            ErrorHandler::ErrorType::DataLoad,
            ErrorHandler::Severity::Error,
            "Data Processing Failed",
            QString::fromStdString(e.what())
        );
        ErrorHandler::handleError(errorInfo, this);
        m_uploadDataButton->setEnabled(true);
    }
}
