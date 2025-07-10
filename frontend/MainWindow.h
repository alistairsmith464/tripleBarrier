#pragma once

#include "plot/LabeledEventPlotter.h"
#include "services/DataService.h"
#include "services/MLService.h"
#include "utils/ErrorHandler.h"
#include "utils/AsyncTaskManager.h"
#include "config/ApplicationConfig.h"
#include "ui/UIStrings.h"
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QProgressBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenu>
#include <QAction>
#include <QComboBox>
#include <vector>
#include <memory>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QDateTimeAxis>
#include <QDateTime>
#include "FeatureSelectionDialog.h"
#include "ui/MainWindowUI.h"
#include "../backend/data/DataRow.h"
#include "../backend/data/BarrierConfig.h"
#include "../backend/data/DataPreprocessor.h"

QT_BEGIN_NAMESPACE
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onClearButtonClicked();
    void onSelectCSVFile();
    void onMLButtonClicked();
    void onApplicationShutdown();

private:
    void setupUI();
    void setupErrorHandling();
    void loadApplicationConfig();
    void saveApplicationConfig();
    void showUploadSuccess(const QString& filePath);
    void showUploadError(const QString& error);
    void plotLabeledEvents(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents);
    void showBarrierConfigurationDialog(const std::vector<DataRow>& rows);
    void processDataWithConfig(const std::vector<DataRow>& rows, const BarrierConfig& cfg, const DataPreprocessor::Params& params);
    void processDataWithUserConfig(const std::vector<DataRow>& rows, const BarrierConfig& cfg, const DataPreprocessor::Params& params);

    QPushButton *m_uploadDataButton;
    QPushButton *m_clearButton;
    QLabel *m_titleLabel;
    QLabel *m_statusLabel;
    QProgressBar *m_progressBar;
    QMenu *m_uploadMenu;
    QAction *m_csvAction;
    QChartView *m_chartView;
    QComboBox *m_plotModeComboBox;

    MainWindowUI m_ui;

    std::unique_ptr<DataService> m_dataService;
    std::unique_ptr<MLService> m_mlService;

    std::vector<PreprocessedRow> m_lastRows;
    std::vector<LabeledEvent> m_lastLabeledEvents;

    PlotMode m_plotMode = PlotMode::TimeSeries;
};
