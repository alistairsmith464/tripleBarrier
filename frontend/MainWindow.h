#pragma once

#include "plot/LabeledEventPlotter.h"
#include "services/DataService.h"
#include "services/MLService.h"
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
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QDateTimeAxis>
#include <QDateTime>
#include "FeatureSelectionDialog.h"
#include "ui/MainWindowUI.h"

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
    void onMLButtonClicked(); // Slot for ML button

private:
    void setupUI();
    void showUploadSuccess(const QString& filePath);
    void showUploadError(const QString& error);
    void plotLabeledEvents(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents);

    // UI components
    QPushButton *m_uploadDataButton;
    QPushButton *m_clearButton;
    QLabel *m_titleLabel;
    QLabel *m_statusLabel;
    QProgressBar *m_progressBar;
    QMenu *m_uploadMenu;
    QAction *m_csvAction;
    QChartView *m_chartView;
    QComboBox *m_plotModeComboBox;

    // UI
    MainWindowUI m_ui;

    // Services
    std::unique_ptr<DataService> m_dataService;
    std::unique_ptr<MLService> m_mlService;

    // Data state
    std::vector<PreprocessedRow> m_lastRows;
    std::vector<LabeledEvent> m_lastLabeledEvents;

    PlotMode m_plotMode = PlotMode::TimeSeries;
};
