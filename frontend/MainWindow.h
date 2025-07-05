#pragma once

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
#include <vector>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QDateTimeAxis>
#include <QDateTime>
#include "FileHandler.h"
#include "../backend/data/CSVDataSource.h"
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"
#include "FeatureSelectionDialog.h"
#include "ui/MainWindowUI.h"

QT_BEGIN_NAMESPACE
QT_END_NAMESPACE

extern std::vector<PreprocessedRow> g_lastRows;
extern std::vector<LabeledEvent> g_lastLabeledEvents;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onClearButtonClicked();
    void onUploadDataButtonClicked();
    void onSelectCSVFile();
    void onMLButtonClicked(); // Slot for ML button
    void onExportCSVClicked(); // Slot for exporting features/labels to CSV

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
    QAction *m_exportCSVAction;
    QChartView *m_chartView;

    // File handler (stack-allocated, not a pointer)
    FileHandler m_fileHandler;

    // UI
    MainWindowUI m_ui;
};
