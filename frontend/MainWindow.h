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
#include "FileHandler.h"
#include "../backend/data/CSVDataSource.h"
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/TripleBarrierLabeler.h"
#include "../backend/data/LabeledEvent.h"

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
    void onUploadDataButtonClicked();
    void onSelectCSVFile();

private:
    void setupUI();
    void showUploadSuccess(const QString& filePath);
    void showUploadSuccess(const QString& filePath, const std::vector<DataRow>& rows);
    void showUploadError(const QString& error);
    void showDataSummary(const std::vector<DataRow>& rows);
    void showPreprocessedSummary(const std::vector<PreprocessedRow>& rows);
    void plotLabeledEvents(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& labeledEvents);

    // UI components
    QPushButton *m_uploadDataButton;
    QPushButton *m_clearButton;
    QLabel *m_titleLabel;
    QLabel *m_statusLabel;
    QTextEdit *m_fileInfoDisplay;
    QProgressBar *m_progressBar;
    QMenu *m_uploadMenu;
    QAction *m_csvAction;
    QChartView *m_chartView;

    // File handler (stack-allocated, not a pointer)
    FileHandler m_fileHandler;
};
