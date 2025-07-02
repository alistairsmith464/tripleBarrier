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
#include "FileHandler.h"
#include "../backend/data/CSVDataSource.h"
#include "../backend/data/PreprocessedRow.h"

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

    // UI components
    QPushButton *m_uploadDataButton;
    QPushButton *m_clearButton;
    QLabel *m_titleLabel;
    QLabel *m_statusLabel;
    QTextEdit *m_fileInfoDisplay;
    QProgressBar *m_progressBar;
    QMenu *m_uploadMenu;
    QAction *m_csvAction;

    // File handler (stack-allocated, not a pointer)
    FileHandler m_fileHandler;
};
