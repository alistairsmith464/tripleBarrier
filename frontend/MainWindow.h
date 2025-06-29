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
#include "../backend/FileHandler.h"

QT_BEGIN_NAMESPACE
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onUploadButtonClicked();
    void onClearButtonClicked();

private:
    void setupUI();
    void showUploadSuccess(const QString& filePath);
    void showUploadError(const QString& error);

    // UI components
    QPushButton *m_uploadButton;
    QPushButton *m_clearButton;
    QLabel *m_titleLabel;
    QLabel *m_statusLabel;
    QTextEdit *m_fileInfoDisplay;
    QProgressBar *m_progressBar;
    
    // Backend
    FileHandler *m_fileHandler;
};
