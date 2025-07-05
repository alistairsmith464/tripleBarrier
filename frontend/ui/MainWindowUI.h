#pragma once
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QMenu>
#include <QAction>
#include <QtCharts/QChartView>

struct MainWindowUI {
    QLabel *titleLabel = nullptr;
    QLabel *statusLabel = nullptr;
    QProgressBar *progressBar = nullptr;
    QPushButton *uploadDataButton = nullptr;
    QPushButton *clearButton = nullptr;
    QMenu *uploadMenu = nullptr;
    QAction *csvAction = nullptr;
    QAction *exportCSVAction = nullptr;
    QChartView *chartView = nullptr;
    QPushButton *mlButton = nullptr;
    QVBoxLayout *mainLayout = nullptr;
    QHBoxLayout *buttonLayout = nullptr;

    void setup(QWidget *parent);
};
