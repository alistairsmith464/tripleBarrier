#include "MainWindowUI.h"
#include <QFont>

void MainWindowUI::setup(QWidget *parent) {
    // Create main layout
    mainLayout = new QVBoxLayout(parent);
    mainLayout->setSpacing(24);
    mainLayout->setContentsMargins(40, 36, 40, 36);
    parent->setStyleSheet("background: #f4f8fb;");

    // Title
    titleLabel = new QLabel("Triple Barrier Data Uploader", parent);
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(22);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    titleLabel->setAlignment(Qt::AlignCenter);
    titleLabel->setStyleSheet("color: #22313f; margin-bottom: 8px; letter-spacing: 1px;");

    // Status label
    statusLabel = new QLabel("Click 'Upload Data' to begin", parent);
    statusLabel->setAlignment(Qt::AlignCenter);
    statusLabel->setStyleSheet("color: #e74c3c; font-size: 15px; font-weight: bold; margin-bottom: 8px;");

    // Progress bar
    progressBar = new QProgressBar(parent);
    progressBar->setVisible(false);
    progressBar->setStyleSheet(
        "QProgressBar { border: 1px solid #b2bec3; border-radius: 5px; background: #eaf0f6; height: 18px; }"
        "QProgressBar::chunk { background: #3498db; border-radius: 5px; }"
    );

    // Buttons layout
    buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(24);

    uploadDataButton = new QPushButton("Upload Data", parent);
    uploadDataButton->setMinimumHeight(44);
    uploadDataButton->setMinimumWidth(220);
    uploadDataButton->setStyleSheet(
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

    uploadMenu = new QMenu(parent);
    csvAction = new QAction("Upload CSV", parent);
    uploadMenu->addAction(csvAction);
    exportCSVAction = new QAction("Export Features/Labels to CSV", parent);
    uploadMenu->addAction(exportCSVAction);
    uploadDataButton->setMenu(uploadMenu);

    clearButton = new QPushButton("Clear", parent);
    clearButton->setMinimumHeight(44);
    clearButton->setMinimumWidth(220);
    clearButton->setStyleSheet(
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

    buttonLayout->addWidget(uploadDataButton);
    buttonLayout->addWidget(clearButton);

    // Add all components to main layout
    mainLayout->addWidget(titleLabel);
    mainLayout->addWidget(statusLabel);
    mainLayout->addWidget(progressBar);
    mainLayout->addLayout(buttonLayout);

    // Chart for graphical display
    chartView = new QChartView(parent);
    chartView->setMinimumHeight(300);
    mainLayout->addWidget(chartView);

    // Machine Learning button
    mlButton = new QPushButton("Next: Machine Learning", parent);
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
    mainLayout->addWidget(mlButton);
}
