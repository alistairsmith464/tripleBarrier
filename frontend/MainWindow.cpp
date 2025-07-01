#include "MainWindow.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <QSizePolicy>
#include <QFont>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_fileHandler(new FileHandler())
{
    setupUI();
    setWindowTitle("Triple Barrier - File Upload");
    setMinimumSize(600, 400);
    resize(800, 500);
}

MainWindow::~MainWindow()
{
    delete m_fileHandler;
}

void MainWindow::setupUI()
{
    // Create central widget
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // Create main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(24);
    mainLayout->setContentsMargins(40, 36, 40, 36);
    centralWidget->setStyleSheet("background: #f4f8fb;");

    // Title
    m_titleLabel = new QLabel("Triple Barrier Data Uploader", this);
    QFont titleFont = m_titleLabel->font();
    titleFont.setPointSize(22);
    titleFont.setBold(true);
    m_titleLabel->setFont(titleFont);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    m_titleLabel->setStyleSheet("color: #22313f; margin-bottom: 8px; letter-spacing: 1px;");

    // Status label
    m_statusLabel = new QLabel("Click 'Upload Data' to begin", this);
    m_statusLabel->setAlignment(Qt::AlignCenter);
    m_statusLabel->setStyleSheet("color: #e74c3c; font-size: 15px; font-weight: bold; margin-bottom: 8px;");

    // Progress bar (initially hidden)
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    m_progressBar->setStyleSheet(
        "QProgressBar { border: 1px solid #b2bec3; border-radius: 5px; background: #eaf0f6; height: 18px; }"
        "QProgressBar::chunk { background: #3498db; border-radius: 5px; }"
    );

    // Buttons layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(24);

    m_uploadDataButton = new QPushButton("Upload Data", this);
    m_uploadDataButton->setMinimumHeight(44);
    m_uploadDataButton->setMinimumWidth(220);
    m_uploadDataButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #3498db;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 17px;"
        "    font-weight: bold;"
        "    letter-spacing: 1px;"
        "    box-shadow: 0 2px 8px #b2bec3;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #21618c;"
        "}"
    );

    m_uploadMenu = new QMenu(this);
    m_csvAction = new QAction("Upload CSV", this);
    m_uploadMenu->addAction(m_csvAction);
    m_uploadDataButton->setMenu(m_uploadMenu);

    m_clearButton = new QPushButton("Clear", this);
    m_clearButton->setMinimumHeight(44);
    m_clearButton->setMinimumWidth(220);
    m_clearButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #b2bec3;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 17px;"
        "    font-weight: bold;"
        "    letter-spacing: 1px;"
        "    box-shadow: 0 2px 8px #dfe6e9;"
        "}"
        "QPushButton:hover {"
        "    background-color: #636e72;"
        "}"
    );

    buttonLayout->addWidget(m_uploadDataButton);
    buttonLayout->addWidget(m_clearButton);

    // File info display
    m_fileInfoDisplay = new QTextEdit(this);
    m_fileInfoDisplay->setReadOnly(true);
    m_fileInfoDisplay->setPlaceholderText("Data summary will appear here after upload...");
    m_fileInfoDisplay->setStyleSheet(
        "QTextEdit {"
        "    border: 2px solid #bdc3c7;"
        "    border-radius: 8px;"
        "    padding: 16px;"
        "    background-color: #f8f9fa;"
        "    font-family: 'Fira Mono', 'Consolas', 'Courier New', monospace;"
        "    font-size: 15px;"
        "    color: #22313f;"
        "}"
    );

    // Add all components to main layout
    mainLayout->addWidget(m_titleLabel);
    mainLayout->addWidget(m_statusLabel);
    mainLayout->addWidget(m_progressBar);
    mainLayout->addLayout(buttonLayout);
    mainLayout->addWidget(m_fileInfoDisplay, 1);

    // Connect signals
    connect(m_csvAction, &QAction::triggered, this, &MainWindow::onSelectCSVFile);
    connect(m_clearButton, &QPushButton::clicked, this, &MainWindow::onClearButtonClicked);
}

void MainWindow::onUploadDataButtonClicked() {
    // Not used, menu is attached directly to button
}

void MainWindow::onSelectCSVFile() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select CSV File",
        "",
        "CSV Files (*.csv);;All Files (*.*)"
    );
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
        showUploadSuccess(fileName, rows);
        showDataSummary(rows);
    } catch (const std::exception& ex) {
        showUploadError(QString("Failed to load CSV: %1").arg(ex.what()));
    }
    m_progressBar->setVisible(false);
    m_uploadDataButton->setEnabled(true);
}

void MainWindow::onClearButtonClicked()
{
    m_fileInfoDisplay->clear();
    m_statusLabel->setText("Select a file to upload");
    m_statusLabel->setStyleSheet("color: #7f8c8d; font-size: 12px;");
}

void MainWindow::showUploadSuccess(const QString& filePath, const std::vector<DataRow>& rows) {
    m_statusLabel->setText("✓ Data loaded successfully!");
    m_statusLabel->setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;");
    QMessageBox::information(this, "Upload Successful", QString("Loaded %1 rows from %2").arg(rows.size()).arg(filePath));
}

void MainWindow::showUploadError(const QString& error) {
    m_statusLabel->setText("✗ Data load failed");
    m_statusLabel->setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;");
    QMessageBox::critical(this, "Load Failed", error);
}

void MainWindow::showDataSummary(const std::vector<DataRow>& rows) {
    if (rows.empty()) {
        m_fileInfoDisplay->setText("No data loaded.");
        return;
    }
    QStringList lines;
    lines << QString("Rows loaded: %1").arg(rows.size());
    lines << "Columns: timestamp, price, open, high, low, close, volume";
    int preview = std::min<int>(rows.size(), 5);
    lines << "\nSample rows:";
    for (int i = 0; i < preview; ++i) {
        const DataRow& r = rows[i];
        lines << QString("%1 | %2 | %3 | %4 | %5 | %6 | %7")
            .arg(QString::fromStdString(r.timestamp))
            .arg(r.price)
            .arg(r.open ? QString::number(*r.open) : "")
            .arg(r.high ? QString::number(*r.high) : "")
            .arg(r.low ? QString::number(*r.low) : "")
            .arg(r.close ? QString::number(*r.close) : "")
            .arg(r.volume ? QString::number(*r.volume) : "");
    }
    m_fileInfoDisplay->setText(lines.join("\n"));
}

#include "MainWindow.moc"
