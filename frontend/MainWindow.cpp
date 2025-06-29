#include "MainWindow.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <QSizePolicy>
#include <QFont>

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
    mainLayout->setSpacing(20);
    mainLayout->setContentsMargins(30, 30, 30, 30);

    // Title
    m_titleLabel = new QLabel("Triple Barrier File Upload", this);
    QFont titleFont = m_titleLabel->font();
    titleFont.setPointSize(18);
    titleFont.setBold(true);
    m_titleLabel->setFont(titleFont);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    m_titleLabel->setStyleSheet("color: #2c3e50; margin-bottom: 10px;");

    // Status label
    m_statusLabel = new QLabel("Select a file to upload", this);
    m_statusLabel->setAlignment(Qt::AlignCenter);
    m_statusLabel->setStyleSheet("color: #7f8c8d; font-size: 12px;");

    // Progress bar (initially hidden)
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);

    // Buttons layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    
    m_uploadButton = new QPushButton("Upload File", this);
    m_uploadButton->setMinimumHeight(40);
    m_uploadButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #3498db;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 5px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #21618c;"
        "}"
    );

    m_clearButton = new QPushButton("Clear", this);
    m_clearButton->setMinimumHeight(40);
    m_clearButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #95a5a6;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 5px;"
        "    font-size: 14px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #7f8c8d;"
        "}"
    );

    buttonLayout->addWidget(m_uploadButton);
    buttonLayout->addWidget(m_clearButton);

    // File info display
    m_fileInfoDisplay = new QTextEdit(this);
    m_fileInfoDisplay->setReadOnly(true);
    m_fileInfoDisplay->setPlaceholderText("File information will appear here after upload...");
    m_fileInfoDisplay->setStyleSheet(
        "QTextEdit {"
        "    border: 2px solid #bdc3c7;"
        "    border-radius: 5px;"
        "    padding: 10px;"
        "    background-color: #f8f9fa;"
        "    font-family: 'Courier New', monospace;"
        "}"
    );

    // Add all components to main layout
    mainLayout->addWidget(m_titleLabel);
    mainLayout->addWidget(m_statusLabel);
    mainLayout->addWidget(m_progressBar);
    mainLayout->addLayout(buttonLayout);
    mainLayout->addWidget(m_fileInfoDisplay, 1); // Give it more space

    // Connect signals
    connect(m_uploadButton, &QPushButton::clicked, this, &MainWindow::onUploadButtonClicked);
    connect(m_clearButton, &QPushButton::clicked, this, &MainWindow::onClearButtonClicked);
}

void MainWindow::onUploadButtonClicked()
{
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select File to Upload",
        "",
        "All Files (*.*)"
    );

    if (fileName.isEmpty()) {
        return; // User cancelled
    }

    // Show progress
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0); // Indeterminate progress
    m_statusLabel->setText("Uploading file...");
    m_uploadButton->setEnabled(false);

    // Simulate some processing time (in real app, this might be in a separate thread)
    QApplication::processEvents();

    // Attempt to upload the file
    if (m_fileHandler->uploadFile(fileName)) {
        showUploadSuccess(m_fileHandler->getLastUploadedFile());
    } else {
        showUploadError("Failed to upload file. Please check file permissions and try again.");
    }

    // Hide progress and re-enable button
    m_progressBar->setVisible(false);
    m_uploadButton->setEnabled(true);
}

void MainWindow::onClearButtonClicked()
{
    m_fileInfoDisplay->clear();
    m_statusLabel->setText("Select a file to upload");
    m_statusLabel->setStyleSheet("color: #7f8c8d; font-size: 12px;");
}

void MainWindow::showUploadSuccess(const QString& filePath)
{
    m_statusLabel->setText("✓ File uploaded successfully!");
    m_statusLabel->setStyleSheet("color: #27ae60; font-size: 12px; font-weight: bold;");

    // Display file information
    QString fileInfo = m_fileHandler->getFileInfo(filePath);
    m_fileInfoDisplay->setText(fileInfo);

    // Show success message
    QMessageBox::information(this, "Upload Successful", 
                           QString("File has been successfully uploaded!\n\nLocation: %1").arg(filePath));
}

void MainWindow::showUploadError(const QString& error)
{
    m_statusLabel->setText("✗ Upload failed");
    m_statusLabel->setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: bold;");

    // Show error message
    QMessageBox::critical(this, "Upload Failed", error);
}

#include "MainWindow.moc"
