#include "FeaturePreviewDialog.h"
#include "../MLHyperparamsDialog.h"
#include "../utils/FeaturePreviewUtils.h"
#include "../services/MLService.h"
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QDialogButtonBox>
#include <QPushButton>

FeaturePreviewDialog::FeaturePreviewDialog(
    const QSet<QString>& selectedFeatures,
    const std::vector<PreprocessedRow>& rows,
    const std::vector<LabeledEvent>& labeledEvents,
    QWidget* parent)
    : QDialog(parent)
    , m_selectedFeatures(selectedFeatures)
    , m_rows(rows)
    , m_labeledEvents(labeledEvents)
{
    setWindowTitle("Feature Preview");
    setupUI();
}

void FeaturePreviewDialog::setupUI() {
    QVBoxLayout* vbox = new QVBoxLayout(this);
    
    // Create and populate feature table
    createFeatureTable();
    
    // Data info label
    m_dataInfoLabel = new QLabel(this);
    m_dataInfoLabel->setStyleSheet("color: #2c3e50; font-size: 12px; padding: 5px;");
    vbox->addWidget(m_dataInfoLabel);
    updateDataInfo();
    
    // Barrier diagnostics label
    m_debugInfoLabel = new QLabel(this);
    m_debugInfoLabel->setStyleSheet("color: #8e44ad; font-size: 11px; padding: 5px;");
    vbox->addWidget(m_debugInfoLabel);
    updateBarrierDiagnostics();
    
    // ML pipeline button
    m_runMLButton = new QPushButton("Run ML Pipeline", this);
    vbox->addWidget(m_runMLButton);
    
    // Results labels
    m_metricsLabel = new QLabel(this);
    m_importancesLabel = new QLabel(this);
    vbox->addWidget(m_metricsLabel);
    vbox->addWidget(m_importancesLabel);
    
    // Hyperparameter tuning checkbox
    m_tuneHyperparamsCheckBox = new QCheckBox("Auto-tune hyperparameters (grid search)", this);
    m_tuneHyperparamsCheckBox->setToolTip("If checked, the pipeline will automatically search for the best hyperparameters (n_rounds, max_depth, nthread).");
    vbox->addWidget(m_tuneHyperparamsCheckBox);
    
    // Connect signals
    connect(m_runMLButton, &QPushButton::clicked, this, &FeaturePreviewDialog::onRunMLClicked);
    
    // OK button
    QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok, this);
    connect(box, &QDialogButtonBox::accepted, this, &QDialog::accept);
    vbox->addWidget(box);
}

void FeaturePreviewDialog::createFeatureTable() {
    // Create ML service instance for feature extraction
    MLServiceImpl mlService;
    auto result = mlService.extractFeatures(m_rows, m_labeledEvents, m_selectedFeatures);
    
    // Create table widget
    QTableWidget* table = new QTableWidget(int(result.features.size()), m_selectedFeatures.size(), this);
    
    // Set headers
    QStringList headers;
    for (const QString& feat : m_selectedFeatures) {
        headers << feat;
    }
    table->setHorizontalHeaderLabels(headers);
    
    // Populate table
    auto featureMap = FeatureExtractor::getFeatureMapping();
    int col = 0;
    for (const QString& feat : m_selectedFeatures) {
        std::string backendId = featureMap[feat.toStdString()];
        for (int row = 0; row < int(result.features.size()); ++row) {
            double val = result.features[row].count(backendId) ? result.features[row][backendId] : NAN;
            table->setItem(row, col, new QTableWidgetItem(QString::number(val)));
        }
        ++col;
    }
    
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    layout()->addWidget(table);
}

void FeaturePreviewDialog::updateDataInfo() {
    QString info = QString("<b>Data Summary:</b> %1 price records, %2 labeled events")
                  .arg(m_rows.size()).arg(m_labeledEvents.size());
    m_dataInfoLabel->setText(info);
}

void FeaturePreviewDialog::updateBarrierDiagnostics() {
    auto diagnostics = PortfolioSimulator::analyzeBarriers(m_labeledEvents, m_rows);
    QString text = FeaturePreviewUtils::formatBarrierDiagnostics(diagnostics, m_labeledEvents);
    m_debugInfoLabel->setText(text);
}

void FeaturePreviewDialog::onRunMLClicked() {
    MLHyperparamsDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted) return;
    
    // Determine TTBM mode
    bool is_ttbm = false;
    if (!m_labeledEvents.empty()) {
        is_ttbm = m_labeledEvents[0].is_ttbm;
    }
    
    // Configure ML pipeline
    MLConfig config;
    config.selectedFeatures = m_selectedFeatures;
    config.useTTBM = is_ttbm;
    config.crossValidationRatio = 0.2;
    config.randomSeed = 42;
    
    // Run ML pipeline using service
    MLServiceImpl mlService;
    MLResults results = mlService.runMLPipeline(m_rows, m_labeledEvents, config);
    
    if (!results.success) {
        m_metricsLabel->setText(QString("<font color='red'>ML Error: %1</font>").arg(results.errorMessage));
        return;
    }
    
    // Update data info label with model information
    QString modelInfo = FeaturePreviewUtils::formatModelInfo(is_ttbm, false, m_labeledEvents);
    m_dataInfoLabel->setText(modelInfo);
    
    // Display results
    if (is_ttbm) {
        showMLRegressionResults(results);
    } else {
        showMLClassificationResults(results);
    }
}

void FeaturePreviewDialog::showMLClassificationResults(const MLResults& results) {
    // Display portfolio simulation results
    QString metrics = FeaturePreviewUtils::formatPortfolioResults(results.portfolioResult, false);
    m_metricsLabel->setText(metrics);
    m_importancesLabel->setText("");
}

void FeaturePreviewDialog::showMLRegressionResults(const MLResults& results) {
    // Display portfolio simulation results 
    QString metrics = FeaturePreviewUtils::formatPortfolioResults(results.portfolioResult, true);
    m_metricsLabel->setText(metrics);
    m_importancesLabel->setText("");
    
    // Show sample trading decisions
    QString debug = FeaturePreviewUtils::formatSampleTradingDecisions(results.predictions, true, 10);
    m_debugInfoLabel->setText(debug);
}
