#include "FeaturePreviewDialog.h"
#include "../MLHyperparamsDialog.h"
#include "../utils/FeaturePreviewUtils.h"
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
    // Convert selected features to backend format
    std::set<std::string> selectedFeaturesStd = FeaturePreviewUtils::convertQSetToStdSet(m_selectedFeatures);
    
    // Extract features using the backend
    auto result = FeatureExtractor::extractFeaturesForClassification(selectedFeaturesStd, m_rows, m_labeledEvents);
    
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
    
    // Determine if this is TTBM or hard barrier
    bool is_ttbm = false;
    if (!m_labeledEvents.empty()) {
        is_ttbm = m_labeledEvents[0].is_ttbm;
    }
    
    bool tune = m_tuneHyperparamsCheckBox && m_tuneHyperparamsCheckBox->isChecked();
    
    // Setup ML pipeline configuration
    MLPipeline::PipelineConfig config;
    config.split_type = MLPipeline::Chronological;
    config.train_ratio = 0.7;
    config.val_ratio = 0.15;
    config.test_ratio = 0.15;
    config.n_splits = 3;
    config.embargo = 0;
    config.random_seed = 42;
    
    if (is_ttbm) {
        config.n_rounds = tune ? dlg.nRounds() : 500;
        config.max_depth = tune ? dlg.maxDepth() : 3;
        config.nthread = dlg.nThread();
        config.objective = "reg:squarederror";
        config.train_ratio = 0.8;
        config.val_ratio = 0.1;
        config.test_ratio = 0.1;
    } else {
        config.n_rounds = dlg.nRounds();
        config.max_depth = dlg.maxDepth();
        config.nthread = dlg.nThread();
        config.objective = "binary:logistic";
    }
    
    // Update data info label with model information
    QString modelInfo = FeaturePreviewUtils::formatModelInfo(is_ttbm, tune, m_labeledEvents);
    m_dataInfoLabel->setText(modelInfo);
    
    // Run the appropriate ML pipeline
    std::set<std::string> selectedFeaturesStd = FeaturePreviewUtils::convertQSetToStdSet(m_selectedFeatures);
    
    if (is_ttbm) {
        auto featureResult = FeatureExtractor::extractFeaturesForRegression(selectedFeaturesStd, m_rows, m_labeledEvents);
        
        MLPipeline::RegressionPipelineResult result;
        if (tune) {
            result = MLPipeline::runPipelineRegressionWithTuning(featureResult.features, featureResult.labels_double, featureResult.returns, config);
        } else {
            result = MLPipeline::runPipelineRegression(featureResult.features, featureResult.labels_double, featureResult.returns, config);
        }
        
        showMLRegressionResults(result);
    } else {
        auto featureResult = FeatureExtractor::extractFeaturesForClassification(selectedFeaturesStd, m_rows, m_labeledEvents);
        
        MLPipeline::PipelineResult result;
        if (tune) {
            result = MLPipeline::runPipelineWithTuning(featureResult.features, featureResult.labels, featureResult.returns, config);
        } else {
            result = MLPipeline::runPipeline(featureResult.features, featureResult.labels, featureResult.returns, config);
        }
        
        showMLClassificationResults(result);
    }
}

void FeaturePreviewDialog::showMLClassificationResults(const MLPipeline::PipelineResult& result) {
    // Convert integer predictions to double for portfolio simulation
    std::vector<double> double_predictions;
    for (int pred : result.predictions) {
        double_predictions.push_back(static_cast<double>(pred));
    }
    
    // Run portfolio simulation
    auto portfolioResults = PortfolioSimulator::runSimulation(double_predictions, m_labeledEvents, false);
    
    // Display results
    QString metrics = FeaturePreviewUtils::formatPortfolioResults(portfolioResults, false);
    m_metricsLabel->setText(metrics);
    m_importancesLabel->setText("");
}

void FeaturePreviewDialog::showMLRegressionResults(const MLPipeline::RegressionPipelineResult& result) {
    // Run portfolio simulation
    auto portfolioResults = PortfolioSimulator::runSimulation(result.predictions, m_labeledEvents, true);
    
    // Display results
    QString metrics = FeaturePreviewUtils::formatPortfolioResults(portfolioResults, true);
    m_metricsLabel->setText(metrics);
    m_importancesLabel->setText("");
    
    // Show sample trading decisions
    QString debug = FeaturePreviewUtils::formatSampleTradingDecisions(result.predictions, true, 10);
    m_debugInfoLabel->setText(debug);
}
