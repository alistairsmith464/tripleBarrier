#include "FeaturePreviewDialog.h"
#include "../MLHyperparamsDialog.h"
#include "../utils/FeaturePreviewUtils.h"
#include "../services/MLService.h"
#include "../ui/UIStrings.h"
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QScrollArea>

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
    setWindowTitle(UIStrings::FEATURE_PREVIEW_TITLE);
    setupUI();
}

void FeaturePreviewDialog::setupUI() {
    QWidget* contentWidget = new QWidget(this);
    QVBoxLayout* vbox = new QVBoxLayout(contentWidget);

    createFeatureTable(vbox);

    m_dataInfoLabel = new QLabel(this);
    m_dataInfoLabel->setStyleSheet("color: #2c3e50; font-size: 12px; padding: 5px;");
    vbox->addWidget(m_dataInfoLabel);
    updateDataInfo();

    m_debugInfoLabel = new QLabel(this);
    m_debugInfoLabel->setStyleSheet("color: #8e44ad; font-size: 11px; padding: 5px;");
    vbox->addWidget(m_debugInfoLabel);
    updateBarrierDiagnostics();

    m_tuneHyperparamsCheckBox = new QCheckBox(UIStrings::AUTO_TUNE_HYPERPARAMS, this);
    m_tuneHyperparamsCheckBox->setToolTip("If checked, the pipeline will automatically search for the best hyperparameters (n_rounds, max_depth, nthread).");
    vbox->addWidget(m_tuneHyperparamsCheckBox);

    m_runMLButton = new QPushButton(UIStrings::RUN_ML, this);
    vbox->addWidget(m_runMLButton);

    m_metricsLabel = new QLabel(this);
    m_importancesLabel = new QLabel(this);
    vbox->addWidget(m_metricsLabel);
    vbox->addWidget(m_importancesLabel);

    m_tradeLogTable = new QTableWidget(this);
    m_tradeLogTable->setColumnCount(4);
    QStringList headers;
    headers << "Signal" << "Trade Return" << "Capital Before" << "Capital After";
    m_tradeLogTable->setHorizontalHeaderLabels(headers);
    m_tradeLogTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_tradeLogTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tradeLogTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tradeLogTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_tradeLogTable->hide();
    m_tradeLogTable->setMinimumHeight(300);
    m_tradeLogTable->setMinimumWidth(800);
    vbox->addWidget(m_tradeLogTable);

    connect(m_runMLButton, &QPushButton::clicked, this, &FeaturePreviewDialog::onRunMLClicked);

    QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok, this);
    box->button(QDialogButtonBox::Ok)->setText(UIStrings::OK);
    connect(box, &QDialogButtonBox::accepted, this, &QDialog::accept);
    vbox->addWidget(box);

    QScrollArea* scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setWidget(contentWidget);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(scrollArea);
    setLayout(mainLayout);

    setMinimumSize(1200, 600);
}

void FeaturePreviewDialog::createFeatureTable(QVBoxLayout* vbox) {
    MLServiceImpl mlService;
    bool is_ttbm = false;
    if (!m_labeledEvents.empty()) {
        is_ttbm = m_labeledEvents[0].is_ttbm;
    }

    FeatureExtractor::FeatureExtractionResult result;
    if (is_ttbm) {
        result = mlService.getFeatureService()->extractFeaturesForRegression(m_rows, m_labeledEvents, m_selectedFeatures);
    } else {
        result = mlService.getFeatureService()->extractFeaturesForClassification(m_rows, m_labeledEvents, m_selectedFeatures);
    }

    QTableWidget* table = new QTableWidget(int(result.features.size()), m_selectedFeatures.size(), this);

    QStringList headers;
    for (const QString& feat : m_selectedFeatures) {
        headers << feat;
    }
    table->setHorizontalHeaderLabels(headers);

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
    vbox->addWidget(table);
}

void FeaturePreviewDialog::updateDataInfo() {
    QString info = QString("<b>Data Summary:</b> %1 price records, %2 labeled events")
                  .arg(m_rows.size()).arg(m_labeledEvents.size());
    m_dataInfoLabel->setText(info);
}

void FeaturePreviewDialog::updateBarrierDiagnostics() {
    auto diagnostics = MLPipeline::analyzeBarriers(m_labeledEvents, m_rows);
    QString text = FeaturePreviewUtils::formatBarrierDiagnostics(diagnostics, m_labeledEvents);
    m_debugInfoLabel->setText(text);
}

void FeaturePreviewDialog::onRunMLClicked() {
    MLHyperparamsDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted) return;
    
    bool is_ttbm = false;
    if (!m_labeledEvents.empty()) {
        is_ttbm = m_labeledEvents[0].is_ttbm;
    }
    
    MLConfig config;
    config.selectedFeatures = m_selectedFeatures;
    config.useTTBM = is_ttbm;
    config.crossValidationRatio = 0.2;
    config.randomSeed = 42;
    config.tuneHyperparameters = m_tuneHyperparamsCheckBox->isChecked();
    
    MLServiceImpl mlService;
    MLResults results = mlService.runMLPipeline(m_rows, m_labeledEvents, config);
    
    if (!results.success) {
        m_metricsLabel->setText(QString("<font color='red'>ML Error: %1</font>").arg(results.errorMessage));
        return;
    }
    
    QString modelInfo = FeaturePreviewUtils::formatModelInfo(is_ttbm, false, m_labeledEvents);
    m_dataInfoLabel->setText(modelInfo);
    
    if (is_ttbm) {
        showMLRegressionResults(results);
    } else {
        showMLClassificationResults(results);
    }
}

void FeaturePreviewDialog::displayTradeLog(const std::vector<MLPipeline::TradeLogEntry>& tradeLog) {
    m_tradeLogTable->setRowCount(static_cast<int>(tradeLog.size()));
    m_tradeLogTable->show();

    for (int i = 0; i < static_cast<int>(tradeLog.size()); ++i) {
        const auto& entry = tradeLog[i];
        m_tradeLogTable->setItem(i, 1, new QTableWidgetItem(QString::number(entry.signal, 'f', 4)));
        m_tradeLogTable->setItem(i, 2, new QTableWidgetItem(QString::number(entry.trade_return, 'f', 4)));
        m_tradeLogTable->setItem(i, 3, new QTableWidgetItem(QString::number(entry.capital_before, 'f', 2)));
        m_tradeLogTable->setItem(i, 4, new QTableWidgetItem(QString::number(entry.capital_after, 'f', 2)));
    }

    m_tradeLogTable->resizeRowsToContents();
}

void FeaturePreviewDialog::showMLClassificationResults(const MLResults& results) {
    QString metrics = FeaturePreviewUtils::formatPortfolioResults(results.portfolioResult, false);
    m_metricsLabel->setText(metrics);
    m_importancesLabel->setText("");
    displayTradeLog(results.trade_log);
}

void FeaturePreviewDialog::showMLRegressionResults(const MLResults& results) {
    QString metrics = FeaturePreviewUtils::formatPortfolioResults(results.portfolioResult, true);
    m_metricsLabel->setText(metrics);
    m_importancesLabel->setText("");
    displayTradeLog(results.trade_log);
    QString debug = FeaturePreviewUtils::formatSampleTradingDecisions(results.predictions, true, 10);
    m_debugInfoLabel->setText(debug);
}
