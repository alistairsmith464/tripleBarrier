#include "FeaturePreviewDialog.h"
#include "../MLHyperparamsDialog.h"
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QFileDialog>
#include "../utils/CSVExportUtils.h"
#include "../utils/DialogUtils.h"
#include "../backend/data/FeatureCalculator.h"
#include "../backend/data/DataCleaningUtils.h"
#include <map>
#include <set>
#include <cmath>

// Forward declaration for cleanFeatureTable
static void cleanFeatureTable(QTableWidget* table);

FeaturePreviewDialog::FeaturePreviewDialog(const QSet<QString>& selectedFeatures,
                                           const std::vector<PreprocessedRow>& rows,
                                           const std::vector<LabeledEvent>& labeledEvents,
                                           QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle("Feature Preview");
    QVBoxLayout* vbox = new QVBoxLayout(this);
    QMap<QString, std::string> featureMap = {
        {"Close-to-close return for the previous day", FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D},
        {"Return over the past 5 days", FeatureCalculator::RETURN_5D},
        {"Return over the past 10 days", FeatureCalculator::RETURN_10D},
        {"Rolling standard deviation of daily returns over the last 5 days", FeatureCalculator::ROLLING_STD_5D},
        {"EWMA volatility over 10 days", FeatureCalculator::EWMA_VOL_10D},
        {"5-day simple moving average (SMA)", FeatureCalculator::SMA_5D},
        {"10-day SMA", FeatureCalculator::SMA_10D},
        {"20-day SMA", FeatureCalculator::SMA_20D},
        {"Distance between current close price and 5-day SMA", FeatureCalculator::DIST_TO_SMA_5D},
        {"Rate of Change (ROC) over 5 days", FeatureCalculator::ROC_5D},
        {"Relative Strength Index (RSI) over 14 days", FeatureCalculator::RSI_14D},
        {"5-day high minus 5-day low (price range)", FeatureCalculator::PRICE_RANGE_5D},
        {"Current close price relative to 5-day high", FeatureCalculator::CLOSE_OVER_HIGH_5D},
        {"Slope of linear regression of close prices over 10 days", FeatureCalculator::SLOPE_LR_10D},
        {"Day of the week", FeatureCalculator::DAY_OF_WEEK},
        {"Days since last event", FeatureCalculator::DAYS_SINCE_LAST_EVENT}
    };
    std::set<std::string> backendFeatures;
    for (const QString& feat : selectedFeatures) {
        if (featureMap.contains(feat)) backendFeatures.insert(featureMap[feat]);
    }
    std::vector<double> prices;
    std::vector<std::string> timestamps;
    std::vector<int> eventIndices;
    for (size_t i = 0; i < rows.size(); ++i) {
        prices.push_back(rows[i].price);
        timestamps.push_back(rows[i].timestamp);
    }
    for (const auto& e : labeledEvents) {
        auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
        if (it != rows.end()) eventIndices.push_back(int(std::distance(rows.begin(), it)));
    }
    std::vector<std::map<std::string, double>> allFeatures;
    for (size_t i = 0; i < eventIndices.size(); ++i) {
        allFeatures.push_back(FeatureCalculator::calculateFeatures(prices, timestamps, eventIndices, int(i), backendFeatures));
    }
    DataCleaningUtils::cleanFeatureRows(allFeatures);
    QTableWidget* table = new QTableWidget(int(allFeatures.size()), int(backendFeatures.size()), this);
    QStringList headers;
    for (const QString& feat : selectedFeatures) headers << feat;
    table->setHorizontalHeaderLabels(headers);
    int col = 0;
    for (const QString& feat : selectedFeatures) {
        std::string backendId = featureMap[feat];
        for (int row = 0; row < int(allFeatures.size()); ++row) {
            double val = allFeatures[row].count(backendId) ? allFeatures[row][backendId] : NAN;
            table->setItem(row, col, new QTableWidgetItem(QString::number(val)));
        }
        ++col;
    }
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    vbox->addWidget(table);
    
    // Add data info label
    m_dataInfoLabel = new QLabel(this);
    m_dataInfoLabel->setText(QString("<b>Data Summary:</b> %1 price records, %2 labeled events")
                            .arg(rows.size()).arg(labeledEvents.size()));
    m_dataInfoLabel->setStyleSheet("color: #2c3e50; font-size: 12px; padding: 5px;");
    vbox->addWidget(m_dataInfoLabel);
    
    QPushButton* exportBtn = new QPushButton("Export to CSV", this);
    vbox->addWidget(exportBtn);
    m_runMLButton = new QPushButton("Run ML Pipeline", this);
    vbox->addWidget(m_runMLButton);
    m_metricsLabel = new QLabel(this);
    m_importancesLabel = new QLabel(this);
    vbox->addWidget(m_metricsLabel);
    vbox->addWidget(m_importancesLabel);
    m_tuneHyperparamsCheckBox = new QCheckBox("Auto-tune hyperparameters (grid search)", this);
    m_tuneHyperparamsCheckBox->setToolTip("If checked, the pipeline will automatically search for the best hyperparameters (n_rounds, max_depth, nthread).");
    vbox->addWidget(m_tuneHyperparamsCheckBox);
    connect(exportBtn, &QPushButton::clicked, this, [=]() {
        cleanFeatureTable(table);
        QString fileName = QFileDialog::getSaveFileName(this, "Export Features to CSV", "features_output.csv", "CSV Files (*.csv);;All Files (*.*)");
        if (fileName.isEmpty()) return;
        QFile file(fileName);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            DialogUtils::showError(this, "Export Error", "Could not open file for writing.");
            return;
        }
        QTextStream out(&file);
        QStringList headers;
        for (const QString& feat : selectedFeatures) headers << feat;
        out << headers.join(",") << "\n";
        for (int row = 0; row < table->rowCount(); ++row) {
            QStringList rowVals;
            for (int col = 0; col < table->columnCount(); ++col) {
                QTableWidgetItem* item = table->item(row, col);
                rowVals << (item ? item->text() : "");
            }
            out << rowVals.join(",") << "\n";
        }
        file.close();
        DialogUtils::showInfo(this, "Export Complete", "Features exported to " + fileName);
    });
    connect(m_runMLButton, &QPushButton::clicked, this, &FeaturePreviewDialog::onRunMLClicked);
    // Store for ML pipeline
    this->m_selectedFeatures = selectedFeatures;
    this->m_rows = rows;
    this->m_labeledEvents = labeledEvents;
    QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok, this);
    connect(box, &QDialogButtonBox::accepted, this, &QDialog::accept);
    vbox->addWidget(box);
}

static void cleanFeatureTable(QTableWidget* table) {
    for (int row = table->rowCount() - 1; row >= 0; --row) {
        bool valid = true;
        for (int col = 0; col < table->columnCount(); ++col) {
            QTableWidgetItem* item = table->item(row, col);
            if (!item) { valid = false; break; }
            bool ok = false;
            double val = item->text().toDouble(&ok);
            if (!ok || std::isnan(val) || std::isinf(val)) {
                valid = false;
                break;
            }
        }
        if (!valid) table->removeRow(row);
    }
}

void FeaturePreviewDialog::extractFeaturesAndLabels(const QSet<QString>& selectedFeatures,
                                 const std::vector<PreprocessedRow>& rows,
                                 const std::vector<LabeledEvent>& labeledEvents,
                                 std::vector<std::map<std::string, double>>& features,
                                 std::vector<int>& labels,
                                 std::vector<double>& returns) {
    QMap<QString, std::string> featureMap = {
        {"Close-to-close return for the previous day", FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D},
        {"Return over the past 5 days", FeatureCalculator::RETURN_5D},
        {"Return over the past 10 days", FeatureCalculator::RETURN_10D},
        {"Rolling standard deviation of daily returns over the last 5 days", FeatureCalculator::ROLLING_STD_5D},
        {"EWMA volatility over 10 days", FeatureCalculator::EWMA_VOL_10D},
        {"5-day simple moving average (SMA)", FeatureCalculator::SMA_5D},
        {"10-day SMA", FeatureCalculator::SMA_10D},
        {"20-day SMA", FeatureCalculator::SMA_20D},
        {"Distance between current close price and 5-day SMA", FeatureCalculator::DIST_TO_SMA_5D},
        {"Rate of Change (ROC) over 5 days", FeatureCalculator::ROC_5D},
        {"Relative Strength Index (RSI) over 14 days", FeatureCalculator::RSI_14D},
        {"5-day high minus 5-day low (price range)", FeatureCalculator::PRICE_RANGE_5D},
        {"Current close price relative to 5-day high", FeatureCalculator::CLOSE_OVER_HIGH_5D},
        {"Slope of linear regression of close prices over 10 days", FeatureCalculator::SLOPE_LR_10D},
        {"Day of the week", FeatureCalculator::DAY_OF_WEEK},
        {"Days since last event", FeatureCalculator::DAYS_SINCE_LAST_EVENT}
    };
    std::set<std::string> backendFeatures;
    for (const QString& feat : selectedFeatures) {
        if (featureMap.contains(feat)) backendFeatures.insert(featureMap[feat]);
    }
    std::vector<double> prices;
    std::vector<std::string> timestamps;
    std::vector<int> eventIndices;
    for (size_t i = 0; i < rows.size(); ++i) {
        prices.push_back(rows[i].price);
        timestamps.push_back(rows[i].timestamp);
    }
    for (const auto& e : labeledEvents) {
        auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
        if (it != rows.end()) eventIndices.push_back(int(std::distance(rows.begin(), it)));
    }
    features.clear();
    labels.clear();
    returns.clear();
    for (size_t i = 0; i < eventIndices.size(); ++i) {
        features.push_back(FeatureCalculator::calculateFeatures(prices, timestamps, eventIndices, int(i), backendFeatures));
        labels.push_back(labeledEvents[i].label);
        returns.push_back(labeledEvents[i].exit_price - labeledEvents[i].entry_price);
    }
}

void FeaturePreviewDialog::extractFeaturesAndLabelsSoft(const QSet<QString>& selectedFeatures,
                                 const std::vector<PreprocessedRow>& rows,
                                 const std::vector<LabeledEvent>& labeledEvents,
                                 std::vector<std::map<std::string, double>>& features,
                                 std::vector<double>& soft_labels,
                                 std::vector<double>& returns) {
    QMap<QString, std::string> featureMap = {
        {"Close-to-close return for the previous day", FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D},
        {"Return over the past 5 days", FeatureCalculator::RETURN_5D},
        {"Return over the past 10 days", FeatureCalculator::RETURN_10D},
        {"Rolling standard deviation of daily returns over the last 5 days", FeatureCalculator::ROLLING_STD_5D},
        {"EWMA volatility over 10 days", FeatureCalculator::EWMA_VOL_10D},
        {"5-day simple moving average (SMA)", FeatureCalculator::SMA_5D},
        {"10-day SMA", FeatureCalculator::SMA_10D},
        {"20-day SMA", FeatureCalculator::SMA_20D},
        {"Distance between current close price and 5-day SMA", FeatureCalculator::DIST_TO_SMA_5D},
        {"Rate of Change (ROC) over 5 days", FeatureCalculator::ROC_5D},
        {"Relative Strength Index (RSI) over 14 days", FeatureCalculator::RSI_14D},
        {"5-day high minus 5-day low (price range)", FeatureCalculator::PRICE_RANGE_5D},
        {"Current close price relative to 5-day high", FeatureCalculator::CLOSE_OVER_HIGH_5D},
        {"Slope of linear regression of close prices over 10 days", FeatureCalculator::SLOPE_LR_10D},
        {"Day of the week", FeatureCalculator::DAY_OF_WEEK},
        {"Days since last event", FeatureCalculator::DAYS_SINCE_LAST_EVENT}
    };
    std::set<std::string> backendFeatures;
    for (const QString& feat : selectedFeatures) {
        if (featureMap.contains(feat)) backendFeatures.insert(featureMap[feat]);
    }
    std::vector<double> prices;
    std::vector<std::string> timestamps;
    std::vector<int> eventIndices;
    for (size_t i = 0; i < rows.size(); ++i) {
        prices.push_back(rows[i].price);
        timestamps.push_back(rows[i].timestamp);
    }
    for (const auto& e : labeledEvents) {
        auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
        if (it != rows.end()) eventIndices.push_back(int(std::distance(rows.begin(), it)));
    }
    features.clear();
    soft_labels.clear();
    returns.clear();
    for (size_t i = 0; i < eventIndices.size(); ++i) {
        features.push_back(FeatureCalculator::calculateFeatures(prices, timestamps, eventIndices, int(i), backendFeatures));
        soft_labels.push_back(labeledEvents[i].soft_label);
        returns.push_back(labeledEvents[i].exit_price - labeledEvents[i].entry_price);
    }
}

void FeaturePreviewDialog::onRunMLClicked() {
    MLHyperparamsDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted) return;
    bool useSoft = false;
    for (const auto& e : m_labeledEvents) {
        if (std::abs(e.soft_label) > 1e-6 || e.soft_label != 0) {
            useSoft = true;
            break;
        }
    }
    MLPipeline::PipelineConfig config;
    config.split_type = MLPipeline::Chronological;
    config.train_ratio = 0.6;
    config.val_ratio = 0.2;
    config.test_ratio = 0.2;
    config.n_splits = 5;
    config.embargo = 0;
    config.random_seed = 42;
    config.n_rounds = dlg.nRounds();
    config.max_depth = dlg.maxDepth();
    config.nthread = dlg.nThread();
    config.objective = dlg.objective().toStdString();
    bool tune = m_tuneHyperparamsCheckBox && m_tuneHyperparamsCheckBox->isChecked();
    if (useSoft) {
        std::vector<std::map<std::string, double>> features;
        std::vector<double> soft_labels, returns;
        extractFeaturesAndLabelsSoft(m_selectedFeatures, m_rows, m_labeledEvents, features, soft_labels, returns);
        MLPipeline::PipelineResult result;
        if (tune) {
            result = MLPipeline::runPipelineSoftWithTuning(features, soft_labels, returns, config);
        } else {
            result = MLPipeline::runPipelineSoft(features, soft_labels, returns, config);
        }
        showMLResults(result);
    } else {
        extractFeaturesAndLabels(m_selectedFeatures, m_rows, m_labeledEvents, m_features, m_labels, m_returns);
        MLPipeline::PipelineResult result;
        if (tune) {
            result = MLPipeline::runPipelineWithTuning(m_features, m_labels, m_returns, config);
        } else {
            result = MLPipeline::runPipeline(m_features, m_labels, m_returns, config);
        }
        showMLResults(result);
    }
}

void FeaturePreviewDialog::showMLResults(const MLPipeline::PipelineResult& result) {
    QString metrics;
    if (result.metrics.f1 == 0 && result.metrics.true_positives == 0 && result.metrics.true_negatives == 0 && result.metrics.false_positives == 0 && result.metrics.false_negatives == 0) {
        metrics = QString("<b>Regression Metrics:</b><br>R²: %1<br>MSE: %2<br>MAE: %3<br>Total: %4"
            "<br><br><b>Financial Metrics:</b><br>Avg Return: %5<br>Sharpe Ratio: %6<br>Hit Ratio: %7")
            .arg(result.metrics.accuracy, 0, 'f', 3)
            .arg(result.metrics.precision, 0, 'f', 3)
            .arg(result.metrics.recall, 0, 'f', 3)
            .arg(result.metrics.total)
            .arg(result.metrics.avg_return, 0, 'f', 3)
            .arg(result.metrics.sharpe_ratio, 0, 'f', 3)
            .arg(result.metrics.hit_ratio, 0, 'f', 3);
    } else {
        metrics = QString("<b>Classification Metrics:</b><br>Accuracy: %1<br>Precision: %2<br>Recall: %3<br>F1: %4<br>"
            "<br><b>Counts:</b> TP: %5, TN: %6, FP: %7, FN: %8, Total: %9"
            "<br><br><b>Financial Metrics:</b><br>Avg Return: %10<br>Sharpe Ratio: %11<br>Hit Ratio: %12")
            .arg(result.metrics.accuracy, 0, 'f', 3)
            .arg(result.metrics.precision, 0, 'f', 3)
            .arg(result.metrics.recall, 0, 'f', 3)
            .arg(result.metrics.f1, 0, 'f', 3)
            .arg(result.metrics.true_positives)
            .arg(result.metrics.true_negatives)
            .arg(result.metrics.false_positives)
            .arg(result.metrics.false_negatives)
            .arg(result.metrics.total)
            .arg(result.metrics.avg_return, 0, 'f', 3)
            .arg(result.metrics.sharpe_ratio, 0, 'f', 3)
            .arg(result.metrics.hit_ratio, 0, 'f', 3);
    }
    m_metricsLabel->setText(metrics);
    QString importances = "<b>Feature Importances:</b><br>";
    for (const auto& kv : result.feature_importances) {
        importances += QString::fromStdString(kv.first) + ": " + QString::number(kv.second, 'f', 3) + "<br>";
    }
    m_importancesLabel->setText(importances);
}
