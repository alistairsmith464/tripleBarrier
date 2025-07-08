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
#include <numeric>
#include <algorithm>

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
    
    // Add debug info about barriers
    m_debugInfoLabel = new QLabel(this);
    if (!labeledEvents.empty()) {
        // Calculate some diagnostic statistics
        int profit_hits = 0, stop_hits = 0, time_hits = 0;
        double avg_volatility = 0.0;
        double max_volatility = 0.0, min_volatility = 1e9;
        
        for (const auto& event : labeledEvents) {
            if (event.label == 1) profit_hits++;
            else if (event.label == -1) stop_hits++;
            else time_hits++;
        }
        
        // Find corresponding volatility values and add detailed barrier analysis
        std::vector<double> entry_prices, profit_barriers, stop_barriers;
        for (const auto& event : labeledEvents) {
            auto it = std::find_if(rows.begin(), rows.end(), 
                [&](const PreprocessedRow& r) { return r.timestamp == event.entry_time; });
            if (it != rows.end()) {
                avg_volatility += it->volatility;
                max_volatility = std::max(max_volatility, it->volatility);
                min_volatility = std::min(min_volatility, it->volatility);
                
                // Calculate estimated barrier levels based on event outcomes
                // Since we don't have the original config, we'll estimate from the actual exit prices
                double entry_price = it->price;
                double exit_price = event.exit_price;
                double price_move = std::abs(exit_price - entry_price);
                double volatility = it->volatility;
                
                // Estimate the multiplier that would have been used
                double estimated_multiple = volatility > 0 ? price_move / volatility : 0.0;
                
                // Use the estimated multiple to show what the barriers would have been
                double profit_barrier = entry_price + estimated_multiple * volatility;
                double stop_barrier = entry_price - estimated_multiple * volatility;
                
                entry_prices.push_back(entry_price);
                profit_barriers.push_back(profit_barrier);
                stop_barriers.push_back(stop_barrier);
            }
        }
        avg_volatility /= labeledEvents.size();
        
        // Calculate barrier statistics
        QString barrier_stats = "";
        if (!entry_prices.empty()) {
            double avg_entry = std::accumulate(entry_prices.begin(), entry_prices.end(), 0.0) / entry_prices.size();
            double avg_profit_barrier = std::accumulate(profit_barriers.begin(), profit_barriers.end(), 0.0) / profit_barriers.size();
            double avg_stop_barrier = std::accumulate(stop_barriers.begin(), stop_barriers.end(), 0.0) / stop_barriers.size();
            
            double barrier_width_pct = ((avg_profit_barrier - avg_stop_barrier) / avg_entry) * 100.0;
            double profit_distance_pct = ((avg_profit_barrier - avg_entry) / avg_entry) * 100.0;
            double stop_distance_pct = ((avg_entry - avg_stop_barrier) / avg_entry) * 100.0;
            
            barrier_stats = QString("<br>Avg barriers: Entry=%1, Profit=%2 (+%3%%), Stop=%4 (-%5%%)<br>"
                                  "Total barrier width: %6%% of entry price")
                           .arg(avg_entry, 0, 'f', 4)
                           .arg(avg_profit_barrier, 0, 'f', 4)
                           .arg(profit_distance_pct, 0, 'f', 2)
                           .arg(avg_stop_barrier, 0, 'f', 4)
                           .arg(stop_distance_pct, 0, 'f', 2)
                           .arg(barrier_width_pct, 0, 'f', 2);
        }
        
        // Calculate timing statistics
        std::vector<int> profit_times, stop_times, time_times;
        for (const auto& event : labeledEvents) {
            if (event.label == 1) profit_times.push_back(event.periods_to_exit);
            else if (event.label == -1) stop_times.push_back(event.periods_to_exit);
            else if (event.label == 0) time_times.push_back(event.periods_to_exit);
        }
        
        auto calc_avg = [](const std::vector<int>& vec) -> double {
            return vec.empty() ? 0.0 : std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
        };
        
        QString timing_stats = QString("<br>Avg periods to exit: Profit=%1, Stop=%2, Time=%3")
                              .arg(calc_avg(profit_times), 0, 'f', 1)
                              .arg(calc_avg(stop_times), 0, 'f', 1)
                              .arg(calc_avg(time_times), 0, 'f', 1);
        
        m_debugInfoLabel->setText(QString("<b>Barrier Diagnostics:</b><br>"
                                        "Profit hits: %1, Stop hits: %2, Time hits: %3<br>"
                                        "Avg volatility: %4, Min: %5, Max: %6%7%8")
                                .arg(profit_hits).arg(stop_hits).arg(time_hits)
                                .arg(avg_volatility, 0, 'e', 3)
                                .arg(min_volatility, 0, 'e', 3)
                                .arg(max_volatility, 0, 'e', 3)
                                .arg(barrier_stats)
                                .arg(timing_stats));
    } else {
        m_debugInfoLabel->setText("<b>Barrier Diagnostics:</b> No events to analyze");
    }
    m_debugInfoLabel->setStyleSheet("color: #8e44ad; font-size: 11px; padding: 5px;");
    vbox->addWidget(m_debugInfoLabel);
    
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



void FeaturePreviewDialog::onRunMLClicked() {
    MLHyperparamsDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted) return;
    
    // Configure ML pipeline for hard labeling only
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
    config.objective = "binary:logistic";  // Only binary classification for hard labels
    
    bool tune = m_tuneHyperparamsCheckBox && m_tuneHyperparamsCheckBox->isChecked();
    
    QString model_info = QString("<b>Model Selection:</b> %1 objective, %2 hyperparameter tuning<br>")
                        .arg(QString::fromStdString(config.objective))
                        .arg(tune ? "with" : "without");
    
    // Show label distribution analysis
    int profit_labels = 0, stop_labels = 0, neutral_labels = 0;
    double mean_returns = 0.0;
    
    for (const auto& e : m_labeledEvents) {
        mean_returns += (e.exit_price - e.entry_price);
        if (e.label == 1) profit_labels++;
        else if (e.label == -1) stop_labels++;
        else neutral_labels++;
    }
    mean_returns /= m_labeledEvents.size();
    
    QString label_analysis = QString("Hard Labels: Profit=%1, Stop=%2, Neutral=%3<br>"
                                   "Average Return: %4")
                            .arg(profit_labels).arg(stop_labels).arg(neutral_labels)
                            .arg(mean_returns, 0, 'f', 6);
    
    m_metricsLabel->setText(model_info + "<b>Signal Analysis:</b> " + label_analysis);
    
    // Run ML pipeline
    extractFeaturesAndLabels(m_selectedFeatures, m_rows, m_labeledEvents, m_features, m_labels, m_returns);
    MLPipeline::PipelineResult result;
    if (tune) {
        result = MLPipeline::runPipelineWithTuning(m_features, m_labels, m_returns, config);
    } else {
        result = MLPipeline::runPipeline(m_features, m_labels, m_returns, config);
    }
    
    // Show results
    showMLResults(result);
    showPortfolioSimulation(m_labels, m_returns);
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

void FeaturePreviewDialog::showPortfolioSimulation(const std::vector<int>& labels, const std::vector<double>& returns) {
    if (labels.size() != returns.size() || labels.empty()) {
        return;
    }
    
    // Portfolio simulation with hard labels (full position or no position)
    double total_pnl = 0.0;
    int long_positions = 0, short_positions = 0, no_positions = 0;
    std::vector<double> position_pnls;
    
    for (size_t i = 0; i < labels.size(); ++i) {
        int label = labels[i];
        
        // Skip neutral signals
        if (label == 0) {
            no_positions++;
            continue;
        }
        
        // Direction: 1 = long, -1 = short
        double direction = static_cast<double>(label);
        
        // Position P&L = direction * return (full position)
        double position_pnl = direction * returns[i];
        
        total_pnl += position_pnl;
        position_pnls.push_back(position_pnl);
        
        if (label == 1) long_positions++;
        else if (label == -1) short_positions++;
    }
    
    // Calculate portfolio metrics
    int total_trades = long_positions + short_positions;
    double avg_pnl = total_trades > 0 ? total_pnl / total_trades : 0.0;
    
    // Calculate volatility of P&L
    double pnl_variance = 0.0;
    for (double pnl : position_pnls) {
        pnl_variance += (pnl - avg_pnl) * (pnl - avg_pnl);
    }
    double pnl_volatility = total_trades > 1 ? std::sqrt(pnl_variance / (total_trades - 1)) : 0.0;
    
    // Sharpe ratio (assuming 0 risk-free rate)
    double sharpe_ratio = pnl_volatility > 0 ? avg_pnl / pnl_volatility : 0.0;
    
    // Hit ratio (% of profitable trades)
    int profitable_trades = 0;
    for (double pnl : position_pnls) {
        if (pnl > 0) profitable_trades++;
    }
    double hit_ratio = total_trades > 0 ? double(profitable_trades) / total_trades : 0.0;
    
    // Max drawdown calculation
    double peak_pnl = 0.0;
    double current_pnl = 0.0;
    double max_drawdown = 0.0;
    for (double pnl : position_pnls) {
        current_pnl += pnl;
        peak_pnl = std::max(peak_pnl, current_pnl);
        double drawdown = peak_pnl - current_pnl;
        max_drawdown = std::max(max_drawdown, drawdown);
    }
    
    QString portfolio_results = QString(
        "<br><br><b>Portfolio Simulation (Hard Labels):</b><br>"
        "Total P&L: %1 | Avg per trade: %2<br>"
        "Positions: Long=%3, Short=%4, Skipped=%5<br>"
        "Hit ratio: %6% | Sharpe ratio: %7<br>"
        "Max drawdown: %8 | P&L volatility: %9"
    ).arg(total_pnl, 0, 'f', 6)
     .arg(avg_pnl, 0, 'f', 6)
     .arg(long_positions)
     .arg(short_positions)
     .arg(no_positions)
     .arg(hit_ratio * 100.0, 0, 'f', 1)
     .arg(sharpe_ratio, 0, 'f', 3)
     .arg(max_drawdown, 0, 'f', 6)
     .arg(pnl_volatility, 0, 'f', 6);
    
    // Append to existing metrics display
    QString current_text = m_metricsLabel->text();
    m_metricsLabel->setText(current_text + portfolio_results);
}
