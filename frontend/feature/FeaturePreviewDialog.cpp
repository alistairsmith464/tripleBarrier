#include "FeaturePreviewDialog.h"
#include "../MLHyperparamsDialog.h"
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QDialogButtonBox>
#include <QPushButton>
#include "../utils/DialogUtils.h"
#include "../backend/data/FeatureCalculator.h"
#include "../backend/data/DataCleaningUtils.h"
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

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
    
    m_dataInfoLabel = new QLabel(this);
    m_dataInfoLabel->setText(QString("<b>Data Summary:</b> %1 price records, %2 labeled events")
                            .arg(rows.size()).arg(labeledEvents.size()));
    m_dataInfoLabel->setStyleSheet("color: #2c3e50; font-size: 12px; padding: 5px;");
    vbox->addWidget(m_dataInfoLabel);
    
    m_debugInfoLabel = new QLabel(this);
    if (!labeledEvents.empty()) {
        int profit_hits = 0, stop_hits = 0, time_hits = 0;
        double avg_volatility = 0.0;
        double max_volatility = 0.0, min_volatility = 1e9;
        
        for (const auto& event : labeledEvents) {
            if (event.label == 1) profit_hits++;
            else if (event.label == -1) stop_hits++;
            else time_hits++;
        }
        
        std::vector<double> entry_prices, profit_barriers, stop_barriers;
        for (const auto& event : labeledEvents) {
            auto it = std::find_if(rows.begin(), rows.end(), 
                [&](const PreprocessedRow& r) { return r.timestamp == event.entry_time; });
            if (it != rows.end()) {
                avg_volatility += it->volatility;
                max_volatility = std::max(max_volatility, it->volatility);
                min_volatility = std::min(min_volatility, it->volatility);
                
                double entry_price = it->price;
                double exit_price = event.exit_price;
                double price_move = std::abs(exit_price - entry_price);
                double volatility = it->volatility;
                
                double estimated_multiple = volatility > 0 ? price_move / volatility : 0.0;
                
                double profit_barrier = entry_price + estimated_multiple * volatility;
                double stop_barrier = entry_price - estimated_multiple * volatility;
                
                entry_prices.push_back(entry_price);
                profit_barriers.push_back(profit_barrier);
                stop_barriers.push_back(stop_barrier);
            }
        }
        avg_volatility /= labeledEvents.size();
        
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
    
    m_runMLButton = new QPushButton("Run ML Pipeline", this);
    vbox->addWidget(m_runMLButton);
    m_metricsLabel = new QLabel(this);
    m_importancesLabel = new QLabel(this);
    vbox->addWidget(m_metricsLabel);
    vbox->addWidget(m_importancesLabel);
    m_tuneHyperparamsCheckBox = new QCheckBox("Auto-tune hyperparameters (grid search)", this);
    m_tuneHyperparamsCheckBox->setToolTip("If checked, the pipeline will automatically search for the best hyperparameters (n_rounds, max_depth, nthread).");
    vbox->addWidget(m_tuneHyperparamsCheckBox);
    connect(m_runMLButton, &QPushButton::clicked, this, &FeaturePreviewDialog::onRunMLClicked);

    this->m_selectedFeatures = selectedFeatures;
    this->m_rows = rows;
    this->m_labeledEvents = labeledEvents;
    QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok, this);
    connect(box, &QDialogButtonBox::accepted, this, &QDialog::accept);
    vbox->addWidget(box);
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

void FeaturePreviewDialog::extractFeaturesAndLabelsRegression(const QSet<QString>& selectedFeatures,
                                 const std::vector<PreprocessedRow>& rows,
                                 const std::vector<LabeledEvent>& labeledEvents,
                                 std::vector<std::map<std::string, double>>& features,
                                 std::vector<double>& labels,
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
        auto base_features = FeatureCalculator::calculateFeatures(prices, timestamps, eventIndices, int(i), backendFeatures);
        
        std::map<std::string, double> enhanced_features = base_features;
        
        if (i < rows.size()) {
            const auto& row = rows[eventIndices[i]];
            if (row.volume.has_value()) {
                double volume = row.volume.value();
                enhanced_features["volume"] = volume;
                
                if (base_features.count(FeatureCalculator::RETURN_5D)) {
                    enhanced_features["volume_return_5d"] = volume * base_features[FeatureCalculator::RETURN_5D];
                }
                
                if (base_features.count(FeatureCalculator::ROLLING_STD_5D)) {
                    enhanced_features["volume_vol_5d"] = volume * base_features[FeatureCalculator::ROLLING_STD_5D];
                }
            }
        }
        
        if (base_features.count(FeatureCalculator::RETURN_5D) && base_features.count(FeatureCalculator::ROLLING_STD_5D)) {
            double return_5d = base_features[FeatureCalculator::RETURN_5D];
            double vol_5d = base_features[FeatureCalculator::ROLLING_STD_5D];
            if (vol_5d > 1e-10) {
                enhanced_features["volatility_adjusted_return_5d"] = return_5d / vol_5d;
            }
        }
        
        if (base_features.count(FeatureCalculator::ROC_5D) && base_features.count(FeatureCalculator::EWMA_VOL_10D)) {
            double roc_5d = base_features[FeatureCalculator::ROC_5D];
            double vol_10d = base_features[FeatureCalculator::EWMA_VOL_10D];
            enhanced_features["momentum_vol_ratio"] = roc_5d * vol_10d;
        }
        
        if (base_features.count(FeatureCalculator::DIST_TO_SMA_5D) && base_features.count(FeatureCalculator::ROLLING_STD_5D)) {
            double dist_sma = base_features[FeatureCalculator::DIST_TO_SMA_5D];
            double vol_5d = base_features[FeatureCalculator::ROLLING_STD_5D];
            if (vol_5d > 1e-10) {
                enhanced_features["sma_distance_vol_adj"] = dist_sma / vol_5d;
            }
        }
        
        if (base_features.count(FeatureCalculator::RSI_14D) && base_features.count(FeatureCalculator::RETURN_5D)) {
            double rsi = base_features[FeatureCalculator::RSI_14D];
            double return_5d = base_features[FeatureCalculator::RETURN_5D];
            enhanced_features["rsi_momentum"] = (rsi - 50.0) * return_5d;
        }
        
        features.push_back(enhanced_features);
        
        labels.push_back(labeledEvents[i].ttbm_label);
        returns.push_back(labeledEvents[i].exit_price - labeledEvents[i].entry_price);
    }
    
    for (auto& feature_row : features) {
        for (auto& kv : feature_row) {
            if (std::isnan(kv.second) || std::isinf(kv.second)) {
                kv.second = 0.0;
            }
        }
    }
    
    if (!labels.empty()) {
        double min_label = *std::min_element(labels.begin(), labels.end());
        double max_label = *std::max_element(labels.begin(), labels.end());
        double mean_label = std::accumulate(labels.begin(), labels.end(), 0.0) / labels.size();
        
        int zero_count = 0;
        int positive_count = 0;
        int negative_count = 0;
        for (double label : labels) {
            if (std::abs(label) < 0.01) zero_count++;
            else if (label > 0) positive_count++;
            else negative_count++;
        }
        
        std::cout << "[DEBUG] Regression: Predicting TTBM labels" << std::endl;
        std::cout << "  Sample size: " << labels.size() << std::endl;
        std::cout << "  Range: [" << min_label << ", " << max_label << "], Mean: " << mean_label << std::endl;
        std::cout << "  Positive: " << positive_count << " (" << (100.0*positive_count/labels.size()) << "%), "
                  << "Negative: " << negative_count << " (" << (100.0*negative_count/labels.size()) << "%), "
                  << "Zero: " << zero_count << " (" << (100.0*zero_count/labels.size()) << "%)" << std::endl;
    }
    
    if (!features.empty()) {
        std::map<std::string, double> feature_medians;
        std::map<std::string, double> feature_iqrs;
        
        for (const auto& feature_row : features) {
            for (const auto& kv : feature_row) {
                if (feature_medians.find(kv.first) == feature_medians.end()) {
                    feature_medians[kv.first] = 0.0;
                    feature_iqrs[kv.first] = 0.0;
                }
            }
        }
        
        for (const auto& kv : feature_medians) {
            std::vector<double> values;
            for (const auto& feature_row : features) {
                auto it = feature_row.find(kv.first);
                if (it != feature_row.end()) {
                    values.push_back(it->second);
                }
            }
            
            if (!values.empty()) {
                std::sort(values.begin(), values.end());
                size_t n = values.size();
                
                if (n % 2 == 0) {
                    feature_medians[kv.first] = (values[n/2-1] + values[n/2]) / 2.0;
                } else {
                    feature_medians[kv.first] = values[n/2];
                }
                
                double q1 = values[n/4];
                double q3 = values[3*n/4];
                feature_iqrs[kv.first] = q3 - q1;
                if (feature_iqrs[kv.first] < 1e-10) feature_iqrs[kv.first] = 1.0;
            }
        }
        
        for (auto& feature_row : features) {
            for (auto& kv : feature_row) {
                kv.second = (kv.second - feature_medians[kv.first]) / feature_iqrs[kv.first];
            }
        }
        
        std::cout << "[DEBUG] Applied robust scaling (median/IQR)" << std::endl;
    }
}

void FeaturePreviewDialog::onRunMLClicked() {
    MLHyperparamsDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted) return;
    
    bool is_ttbm = false;
    if (!m_labeledEvents.empty()) {
        is_ttbm = m_labeledEvents[0].is_ttbm;
    }
    
    bool tune = m_tuneHyperparamsCheckBox && m_tuneHyperparamsCheckBox->isChecked();
    
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
    
    QString model_type = is_ttbm ? "TTBM Regression" : "Hard Barrier Classification";
    QString model_info = QString("<b>Model Type:</b> %1 (%2)<br><b>Hyperparameter Tuning:</b> %3<br>")
                        .arg(model_type)
                        .arg(QString::fromStdString(config.objective))
                        .arg(tune ? "enabled" : "disabled");
    
    if (is_ttbm) {
        double min_label = 1.0, max_label = -1.0, mean_label = 0.0;
        int zero_labels = 0, positive_count = 0, negative_count = 0;
        
        for (const auto& e : m_labeledEvents) {
            double label = e.ttbm_label;
            min_label = std::min(min_label, label);
            max_label = std::max(max_label, label);
            mean_label += label;
            if (std::abs(label) < 0.01) zero_labels++;
            else if (label > 0) positive_count++;
            else negative_count++;
        }
        mean_label /= m_labeledEvents.size();
        
        QString label_analysis = QString("TTBM Regression: Predicting Directional Confidence<br>"
                                       "Target: TTBM label (directional confidence from -1 to +1)<br>"
                                       "Range: [%1, %2], Mean: %3<br>"
                                       "Zero/Neutral: %4/%5 (%6%)<br>"
                                       "Positive Labels: %7/%8 (%9%)<br>"
                                       "Negative Labels: %10/%11 (%12%)<br>"
                                       "<b>Portfolio simulation scales bet size by signal strength</b><br>"
                                       "<b>Position size = |prediction| * 3% of portfolio (max 3%)</b>")
                                .arg(min_label, 0, 'f', 4)
                                .arg(max_label, 0, 'f', 4)
                                .arg(mean_label, 0, 'f', 4)
                                .arg(zero_labels).arg(m_labeledEvents.size())
                                .arg(100.0 * zero_labels / m_labeledEvents.size(), 0, 'f', 1)
                                .arg(positive_count).arg(m_labeledEvents.size())
                                .arg(100.0 * positive_count / m_labeledEvents.size(), 0, 'f', 1)
                                .arg(negative_count).arg(m_labeledEvents.size())
                                .arg(100.0 * negative_count / m_labeledEvents.size(), 0, 'f', 1);
        
        m_dataInfoLabel->setText(model_info + label_analysis);
    } else {
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
        
        m_dataInfoLabel->setText(model_info + label_analysis);
    }
    
    if (is_ttbm) {
        extractFeaturesAndLabelsRegression(m_selectedFeatures, m_rows, m_labeledEvents, m_features, m_labels_double, m_returns);
        
        MLPipeline::RegressionPipelineResult result;
        if (tune) {
            result = MLPipeline::runPipelineRegressionWithTuning(m_features, m_labels_double, m_returns, config);
        } else {
            result = MLPipeline::runPipelineRegression(m_features, m_labels_double, m_returns, config);
        }
        
        showMLRegressionResults(result);
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
    std::vector<double> double_predictions;
    for (int pred : result.predictions) {
        double_predictions.push_back(static_cast<double>(pred));
    }
    
    auto portfolioResults = runPortfolioSimulation(double_predictions, m_labeledEvents, false);
    
    QString metrics = QString("<b>Hard Barrier Classification - Portfolio Simulation:</b><br><br>"
                            "<b>Trading Strategy:</b><br>"
                            "• Signal +1: Long position (2%% of portfolio)<br>"
                            "• Signal -1: Short position (2%% of portfolio)<br>"
                            "• Signal 0: No position<br><br>"
                            
                            "<b>Portfolio Performance:</b><br>"
                            "Starting Capital: $%1<br>"
                            "Final Portfolio Value: $%2<br>"
                            "Total Return: %3%%<br>"
                            "Annualized Return: %4%%<br>"
                            "Maximum Drawdown: %5%%<br>"
                            "Sharpe Ratio: %6<br><br>"
                            
                            "<b>Trading Statistics:</b><br>"
                            "Total Trades: %7<br>"
                            "Winning Trades: %8 (%9%%)<br>"
                            "Losing Trades: %10 (%11%%)<br>"
                            "Average Trade Return: %12%%<br>"
                            "Best Trade: %13%%<br>"
                            "Worst Trade: %14%%")
                            .arg(portfolioResults.starting_capital, 0, 'f', 0)
                            .arg(portfolioResults.final_value, 0, 'f', 0)
                            .arg(portfolioResults.total_return * 100, 0, 'f', 2)
                            .arg(portfolioResults.annualized_return * 100, 0, 'f', 2)
                            .arg(portfolioResults.max_drawdown * 100, 0, 'f', 2)
                            .arg(portfolioResults.sharpe_ratio, 0, 'f', 3)
                            .arg(portfolioResults.total_trades)
                            .arg(portfolioResults.winning_trades)
                            .arg(portfolioResults.win_rate * 100, 0, 'f', 1)
                            .arg(portfolioResults.losing_trades)
                            .arg((100.0 - portfolioResults.win_rate * 100), 0, 'f', 1)
                            .arg(portfolioResults.avg_trade_return * 100, 0, 'f', 3)
                            .arg(portfolioResults.best_trade * 100, 0, 'f', 2)
                            .arg(portfolioResults.worst_trade * 100, 0, 'f', 2);
    
    m_metricsLabel->setText(metrics);
    m_importancesLabel->setText("");
}

void FeaturePreviewDialog::showMLRegressionResults(const MLPipeline::RegressionPipelineResult& result) {
    auto portfolioResults = runPortfolioSimulation(result.predictions, m_labeledEvents, true);
    
    QString title = "<b>TTBM Regression - Portfolio Simulation:</b><br><br>";
    QString strategy_description = "<b>Trading Strategy:</b><br>"
                                  "• Position size = Signal strength × 3%% of portfolio<br>"
                                  "• Positive signal: Long position<br>"
                                  "• Negative signal: Short position<br>"
                                  "• Signal near zero: Small/no position<br><br>";
    
    QString metrics = QString("%1%2"
                            "<b>Portfolio Performance:</b><br>"
                            "Starting Capital: $%3<br>"
                            "Final Portfolio Value: $%4<br>"
                            "Total Return: %5%%<br>"
                            "Annualized Return: %6%%<br>"
                            "Maximum Drawdown: %7%%<br>"
                            "Sharpe Ratio: %8<br><br>"
                            
                            "<b>Trading Statistics:</b><br>"
                            "Total Trades: %9<br>"
                            "Winning Trades: %10 (%11%%)<br>"
                            "Losing Trades: %12 (%13%%)<br>"
                            "Average Trade Return: %14%%<br>"
                            "Best Trade: %15%%<br>"
                            "Worst Trade: %16%%")
                            .arg(title)
                            .arg(strategy_description)
                            .arg(portfolioResults.starting_capital, 0, 'f', 0)
                            .arg(portfolioResults.final_value, 0, 'f', 0)
                            .arg(portfolioResults.total_return * 100, 0, 'f', 2)
                            .arg(portfolioResults.annualized_return * 100, 0, 'f', 2)
                            .arg(portfolioResults.max_drawdown * 100, 0, 'f', 2)
                            .arg(portfolioResults.sharpe_ratio, 0, 'f', 3)
                            .arg(portfolioResults.total_trades)
                            .arg(portfolioResults.winning_trades)
                            .arg(portfolioResults.win_rate * 100, 0, 'f', 1)
                            .arg(portfolioResults.losing_trades)
                            .arg((100.0 - portfolioResults.win_rate * 100), 0, 'f', 1)
                            .arg(portfolioResults.avg_trade_return * 100, 0, 'f', 3)
                            .arg(portfolioResults.best_trade * 100, 0, 'f', 2)
                            .arg(portfolioResults.worst_trade * 100, 0, 'f', 2);
    
    m_metricsLabel->setText(metrics);
    m_importancesLabel->setText("");
    
    QString debug = "<b>Sample Trading Decisions (first 10):</b><br>";
    for (size_t i = 0; i < std::min(size_t(10), result.predictions.size()); ++i) {
        double pred = result.predictions[i];
        
        double position_size = std::abs(pred) * 3.0;
        QString direction = pred > 0 ? "LONG" : "SHORT";
        
        QString trade_info;
        if (std::abs(pred) < 0.1) {
            trade_info = QString("Sample %1: Signal=%2 → NO TRADE (signal too weak)")
                        .arg(i + 1).arg(pred, 0, 'f', 4);
        } else {
            trade_info = QString("Sample %1: Signal=%2 → %3 %4%% position")
                        .arg(i + 1).arg(pred, 0, 'f', 4).arg(direction).arg(position_size, 0, 'f', 2);
        }
        
        debug += trade_info + "<br>";
    }
    m_debugInfoLabel->setText(debug);
}

PortfolioResults FeaturePreviewDialog::runPortfolioSimulation(const std::vector<double>& predictions,
                                                             const std::vector<LabeledEvent>& events,
                                                             bool is_ttbm) {
    PortfolioResults results;
    double portfolio_value = results.starting_capital;
    results.portfolio_values.push_back(portfolio_value);
    
    for (size_t i = 0; i < predictions.size() && i < events.size(); ++i) {
        double prediction = predictions[i];
        double actual_return = (events[i].exit_price - events[i].entry_price) / events[i].entry_price;
        
        double position_size = 0.0;
        if (is_ttbm) {
            double signal_strength = std::abs(prediction);
            if (signal_strength > 0.1) {
                position_size = std::min(signal_strength * 0.03, 0.03);
                if (prediction < 0) position_size = -position_size;
            }
        } else {
            if (std::abs(prediction - 1.0) < 0.1) {
                position_size = 0.02;
            } else if (std::abs(prediction + 1.0) < 0.1) {
                position_size = -0.02;
            }
        }
        
        double trade_return = position_size * actual_return;
        
        portfolio_value *= (1.0 + trade_return);
        results.portfolio_values.push_back(portfolio_value);
        
        if (std::abs(position_size) > 0.001) {
            results.total_trades++;
            results.trade_returns.push_back(trade_return);
            
            if (trade_return > 0) {
                results.winning_trades++;
            } else {
                results.losing_trades++;
            }
            
            results.best_trade = std::max(results.best_trade, trade_return);
            results.worst_trade = std::min(results.worst_trade, trade_return);
        }
    }
    
    results.final_value = portfolio_value;
    results.total_return = (results.final_value - results.starting_capital) / results.starting_capital;
    
    double periods = static_cast<double>(events.size());
    if (periods > 0) {
        results.annualized_return = std::pow(results.final_value / results.starting_capital, 252.0 / periods) - 1.0;
    }
    
    results.max_drawdown = calculateMaxDrawdown(results.portfolio_values);
    results.sharpe_ratio = calculateSharpeRatio(results.trade_returns);
    
    if (results.total_trades > 0) {
        results.win_rate = static_cast<double>(results.winning_trades) / results.total_trades;
        results.avg_trade_return = std::accumulate(results.trade_returns.begin(), results.trade_returns.end(), 0.0) / results.total_trades;
    }
    
    return results;
}

double FeaturePreviewDialog::calculateSharpeRatio(const std::vector<double>& returns) {
    if (returns.empty()) return 0.0;
    
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();
    
    double std_dev = std::sqrt(variance);
    if (std_dev < 1e-10) return 0.0;
    
    return (mean_return * 252.0) / (std_dev * std::sqrt(252.0));
}

double FeaturePreviewDialog::calculateMaxDrawdown(const std::vector<double>& portfolio_values) {
    if (portfolio_values.empty()) return 0.0;
    
    double max_drawdown = 0.0;
    double peak = portfolio_values[0];
    
    for (double value : portfolio_values) {
        if (value > peak) {
            peak = value;
        } else {
            double drawdown = (peak - value) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
    }
    
    return max_drawdown;
}
