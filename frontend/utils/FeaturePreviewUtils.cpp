#include "FeaturePreviewUtils.h"
#include "../config/VisualizationConfig.h"
#include <numeric>
#include <algorithm>

std::set<std::string> FeaturePreviewUtils::convertQSetToStdSet(const QSet<QString>& qset) {
    std::set<std::string> result;
    for (const QString& item : qset) {
        result.insert(item.toStdString());
    }
    return result;
}

QString FeaturePreviewUtils::formatBarrierDiagnostics(
    const MLPipeline::BarrierDiagnostics& diagnostics,
    const std::vector<LabeledEvent>& labeledEvents) {
    if (labeledEvents.empty()) {
        return "<b>Barrier Diagnostics:</b> No events to analyze";
    }
    
    QString barrier_stats = "";
    if (diagnostics.avg_entry_price > 0) {
        barrier_stats = QString("<br>Avg barriers: Entry=%1, Profit=%2 (+%3%%), Stop=%4 (-%5%%)<br>"
                               "Total barrier width: %6%% of entry price")
                       .arg(diagnostics.avg_entry_price, 0, 'f', 4)
                       .arg(diagnostics.avg_profit_barrier, 0, 'f', 4)
                       .arg(diagnostics.profit_distance_pct, 0, 'f', 2)
                       .arg(diagnostics.avg_stop_barrier, 0, 'f', 4)
                       .arg(diagnostics.stop_distance_pct, 0, 'f', 2)
                       .arg(diagnostics.barrier_width_pct, 0, 'f', 2);
    }
    
    QString timing_stats = QString("<br>Avg periods to exit: Profit=%1, Stop=%2, Time=%3")
                          .arg(diagnostics.avg_profit_time, 0, 'f', 1)
                          .arg(diagnostics.avg_stop_time, 0, 'f', 1)
                          .arg(diagnostics.avg_time_time, 0, 'f', 1);
    
    return QString("<b>Barrier Diagnostics:</b><br>"
                   "Profit hits: %1, Stop hits: %2, Time hits: %3<br>"
                   "Avg volatility: %4, Min: %5, Max: %6%7%8")
           .arg(diagnostics.profit_hits).arg(diagnostics.stop_hits).arg(diagnostics.time_hits)
           .arg(diagnostics.avg_volatility, 0, 'e', 3)
           .arg(diagnostics.min_volatility, 0, 'e', 3)
           .arg(diagnostics.max_volatility, 0, 'e', 3)
           .arg(barrier_stats)
           .arg(timing_stats);
}

QString FeaturePreviewUtils::formatPortfolioResults(
    const MLPipeline::PortfolioResults& results,
    bool is_ttbm
) {
    QString title = is_ttbm ? "<b>TTBM Regression - Portfolio Simulation:</b><br><br>" 
                            : "<b>Hard Barrier Classification - Portfolio Simulation:</b><br><br>";
    
    QString strategy = formatTradingStrategy(is_ttbm);
    
    return QString("%1%2"
                   "<b>Portfolio Performance:</b><br>"
                   "Starting Capital: £%3<br>"
                   "Final Portfolio Value: £%4<br>"
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
           .arg(strategy)
           .arg(results.starting_capital, 0, 'f', 0)
           .arg(results.final_value, 0, 'f', 0)
           .arg(results.total_return * 100, 0, 'f', 2)
           .arg(results.annualized_return * 100, 0, 'f', 2)
           .arg(results.max_drawdown * 100, 0, 'f', 2)
           .arg(results.sharpe_ratio, 0, 'f', 3)
           .arg(results.total_trades)
           .arg(results.winning_trades)
           .arg(results.win_rate * 100, 0, 'f', 1)
           .arg(results.losing_trades)
           .arg((100.0 - results.win_rate * 100), 0, 'f', 1)
           .arg(results.avg_trade_return * 100, 0, 'f', 3)
           .arg(results.best_trade * 100, 0, 'f', 2)
           .arg(results.worst_trade * 100, 0, 'f', 2);
}

QString FeaturePreviewUtils::formatTradingStrategy(bool is_ttbm) {
    if (is_ttbm) {
        double multiplier = VisualizationConfig::getTTBMPositionMultiplier();
        return QString("<b>Trading Strategy:</b><br>"
               "• Position size = Signal strength × %1%% of portfolio<br>"
               "• Positive signal: Long position<br>"
               "• Negative signal: Short position<br>"
               "• Signal near zero: Small/no position<br><br>").arg(multiplier);
    } else {
        double positionSize = VisualizationConfig::getHardBarrierPositionSize();
        return QString("<b>Trading Strategy:</b><br>"
               "• Signal +1: Long position (%1%% of portfolio)<br>"
               "• Signal -1: Short position (%1%% of portfolio)<br>"
               "• Signal 0: No position<br><br>").arg(positionSize);
    }
}

QString FeaturePreviewUtils::formatModelInfo(
    bool is_ttbm,
    bool tune_enabled,
    const std::vector<LabeledEvent>& labeledEvents
) {
    QString model_type = is_ttbm ? "TTBM Regression" : "Hard Barrier Classification";
    QString objective = is_ttbm ? "reg:squarederror" : "binary:logistic";
    
    QString model_info = QString("<b>Model Type:</b> %1 (%2)<br><b>Hyperparameter Tuning:</b> %3<br>")
                        .arg(model_type)
                        .arg(objective)
                        .arg(tune_enabled ? "enabled" : "disabled");
    
    if (is_ttbm) {
        double min_label = 1.0, max_label = -1.0, mean_label = 0.0;
        int zero_labels = 0, positive_count = 0, negative_count = 0;
        
        for (const auto& e : labeledEvents) {
            double label = e.ttbm_label;
            min_label = std::min(min_label, label);
            max_label = std::max(max_label, label);
            mean_label += label;
            if (std::abs(label) < 0.01) zero_labels++;
            else if (label > 0) positive_count++;
            else negative_count++;
        }
        mean_label /= labeledEvents.size();
        
        double positionMultiplier = VisualizationConfig::getTTBMPositionMultiplier();
        QString label_analysis = QString("TTBM Regression: Predicting Directional Confidence<br>"
                                       "Target: TTBM label (directional confidence from -1 to +1)<br>"
                                       "Range: [%1, %2], Mean: %3<br>"
                                       "Zero/Neutral: %4/%5 (%6%)<br>"
                                       "Positive Labels: %7/%8 (%9%)<br>"
                                       "Negative Labels: %10/%11 (%12%)<br>"
                                       "<b>Portfolio simulation scales bet size by signal strength</b><br>"
                                       "<b>Position size = |prediction| * %13% of portfolio (max %14%)</b>")
                                .arg(min_label, 0, 'f', 4)
                                .arg(max_label, 0, 'f', 4)
                                .arg(mean_label, 0, 'f', 4)
                                .arg(zero_labels).arg(labeledEvents.size())
                                .arg(100.0 * zero_labels / labeledEvents.size(), 0, 'f', 1)
                                .arg(positive_count).arg(labeledEvents.size())
                                .arg(100.0 * positive_count / labeledEvents.size(), 0, 'f', 1)
                                .arg(negative_count).arg(labeledEvents.size())
                                .arg(100.0 * negative_count / labeledEvents.size(), 0, 'f', 1)
                                .arg(positionMultiplier)
                                .arg(positionMultiplier);
        
        return model_info + label_analysis;
    } else {
        int profit_labels = 0, stop_labels = 0, neutral_labels = 0;
        double mean_returns = 0.0;
        
        for (const auto& e : labeledEvents) {
            mean_returns += (e.exit_price - e.entry_price);
            if (e.label == 1) profit_labels++;
            else if (e.label == -1) stop_labels++;
            else neutral_labels++;
        }
        mean_returns /= labeledEvents.size();
        
        QString label_analysis = QString("Hard Labels: Profit=%1, Stop=%2, Neutral=%3<br>"
                                       "Average Return: %4")
                                .arg(profit_labels).arg(stop_labels).arg(neutral_labels)
                                .arg(mean_returns, 0, 'f', 6);
        
        return model_info + label_analysis;
    }
}

QString FeaturePreviewUtils::formatSampleTradingDecisions(
    const std::vector<double>& predictions,
    bool is_ttbm,
    int max_samples
) {
    QString debug = "<b>Sample Trading Decisions (first " + QString::number(max_samples) + "):</b><br>";
    
    for (size_t i = 0; i < std::min(size_t(max_samples), predictions.size()); ++i) {
        double pred = predictions[i];
        
        QString trade_info;
        if (is_ttbm) {
            double multiplier = VisualizationConfig::getTTBMPositionMultiplier();
            double threshold = VisualizationConfig::getTradingThreshold();
            double position_size = std::abs(pred) * multiplier;
            QString direction = pred > 0 ? "LONG" : "SHORT";
            
            if (std::abs(pred) < threshold) {
                trade_info = QString("Sample %1: Signal=%2 → NO TRADE (signal too weak)")
                            .arg(i + 1).arg(pred, 0, 'f', 4);
            } else {
                trade_info = QString("Sample %1: Signal=%2 → %3 %4%% position")
                            .arg(i + 1).arg(pred, 0, 'f', 4).arg(direction).arg(position_size, 0, 'f', 2);
            }
        } else {
            double positionSize = VisualizationConfig::getHardBarrierPositionSize();
            double threshold = VisualizationConfig::getTradingThreshold();
            if (std::abs(pred - 1.0) < threshold) {
                trade_info = QString("Sample %1: Signal=%2 → LONG %3%% position")
                            .arg(i + 1).arg(pred, 0, 'f', 4).arg(positionSize);
            } else if (std::abs(pred + 1.0) < threshold) {
                trade_info = QString("Sample %1: Signal=%2 → SHORT %3%% position")
                            .arg(i + 1).arg(pred, 0, 'f', 4).arg(positionSize);
            } else {
                trade_info = QString("Sample %1: Signal=%2 → NO TRADE")
                            .arg(i + 1).arg(pred, 0, 'f', 4);
            }
        }
        
        debug += trade_info + "<br>";
    }
    
    return debug;
}
