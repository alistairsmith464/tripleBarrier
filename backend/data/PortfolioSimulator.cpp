#include "PortfolioSimulator.h"
#include <algorithm>
#include <numeric>
#include <cmath>

PortfolioResults PortfolioSimulator::runSimulation(
    const std::vector<double>& predictions,
    const std::vector<LabeledEvent>& events,
    bool is_ttbm
) {
    PortfolioResults results;
    double portfolio_value = results.starting_capital;
    results.portfolio_values.push_back(portfolio_value);
    
    for (size_t i = 0; i < predictions.size() && i < events.size(); ++i) {
        double prediction = predictions[i];
        double actual_return = (events[i].exit_price - events[i].entry_price) / events[i].entry_price;
        
        double position_size = calculatePositionSize(prediction, is_ttbm);
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

BarrierDiagnostics PortfolioSimulator::analyzeBarriers(
    const std::vector<LabeledEvent>& labeledEvents,
    const std::vector<PreprocessedRow>& rows
) {
    BarrierDiagnostics diagnostics;
    
    if (labeledEvents.empty()) return diagnostics;
    
    std::vector<double> entry_prices, profit_barriers, stop_barriers;
    std::vector<int> profit_times, stop_times, time_times;
    
    for (const auto& event : labeledEvents) {
        // Count label types
        if (event.label == 1) {
            diagnostics.profit_hits++;
            profit_times.push_back(event.periods_to_exit);
        } else if (event.label == -1) {
            diagnostics.stop_hits++;
            stop_times.push_back(event.periods_to_exit);
        } else {
            diagnostics.time_hits++;
            time_times.push_back(event.periods_to_exit);
        }
        
        // Find corresponding row for volatility analysis
        auto it = std::find_if(rows.begin(), rows.end(), 
            [&](const PreprocessedRow& r) { return r.timestamp == event.entry_time; });
        
        if (it != rows.end()) {
            diagnostics.avg_volatility += it->volatility;
            diagnostics.max_volatility = std::max(diagnostics.max_volatility, it->volatility);
            
            if (diagnostics.min_volatility == 0.0) {
                diagnostics.min_volatility = it->volatility;
            } else {
                diagnostics.min_volatility = std::min(diagnostics.min_volatility, it->volatility);
            }
            
            // Calculate estimated barriers
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
    
    diagnostics.avg_volatility /= labeledEvents.size();
    
    // Calculate barrier statistics
    if (!entry_prices.empty()) {
        diagnostics.avg_entry_price = std::accumulate(entry_prices.begin(), entry_prices.end(), 0.0) / entry_prices.size();
        diagnostics.avg_profit_barrier = std::accumulate(profit_barriers.begin(), profit_barriers.end(), 0.0) / profit_barriers.size();
        diagnostics.avg_stop_barrier = std::accumulate(stop_barriers.begin(), stop_barriers.end(), 0.0) / stop_barriers.size();
        
        diagnostics.barrier_width_pct = ((diagnostics.avg_profit_barrier - diagnostics.avg_stop_barrier) / diagnostics.avg_entry_price) * 100.0;
        diagnostics.profit_distance_pct = ((diagnostics.avg_profit_barrier - diagnostics.avg_entry_price) / diagnostics.avg_entry_price) * 100.0;
        diagnostics.stop_distance_pct = ((diagnostics.avg_entry_price - diagnostics.avg_stop_barrier) / diagnostics.avg_entry_price) * 100.0;
    }
    
    // Calculate average exit times
    auto calc_avg = [](const std::vector<int>& vec) -> double {
        return vec.empty() ? 0.0 : std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
    };
    
    diagnostics.avg_profit_time = calc_avg(profit_times);
    diagnostics.avg_stop_time = calc_avg(stop_times);
    diagnostics.avg_time_time = calc_avg(time_times);
    
    return diagnostics;
}

double PortfolioSimulator::calculateSharpeRatio(const std::vector<double>& returns) {
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

double PortfolioSimulator::calculateMaxDrawdown(const std::vector<double>& portfolio_values) {
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

double PortfolioSimulator::calculatePositionSize(double prediction, bool is_ttbm) {
    if (is_ttbm) {
        double signal_strength = std::abs(prediction);
        if (signal_strength > 0.1) {
            double position_size = std::min(signal_strength * 0.03, 0.03);
            return prediction < 0 ? -position_size : position_size;
        }
    } else {
        if (std::abs(prediction - 1.0) < 0.1) {
            return 0.02;  // Long position
        } else if (std::abs(prediction + 1.0) < 0.1) {
            return -0.02;  // Short position
        }
    }
    return 0.0;  // No position
}
