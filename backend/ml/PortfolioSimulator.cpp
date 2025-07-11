#include "PortfolioSimulator.h"
#include "../data/LabeledEvent.h"
#include "../data/PreprocessedRow.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>

namespace MLPipeline {

PortfolioSimulation simulate_portfolio(
    const std::vector<double>& trading_signals,
    const std::vector<double>& returns,
    bool is_hard_barrier,
    const PortfolioConfig& portfolio_config
) {
    if (trading_signals.empty() || returns.empty()) {
        throw std::invalid_argument("Signals and returns cannot be empty");
    }
    if (trading_signals.size() != returns.size()) {
        throw std::invalid_argument("Signals and returns must have the same size");
    }
    
    double capital = portfolio_config.starting_capital;
    double max_capital = capital;
    double min_capital = capital;
    std::vector<double> capital_history;
    capital_history.push_back(capital);
    
    int total_trades = 0;
    int winning_trades = 0;
    double total_pnl = 0;
    std::vector<std::string> trade_decisions;
    std::vector<double> trade_returns; // Store PnL for each trade
    for (size_t i = 0; i < trading_signals.size(); ++i) {
        double position_pct = 0;
        std::string decision;
        double trade_capital = capital; // Use capital before trade for PnL calculation
        
        if (is_hard_barrier) {
            if (trading_signals[i] > 0.5) {
                position_pct = portfolio_config.hard_barrier_position_pct;
                decision = "BUY " + std::to_string(portfolio_config.hard_barrier_position_pct * 100) + "%";
            } else if (trading_signals[i] < -0.5) {
                position_pct = -portfolio_config.hard_barrier_position_pct;
                decision = "SELL " + std::to_string(portfolio_config.hard_barrier_position_pct * 100) + "%";
            } else {
                position_pct = 0;
                decision = "HOLD";
            }
        } else {
            position_pct = std::min(std::abs(trading_signals[i]) * portfolio_config.max_position_pct, portfolio_config.max_position_pct);
            if (trading_signals[i] < 0) position_pct = -position_pct;
            
            if (position_pct > portfolio_config.position_threshold) {
                decision = "BUY " + std::to_string(position_pct * 100) + "%";
            } else if (position_pct < -portfolio_config.position_threshold) {
                decision = "SELL " + std::to_string(std::abs(position_pct) * 100) + "%";
            } else {
                decision = "HOLD";
            }
        }
        
        if (std::abs(position_pct) > portfolio_config.position_threshold) {
            total_trades++;
            double pnl = position_pct * trade_capital * returns[i]; 
            total_pnl += pnl;
            capital += pnl;
            trade_returns.push_back(pnl); 
            if (pnl > 0) winning_trades++;
        }
        
        max_capital = std::max(max_capital, capital);
        min_capital = std::min(min_capital, capital);
        capital_history.push_back(capital);
        
        if (trade_decisions.size() < static_cast<size_t>(portfolio_config.max_trade_decisions_logged)) {
            trade_decisions.push_back(decision);
        }
    }
    
    double total_return = (capital - portfolio_config.starting_capital) / portfolio_config.starting_capital;
    double max_drawdown = (max_capital - min_capital) / max_capital;
    
    double annualized_return = total_return * portfolio_config.trading_days_per_year / trading_signals.size();
    
    double avg_daily_return = total_return / trading_signals.size();
    double daily_variance = 0;
    if (capital_history.size() > 1) {
        for (size_t i = 1; i < capital_history.size(); ++i) {
            double daily_ret = (capital_history[i] - capital_history[i-1]) / capital_history[i-1];
            daily_variance += (daily_ret - avg_daily_return) * (daily_ret - avg_daily_return);
        }
        daily_variance /= (capital_history.size() - 1);
    }
    double daily_std = std::sqrt(daily_variance);
    double sharpe_ratio = (daily_std > 1e-10) ? avg_daily_return / daily_std * std::sqrt(portfolio_config.trading_days_per_year) : 0;
    
    double win_rate = total_trades > 0 ? winning_trades / static_cast<double>(total_trades) : 0;
    
    return {portfolio_config.starting_capital, capital, total_return, annualized_return, max_drawdown, 
            sharpe_ratio, total_trades, win_rate, trade_decisions, trade_returns};
}

BarrierDiagnostics analyzeBarriers(
    const std::vector<LabeledEvent>& labeledEvents,
    const std::vector<PreprocessedRow>& rows
) {
    BarrierDiagnostics diagnostics;
    
    if (labeledEvents.empty()) return diagnostics;
    
    std::vector<double> entry_prices, profit_barriers, stop_barriers;
    std::vector<int> profit_times, stop_times, time_times;
    
    for (const auto& event : labeledEvents) {
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
        
        for (const auto& row : rows) {
            if (row.timestamp == event.entry_time) {
                diagnostics.avg_volatility += row.volatility;
                diagnostics.max_volatility = std::max(diagnostics.max_volatility, row.volatility);
                
                if (diagnostics.min_volatility == 0.0) {
                    diagnostics.min_volatility = row.volatility;
                } else {
                    diagnostics.min_volatility = std::min(diagnostics.min_volatility, row.volatility);
                }
                
                double entry_price = row.price;
                double exit_price = event.exit_price;
                double price_move = std::abs(exit_price - entry_price);
                double volatility = row.volatility;
                
                double estimated_multiple = volatility > 0 ? price_move / volatility : 0.0;
                double profit_barrier = entry_price + estimated_multiple * volatility;
                double stop_barrier = entry_price - estimated_multiple * volatility;
                
                entry_prices.push_back(entry_price);
                profit_barriers.push_back(profit_barrier);
                stop_barriers.push_back(stop_barrier);
                break;
            }
        }
    }
    
    diagnostics.avg_volatility /= labeledEvents.size();
    
    if (!entry_prices.empty()) {
        diagnostics.avg_entry_price = std::accumulate(entry_prices.begin(), entry_prices.end(), 0.0) / entry_prices.size();
        diagnostics.avg_profit_barrier = std::accumulate(profit_barriers.begin(), profit_barriers.end(), 0.0) / profit_barriers.size();
        diagnostics.avg_stop_barrier = std::accumulate(stop_barriers.begin(), stop_barriers.end(), 0.0) / stop_barriers.size();
        
        diagnostics.barrier_width_pct = ((diagnostics.avg_profit_barrier - diagnostics.avg_stop_barrier) / diagnostics.avg_entry_price) * 100.0;
        diagnostics.profit_distance_pct = ((diagnostics.avg_profit_barrier - diagnostics.avg_entry_price) / diagnostics.avg_entry_price) * 100.0;
        diagnostics.stop_distance_pct = ((diagnostics.avg_entry_price - diagnostics.avg_stop_barrier) / diagnostics.avg_entry_price) * 100.0;
    }
    
    auto calc_avg = [](const std::vector<int>& vec) -> double {
        return vec.empty() ? 0.0 : std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
    };
    
    diagnostics.avg_profit_time = calc_avg(profit_times);
    diagnostics.avg_stop_time = calc_avg(stop_times);
    diagnostics.avg_time_time = calc_avg(time_times);
    
    return diagnostics;
}

} // namespace MLPipeline
