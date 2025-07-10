#include "PortfolioSimulator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace MLPipeline {

PortfolioSimulation simulate_portfolio(
    const std::vector<double>& signals,
    const std::vector<double>& returns,
    bool is_hard_barrier,
    const PortfolioConfig& portfolio_config
) {
    if (signals.empty() || returns.empty()) {
        throw std::invalid_argument("Signals and returns cannot be empty");
    }
    if (signals.size() != returns.size()) {
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
    
    for (size_t i = 0; i < signals.size(); ++i) {
        double position_pct = 0;
        std::string decision;
        
        if (is_hard_barrier) {
            if (signals[i] > 0.5) {
                position_pct = portfolio_config.hard_barrier_position_pct;
                decision = "BUY " + std::to_string(portfolio_config.hard_barrier_position_pct * 100) + "%";
            } else if (signals[i] < -0.5) {
                position_pct = -portfolio_config.hard_barrier_position_pct;
                decision = "SELL " + std::to_string(portfolio_config.hard_barrier_position_pct * 100) + "%";
            } else {
                position_pct = 0;
                decision = "HOLD";
            }
        } else {
            position_pct = std::min(std::abs(signals[i]) * portfolio_config.max_position_pct, portfolio_config.max_position_pct);
            if (signals[i] < 0) position_pct = -position_pct;
            
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
            double pnl = position_pct * capital * returns[i];
            total_pnl += pnl;
            capital += pnl;
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
    
    double annualized_return = total_return * portfolio_config.trading_days_per_year / signals.size();
    
    double avg_daily_return = total_return / signals.size();
    double daily_variance = 0;
    for (size_t i = 1; i < capital_history.size(); ++i) {
        double daily_ret = (capital_history[i] - capital_history[i-1]) / capital_history[i-1];
        daily_variance += (daily_ret - avg_daily_return) * (daily_ret - avg_daily_return);
    }
    double daily_std = std::sqrt(daily_variance / (capital_history.size() - 1));
    double sharpe_ratio = daily_std > 0 ? avg_daily_return / daily_std * std::sqrt(portfolio_config.trading_days_per_year) : 0;
    
    double win_rate = total_trades > 0 ? winning_trades / static_cast<double>(total_trades) : 0;
    
    return {portfolio_config.starting_capital, capital, total_return, annualized_return, max_drawdown, 
            sharpe_ratio, total_trades, win_rate, trade_decisions};
}

} // namespace MLPipeline
