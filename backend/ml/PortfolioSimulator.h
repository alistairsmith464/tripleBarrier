#pragma once
#include <vector>
#include <string>

namespace MLPipeline {

struct PortfolioSimulation {
    double starting_capital;
    double final_capital;
    double total_return;
    double annualized_return;
    double max_drawdown;
    double sharpe_ratio;
    int total_trades;
    double win_rate;
    std::vector<std::string> trade_decisions;
};

struct PortfolioConfig {
    double starting_capital = 10000.0;
    double max_position_pct = 0.1;
    double position_threshold = 0.01;
    double hard_barrier_position_pct = 0.05;
    double trading_days_per_year = 252.0;
    int max_trade_decisions_logged = 100;
};

/**
 * Simulates portfolio performance based on trading signals and returns
 * @param signals Trading signals (probabilities or hard decisions)
 * @param returns Asset returns for each time period
 * @param is_hard_barrier Whether to use hard barrier strategy (discrete) or continuous
 * @param portfolio_config Portfolio configuration parameters
 * @return Portfolio simulation results
 */
PortfolioSimulation simulate_portfolio(
    const std::vector<double>& signals,
    const std::vector<double>& returns,
    bool is_hard_barrier,
    const PortfolioConfig& portfolio_config = PortfolioConfig{}
);

} // namespace MLPipeline
