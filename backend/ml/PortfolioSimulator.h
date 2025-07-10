#pragma once
#include <vector>
#include <string>

// Forward declarations
struct LabeledEvent;
struct PreprocessedRow;

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

// Compatibility struct for frontend interface
struct PortfolioResults {
    double starting_capital = 100000.0;
    double final_value = 0.0;
    double total_return = 0.0;
    double annualized_return = 0.0;
    double max_drawdown = 0.0;
    double sharpe_ratio = 0.0;
    int total_trades = 0;
    int winning_trades = 0;
    int losing_trades = 0;
    double win_rate = 0.0;
    double avg_trade_return = 0.0;
    double best_trade = 0.0;
    double worst_trade = 0.0;
    std::vector<double> portfolio_values;
    std::vector<double> trade_returns;
};

struct BarrierDiagnostics {
    int profit_hits = 0;
    int stop_hits = 0;
    int time_hits = 0;
    double avg_volatility = 0.0;
    double min_volatility = 0.0;
    double max_volatility = 0.0;
    double avg_profit_time = 0.0;
    double avg_stop_time = 0.0;
    double avg_time_time = 0.0;
    double avg_entry_price = 0.0;
    double avg_profit_barrier = 0.0;
    double avg_stop_barrier = 0.0;
    double barrier_width_pct = 0.0;
    double profit_distance_pct = 0.0;
    double stop_distance_pct = 0.0;
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
    const std::vector<double>& trading_signals,
    const std::vector<double>& returns,
    bool is_hard_barrier,
    const PortfolioConfig& portfolio_config = PortfolioConfig{}
);

/**
 * Analyzes barrier statistics from labeled events
 * @param labeledEvents Labeled events to analyze
 * @param rows Preprocessed data rows
 * @return Barrier diagnostics
 */
BarrierDiagnostics analyzeBarriers(
    const std::vector<LabeledEvent>& labeledEvents,
    const std::vector<PreprocessedRow>& rows
);

} // namespace MLPipeline
