#pragma once
#include <vector>
#include <string>
#include "LabeledEvent.h"
#include "PreprocessedRow.h"

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

class PortfolioSimulator {
public:
    static PortfolioResults runSimulation(
        const std::vector<double>& predictions,
        const std::vector<LabeledEvent>& events,
        bool is_ttbm = false
    );
    
    static BarrierDiagnostics analyzeBarriers(
        const std::vector<LabeledEvent>& labeledEvents,
        const std::vector<PreprocessedRow>& rows
    );
    
    static double calculateSharpeRatio(const std::vector<double>& returns);
    static double calculateMaxDrawdown(const std::vector<double>& portfolio_values);
    
private:
    static double calculatePositionSize(double prediction, bool is_ttbm);
};
