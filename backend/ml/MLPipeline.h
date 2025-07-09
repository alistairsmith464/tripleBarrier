#pragma once
#include <vector>
#include <map>
#include <string>
#include "XGBoostModel.h"

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

    struct PipelineResult {
        std::vector<int> predictions;
        std::vector<double> probabilities;
        std::map<std::string, double> feature_importances;
        PortfolioSimulation portfolio;
    };

    struct RegressionPipelineResult {
        std::vector<double> predictions;
        std::vector<double> uncertainties;
        std::map<std::string, double> feature_importances;
        PortfolioSimulation portfolio;
    };

    enum SplitType {
        Chronological,
        PurgedKFold
    };

    struct PipelineConfig {
        SplitType split_type;
        int n_splits;
        int embargo;
        double train_ratio, val_ratio, test_ratio;
        int random_seed;
        int n_rounds = 20;
        int max_depth = 3;
        int nthread = 4;
        std::string objective = "binary:logistic";
    };

    PipelineResult runPipeline(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        const std::vector<double>& returns,
        const PipelineConfig& config
    );

    PipelineResult runPipelineWithTuning(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        const std::vector<double>& returns,
        PipelineConfig config
    );

    RegressionPipelineResult runPipelineRegression(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& returns,
        const PipelineConfig& config
    );

    RegressionPipelineResult runPipelineRegressionWithTuning(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<double>& y,
        const std::vector<double>& returns,
        PipelineConfig config
    );
}
