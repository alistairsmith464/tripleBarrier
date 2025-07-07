#pragma once
#include <vector>
#include <map>
#include <string>
#include "XGBoostModel.h"

namespace MLPipeline {
    struct MetricsResult {
        double accuracy;
        double precision;
        double recall;
        double f1;
        double avg_return;
        double sharpe_ratio;
        double hit_ratio;
        int true_positives = 0;
        int true_negatives = 0;
        int false_positives = 0;
        int false_negatives = 0;
        int total = 0;
    };

    struct PipelineResult {
        std::vector<int> predictions;
        std::vector<double> probabilities;
        std::map<std::string, double> feature_importances;
        MetricsResult metrics;
    };

    enum SplitType {
        Chronological,
        PurgedKFold
    };

    struct PipelineConfig {
        SplitType split_type;
        int n_splits; // for KFold
        int embargo;  // for PurgedKFold
        double train_ratio, val_ratio, test_ratio; // for Chronological
        int random_seed;
        // XGBoost hyperparameters
        int n_rounds = 20;
        int max_depth = 3;
        int nthread = 4;
        std::string objective = "binary:logistic";
    };

    // Main pipeline entry point (classification, hard labels)
    PipelineResult runPipeline(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        const std::vector<double>& returns, // for financial metrics
        const PipelineConfig& config
    );

    // New: Regression/soft-label pipeline entry point (probabilistic barriers)
    PipelineResult runPipelineSoft(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<double>& y_soft,
        const std::vector<double>& returns, // for financial metrics
        const PipelineConfig& config
    );

    // Grid search for best hyperparameters (classification, hard labels)
    PipelineResult runPipelineWithTuning(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        const std::vector<double>& returns,
        PipelineConfig config // used as base config, but n_rounds/max_depth/nthread/objective are tuned
    );

    // Grid search for best hyperparameters (regression, soft labels)
    PipelineResult runPipelineSoftWithTuning(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<double>& y_soft,
        const std::vector<double>& returns,
        PipelineConfig config // used as base config, but n_rounds/max_depth/nthread/objective are tuned
    );
}
