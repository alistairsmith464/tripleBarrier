#include "MLPipeline.h"
#include "MLSplits.h"
#include "DataUtils.h"
#include "ModelUtils.h"
#include "MetricsCalculator.h"
#include "PortfolioSimulator.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace MLPipeline {

template<typename T>
void validatePipelineInputs(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<T>& y,
    const std::vector<double>& returns
) {
    if (X.empty() || y.empty() || returns.empty()) {
        throw std::invalid_argument("Input data cannot be empty");
    }
    if (X.size() != y.size() || X.size() != returns.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }
}

template<typename T, typename ResultType>
ResultType runPipelineTemplate(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<T>& y,
    const std::vector<double>& returns,
    const UnifiedPipelineConfig& config,
    bool is_classification
) {
    validatePipelineInputs(X, y, returns);
    
    DataProcessor::CleaningOptions cleaningOpts;
    cleaningOpts.remove_nan = true;
    cleaningOpts.remove_inf = true;
    cleaningOpts.remove_outliers = true;
    
    auto [X_clean, y_clean, returns_clean] = DataProcessor::cleanData(X, y, returns, cleaningOpts);
    
    auto [train_idx, val_idx, test_idx] = createSplits(X_clean.size(), config);

    auto X_train_f = toFloatMatrix(select_rows(X_clean, train_idx));
    
    std::vector<size_t> eval_idx;
    if (val_idx.empty()) {
        eval_idx = test_idx;
        std::cerr << "Warning: No validation set available, using test set for evaluation (potential data leakage)" << std::endl;
    } else {
        eval_idx = val_idx;
    }
    
    auto X_eval_f = toFloatMatrix(select_rows(X_clean, eval_idx));
    auto returns_eval = select_rows(returns_clean, eval_idx);

    XGBoostConfig model_config;
    model_config.n_rounds = config.n_rounds;
    model_config.max_depth = config.max_depth;
    model_config.nthread = config.nthread;
    model_config.objective = config.objective;
    model_config.learning_rate = config.learning_rate;
    model_config.subsample = config.subsample;
    model_config.colsample_bytree = config.colsample_bytree;

    XGBoostModel model;
    
    if constexpr (std::is_same_v<T, int>) {
        auto y_train_f = toFloatVecInt(select_rows(y_clean, train_idx));
        model.fit(X_train_f, y_train_f, model_config);
        
        auto y_pred = model.predict(X_eval_f);
        auto y_prob = model.predict_proba(X_eval_f);
        
        std::vector<double> signals;
        if (config.barrier_type == BarrierType::HARD) {
            signals.assign(y_pred.begin(), y_pred.end());
        } else {
            signals.assign(y_prob.begin(), y_prob.end());
        }
        
        PortfolioSimulation portfolio = simulate_portfolio(signals, returns_eval, 
                                                         config.barrier_type == BarrierType::HARD);
        
        std::vector<double> y_prob_d(y_prob.begin(), y_prob.end());
        std::vector<TradeLogEntry> trade_log = portfolio.trade_log;
        return ResultType{y_pred, y_prob_d, portfolio, trade_log};
    } else {
        auto y_train_f = toFloatVecDouble(select_rows(y_clean, train_idx));
        
        if (!y_train_f.empty()) {
            float min_train = *std::min_element(y_train_f.begin(), y_train_f.end());
            float max_train = *std::max_element(y_train_f.begin(), y_train_f.end());
            float sum_train = std::accumulate(y_train_f.begin(), y_train_f.end(), 0.0f);
            float mean_train = sum_train / y_train_f.size();
        }
        
        model.fit(X_train_f, y_train_f, model_config);
        
        auto y_pred_f = model.predict_raw(X_eval_f); 
        
        if (!y_pred_f.empty()) {
            float min_raw = *std::min_element(y_pred_f.begin(), y_pred_f.end());
            float max_raw = *std::max_element(y_pred_f.begin(), y_pred_f.end());
            float sum_raw = std::accumulate(y_pred_f.begin(), y_pred_f.end(), 0.0f);
            float mean_raw = sum_raw / y_pred_f.size();
        }
        
        std::vector<double> y_pred(y_pred_f.begin(), y_pred_f.end());
        
        if (!y_pred.empty()) {
            double min_pred = *std::min_element(y_pred.begin(), y_pred.end());
            double max_pred = *std::max_element(y_pred.begin(), y_pred.end());
            double sum_pred = std::accumulate(y_pred.begin(), y_pred.end(), 0.0);
            double mean_pred = sum_pred / y_pred.size();
        }
        
        PortfolioSimulation portfolio = simulate_portfolio(y_pred, returns_eval, false);
        
        std::vector<TradeLogEntry> trade_log = portfolio.trade_log;
        return ResultType{y_pred, portfolio, trade_log};
    }
}

template<typename T, typename ResultType>
ResultType runPipelineWithTuningTemplate(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<T>& y,
    const std::vector<double>& returns,
    UnifiedPipelineConfig config,
    bool is_classification
) {
    validatePipelineInputs(X, y, returns);
    
    auto [X_clean, y_clean, returns_clean] = DataProcessor::cleanData(X, y, returns);
    auto [train_idx, val_idx, test_idx] = createSplits(X_clean.size(), config);

    if (val_idx.empty()) {
        throw std::invalid_argument("Hyperparameter tuning requires a validation set");
    }

    auto X_train_f = toFloatMatrix(select_rows(X_clean, train_idx));
    auto X_val_f = toFloatMatrix(select_rows(X_clean, val_idx));
    auto y_val = select_rows(y_clean, val_idx);

    double best_score = is_classification ? -1.0 : -std::numeric_limits<double>::infinity();
    UnifiedPipelineConfig best_config = config;
    
    const auto& grid = config.hyperparameter_grid;
    size_t total_combinations = grid.n_rounds.size() * grid.max_depth.size() * 
                               grid.learning_rate.size() * grid.subsample.size() * 
                               grid.colsample_bytree.size();
    
    size_t combination_count = 0;
    bool early_stop = false;
    
    for (int n_rounds : grid.n_rounds) {
        if (early_stop) break;
        for (int max_depth : grid.max_depth) {
            if (early_stop) break;
            for (double lr : grid.learning_rate) {
                if (early_stop) break;
                for (double subsample : grid.subsample) {
                    if (early_stop) break;
                    for (double colsample : grid.colsample_bytree) {
                        combination_count++;
                        
                        XGBoostConfig model_config;
                        model_config.n_rounds = n_rounds;
                        model_config.max_depth = max_depth;
                        model_config.nthread = config.nthread;
                        model_config.objective = config.objective;
                        model_config.learning_rate = lr;
                        model_config.subsample = subsample;
                        model_config.colsample_bytree = colsample;

                        try {
                            XGBoostModel model;
                            
                            double score;
                            MetricsCalculator metricsCalc;
                            if constexpr (std::is_same_v<T, int>) {
                                auto y_train_f = toFloatVecInt(select_rows(y_clean, train_idx));
                                model.fit(X_train_f, y_train_f, model_config);
                                auto y_pred_val = model.predict(X_val_f);
                                score = metricsCalc.calculateF1Score(y_val, y_pred_val);
                            } else {
                                auto y_train_f = toFloatVecDouble(select_rows(y_clean, train_idx));
                                model.fit(X_train_f, y_train_f, model_config);
                                auto y_pred_val_f = model.predict(X_val_f);
                                std::vector<double> y_pred_val(y_pred_val_f.begin(), y_pred_val_f.end());
                                score = metricsCalc.calculateR2Score(y_val, y_pred_val);
                            }
                            
                            if (score > best_score) {
                                best_score = score;
                                best_config.n_rounds = n_rounds;
                                best_config.max_depth = max_depth;
                                best_config.learning_rate = lr;
                                best_config.subsample = subsample;
                                best_config.colsample_bytree = colsample;
                            }
                            
                            if ((is_classification && score > 0.95) || (!is_classification && score > 0.99)) {
                                early_stop = true;
                                break;
                            }
                            
                        } catch (const std::exception& e) {
                            std::cerr << "Error in hyperparameter combination " << combination_count 
                                     << ": " << e.what() << std::endl;
                            continue;
                        }
                        
                        if (early_stop) break;
                    }
                }
            }
        }
    }
    
    return runPipelineTemplate<T, ResultType>(X_clean, y_clean, returns_clean, best_config, is_classification);
}

PipelineResult runPipeline(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<int>& y,
    const std::vector<double>& returns,
    const UnifiedPipelineConfig& config
) {
    return runPipelineTemplate<int, PipelineResult>(X, y, returns, config, true);
}

PipelineResult runPipelineWithTuning(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<int>& y,
    const std::vector<double>& returns,
    UnifiedPipelineConfig config
) {
    return runPipelineWithTuningTemplate<int, PipelineResult>(X, y, returns, config, true);
}

RegressionPipelineResult runPipelineRegression(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& returns,
    const UnifiedPipelineConfig& config
) {
    return runPipelineTemplate<double, RegressionPipelineResult>(X, y, returns, config, false);
}

RegressionPipelineResult runPipelineRegressionWithTuning(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& returns,
    UnifiedPipelineConfig config
) {
    return runPipelineWithTuningTemplate<double, RegressionPipelineResult>(X, y, returns, config, false);
}

PipelineResult runPipeline(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<int>& y,
    const std::vector<double>& returns,
    const PipelineConfig& config
) {
    UnifiedPipelineConfig unified_config;
    unified_config.test_size = config.test_size;
    unified_config.val_size = config.val_size;
    unified_config.n_rounds = config.n_rounds;
    unified_config.max_depth = config.max_depth;
    unified_config.nthread = config.nthread;
    unified_config.objective = config.objective;
    unified_config.barrier_type = (config.objective == "binary:logistic") ? BarrierType::HARD : BarrierType::SOFT;
    
    return runPipeline(X, y, returns, unified_config);
}

PipelineResult runPipelineWithTuning(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<int>& y,
    const std::vector<double>& returns,
    PipelineConfig config
) {
    UnifiedPipelineConfig unified_config;
    unified_config.test_size = config.test_size;
    unified_config.val_size = config.val_size;
    unified_config.n_rounds = config.n_rounds;
    unified_config.max_depth = config.max_depth;
    unified_config.nthread = config.nthread;
    unified_config.objective = config.objective;
    unified_config.barrier_type = (config.objective == "binary:logistic") ? BarrierType::HARD : BarrierType::SOFT;
    
    return runPipelineWithTuning(X, y, returns, unified_config);
}

RegressionPipelineResult runPipelineRegression(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& returns,
    const PipelineConfig& config
) {
    UnifiedPipelineConfig unified_config;
    unified_config.test_size = config.test_size;
    unified_config.val_size = config.val_size;
    unified_config.n_rounds = config.n_rounds;
    unified_config.max_depth = config.max_depth;
    unified_config.nthread = config.nthread;
    unified_config.objective = config.objective;
    unified_config.barrier_type = BarrierType::SOFT;
    
    return runPipelineRegression(X, y, returns, unified_config);
}

RegressionPipelineResult runPipelineRegressionWithTuning(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& returns,
    const PipelineConfig& config
) {
    UnifiedPipelineConfig unified_config;
    unified_config.test_size = config.test_size;
    unified_config.val_size = config.val_size;
    unified_config.n_rounds = config.n_rounds;
    unified_config.max_depth = config.max_depth;
    unified_config.nthread = config.nthread;
    unified_config.objective = config.objective;
    unified_config.barrier_type = BarrierType::SOFT;
    
    return runPipelineRegressionWithTuning(X, y, returns, unified_config);
}

}
