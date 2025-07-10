#include "MLPipeline.h"
#include "../data/DataCleaningUtils.h"
#include "MLSplits.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace MLPipeline;

namespace {
double calculate_f1_score(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    int tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1 && y_pred[i] == 1) tp++;
        else if (y_true[i] == 0 && y_pred[i] == 1) fp++;
        else if (y_true[i] == 1 && y_pred[i] == 0) fn++;
    }
    
    double precision = (tp + fp == 0) ? 0 : tp / static_cast<double>(tp + fp);
    double recall = (tp + fn == 0) ? 0 : tp / static_cast<double>(tp + fn);
    
    return (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);
}

double calculate_r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;
    
    double y_mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();
    
    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        ss_tot += (y_true[i] - y_mean) * (y_true[i] - y_mean);
    }
    
    return (ss_tot == 0) ? 0.0 : 1.0 - (ss_res / ss_tot);
}

PortfolioSimulation simulate_portfolio(
    const std::vector<double>& signals,
    const std::vector<double>& returns,
    bool is_hard_barrier = false
) {
    double capital = 100000.0;
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
                position_pct = 0.02;
                decision = "BUY 2%";
            } else if (signals[i] < -0.5) {
                position_pct = -0.02;
                decision = "SELL 2%";
            } else {
                position_pct = 0;
                decision = "HOLD";
            }
        } else {
            position_pct = std::min(std::abs(signals[i]) * 0.03, 0.03);
            if (signals[i] < 0) position_pct = -position_pct;
            
            if (position_pct > 0.001) {
                decision = "BUY " + std::to_string(position_pct * 100) + "%";
            } else if (position_pct < -0.001) {
                decision = "SELL " + std::to_string(std::abs(position_pct) * 100) + "%";
            } else {
                decision = "HOLD";
            }
        }
        
        if (std::abs(position_pct) > 0.001) {
            total_trades++;
            double pnl = position_pct * capital * returns[i];
            total_pnl += pnl;
            capital += pnl;
            if (pnl > 0) winning_trades++;
        }
        
        max_capital = std::max(max_capital, capital);
        min_capital = std::min(min_capital, capital);
        capital_history.push_back(capital);
        
        if (trade_decisions.size() < 10) {
            trade_decisions.push_back(decision);
        }
    }
    
    double total_return = (capital - 100000.0) / 100000.0;
    double max_drawdown = (max_capital - min_capital) / max_capital;
    
    double annualized_return = total_return * 252.0 / signals.size();
    
    double avg_daily_return = total_return / signals.size();
    double daily_variance = 0;
    for (size_t i = 1; i < capital_history.size(); ++i) {
        double daily_ret = (capital_history[i] - capital_history[i-1]) / capital_history[i-1];
        daily_variance += (daily_ret - avg_daily_return) * (daily_ret - avg_daily_return);
    }
    double daily_std = std::sqrt(daily_variance / (capital_history.size() - 1));
    double sharpe_ratio = daily_std > 0 ? avg_daily_return / daily_std * std::sqrt(252) : 0;
    
    double win_rate = total_trades > 0 ? winning_trades / static_cast<double>(total_trades) : 0;
    
    return {100000.0, capital, total_return, annualized_return, max_drawdown, 
            sharpe_ratio, total_trades, win_rate, trade_decisions};
}

}

namespace MLPipeline {

std::vector<std::map<std::string, double>> select_rows(const std::vector<std::map<std::string, double>>& X, const std::vector<size_t>& idxs) {
    std::vector<std::map<std::string, double>> out;
    out.reserve(idxs.size());
    for (auto i : idxs) out.push_back(X[i]);
    return out;
}
std::vector<int> select_rows(const std::vector<int>& y, const std::vector<size_t>& idxs) {
    std::vector<int> out;
    out.reserve(idxs.size());
    for (auto i : idxs) out.push_back(y[i]);
    return out;
}
std::vector<double> select_rows(const std::vector<double>& v, const std::vector<size_t>& idxs) {
    std::vector<double> out;
    out.reserve(idxs.size());
    for (auto i : idxs) out.push_back(v[i]);
    return out;
}

PipelineResult runPipeline(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<int>& y,
    const std::vector<double>& returns,
    const PipelineConfig& config
) {
    std::vector<std::map<std::string, double>> X_clean;
    std::vector<int> y_clean;
    std::vector<double> returns_clean;
    for (size_t i = 0; i < X.size(); ++i) {
        bool valid = true;
        for (const auto& kv : X[i]) {
            if (std::isnan(kv.second) || std::isinf(kv.second)) {
                valid = false;
                break;
            }
        }
        if (valid) {
            X_clean.push_back(X[i]);
            y_clean.push_back(y[i]);
            returns_clean.push_back(returns[i]);
        }
    }

    std::vector<size_t> train_idx, val_idx, test_idx;
    if (config.split_type == Chronological) {
        size_t N = X_clean.size();
        size_t n_train = size_t(N * config.train_ratio);
        size_t n_val = size_t(N * config.val_ratio);
        size_t n_test = N - n_train - n_val;
        for (size_t i = 0; i < n_train; ++i) train_idx.push_back(i);
        for (size_t i = n_train; i < n_train + n_val; ++i) val_idx.push_back(i);
        for (size_t i = n_train + n_val; i < N; ++i) test_idx.push_back(i);
    } else {
        auto folds = MLSplitUtils::purgedKFoldSplit(X_clean.size(), config.n_splits, config.embargo);
        if (!folds.empty()) {
            train_idx = folds[0].train_indices;
            val_idx = folds[0].val_indices;
            test_idx = folds.back().val_indices;
        }
    }

    auto to_float_matrix = [](const std::vector<std::map<std::string, double>>& X) {
        std::vector<std::vector<float>> Xf;
        for (const auto& row : X) {
            std::vector<float> v;
            for (const auto& kv : row) v.push_back(static_cast<float>(kv.second));
            Xf.push_back(v);
        }
        return Xf;
    };
    auto to_float_vec = [](const std::vector<int>& y) {
        std::vector<float> yf(y.begin(), y.end());
        return yf;
    };

    auto X_train_f = to_float_matrix(select_rows(X_clean, train_idx));
    auto y_train_f = to_float_vec(select_rows(y_clean, train_idx));
    
    // Use validation set for predictions instead of test set
    auto X_pred_f = val_idx.empty() ? to_float_matrix(select_rows(X_clean, test_idx)) : to_float_matrix(select_rows(X_clean, val_idx));
    auto y_pred_true = val_idx.empty() ? select_rows(y_clean, test_idx) : select_rows(y_clean, val_idx);
    auto returns_pred = val_idx.empty() ? select_rows(returns_clean, test_idx) : select_rows(returns_clean, val_idx);

    XGBoostModel model;
    model.fit(X_train_f, y_train_f, config.n_rounds, config.max_depth, config.nthread, config.objective);

    std::vector<int> y_pred = model.predict(X_pred_f);
    std::vector<float> y_prob = model.predict_proba(X_pred_f);

    std::vector<double> signals;
    bool is_hard_barrier = (config.objective == "binary:logistic");
    
    if (is_hard_barrier) {
        for (int pred : y_pred) {
            signals.push_back(static_cast<double>(pred));
        }
    } else {
        for (float prob : y_prob) {
            signals.push_back(static_cast<double>(prob));
        }
    }
    
    PortfolioSimulation portfolio = simulate_portfolio(signals, returns_pred, is_hard_barrier);

    std::vector<double> y_prob_d(y_prob.begin(), y_prob.end());
    std::map<std::string, double> empty_importances;
    return {y_pred, y_prob_d, empty_importances, portfolio};
}

PipelineResult runPipelineWithTuning(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<int>& y,
    const std::vector<double>& returns,
    PipelineConfig config
) {
    std::vector<std::map<std::string, double>> X_clean;
    std::vector<int> y_clean;
    std::vector<double> returns_clean;
    for (size_t i = 0; i < X.size(); ++i) {
        bool valid = true;
        for (const auto& kv : X[i]) {
            if (std::isnan(kv.second) || std::isinf(kv.second)) {
                valid = false;
                break;
            }
        }
        if (valid) {
            X_clean.push_back(X[i]);
            y_clean.push_back(y[i]);
            returns_clean.push_back(returns[i]);
        }
    }

    std::vector<size_t> train_idx, val_idx, test_idx;
    if (config.split_type == Chronological) {
        size_t N = X_clean.size();
        size_t n_train = size_t(N * config.train_ratio);
        size_t n_val = size_t(N * config.val_ratio);
        size_t n_test = N - n_train - n_val;
        for (size_t i = 0; i < n_train; ++i) train_idx.push_back(i);
        for (size_t i = n_train; i < n_train + n_val; ++i) val_idx.push_back(i);
        for (size_t i = n_train + n_val; i < N; ++i) test_idx.push_back(i);
    } else {
        auto folds = MLSplitUtils::purgedKFoldSplit(X_clean.size(), config.n_splits, config.embargo);
        if (!folds.empty()) {
            train_idx = folds[0].train_indices;
            val_idx = folds[0].val_indices;
            test_idx = folds.back().val_indices;
        }
    }

    auto y_val = select_rows(y_clean, val_idx);
    
    std::vector<int> n_rounds_grid = {10, 20, 50};
    std::vector<int> max_depth_grid = {3, 5, 7};
    std::vector<int> nthread_grid = {2, 4};
    std::vector<std::string> objective_grid = {"binary:logistic"};
    
    double best_score = -1e9;
    PipelineResult best_result;
    
    for (int n_rounds : n_rounds_grid) {
        for (int max_depth : max_depth_grid) {
            for (int nthread : nthread_grid) {
                for (const std::string& obj : objective_grid) {
                    config.n_rounds = n_rounds;
                    config.max_depth = max_depth;
                    config.nthread = nthread;
                    config.objective = obj;
                    
                    auto to_float_matrix = [](const std::vector<std::map<std::string, double>>& X) {
                        std::vector<std::vector<float>> Xf;
                        for (const auto& row : X) {
                            std::vector<float> v;
                            for (const auto& kv : row) v.push_back(static_cast<float>(kv.second));
                            Xf.push_back(v);
                        }
                        return Xf;
                    };
                    auto to_float_vec = [](const std::vector<int>& y) {
                        std::vector<float> yf(y.begin(), y.end());
                        return yf;
                    };

                    auto X_train_f = to_float_matrix(select_rows(X_clean, train_idx));
                    auto y_train_f = to_float_vec(select_rows(y_clean, train_idx));
                    auto X_val_f = to_float_matrix(select_rows(X_clean, val_idx));

                    XGBoostModel model;
                    model.fit(X_train_f, y_train_f, config.n_rounds, config.max_depth, config.nthread, config.objective);
                    
                    std::vector<int> y_pred_val = model.predict(X_val_f);
                    
                    double score = calculate_f1_score(y_val, y_pred_val);
                    if (score > best_score) {
                        best_score = score;
                        best_result = runPipeline(X_clean, y_clean, returns_clean, config);
                    }
                }
            }
        }
    }
    return best_result;
}

RegressionPipelineResult runPipelineRegression(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& returns,
    const PipelineConfig& config
) {
    std::vector<std::map<std::string, double>> X_clean;
    std::vector<double> y_clean;
    std::vector<double> returns_clean;
    
    for (size_t i = 0; i < X.size(); ++i) {
        bool valid = true;
        for (const auto& kv : X[i]) {
            if (std::isnan(kv.second) || std::isinf(kv.second)) {
                valid = false;
                break;
            }
        }
        if (valid) {
            X_clean.push_back(X[i]);
            y_clean.push_back(y[i]);
            returns_clean.push_back(returns[i]);
        }
    }

    std::vector<size_t> train_idx, val_idx, test_idx;
    if (config.split_type == Chronological) {
        size_t N = X_clean.size();
        size_t n_train = size_t(N * config.train_ratio);
        size_t n_val = size_t(N * config.val_ratio);
        size_t n_test = N - n_train - n_val;
        for (size_t i = 0; i < n_train; ++i) train_idx.push_back(i);
        for (size_t i = n_train; i < n_train + n_val; ++i) val_idx.push_back(i);
        for (size_t i = n_train + n_val; i < N; ++i) test_idx.push_back(i);
    } else {
        auto folds = MLSplitUtils::purgedKFoldSplit(X_clean.size(), config.n_splits, config.embargo);
        if (!folds.empty()) {
            train_idx = folds[0].train_indices;
            val_idx = folds[0].val_indices;
            test_idx = folds.back().val_indices;
        }
    }

    auto to_float_matrix = [](const std::vector<std::map<std::string, double>>& X) {
        std::vector<std::vector<float>> Xf;
        for (const auto& row : X) {
            std::vector<float> v;
            for (const auto& kv : row) v.push_back(static_cast<float>(kv.second));
            Xf.push_back(v);
        }
        return Xf;
    };
    auto to_float_vec_double = [](const std::vector<double>& y) {
        std::vector<float> yf(y.begin(), y.end());
        return yf;
    };

    auto X_train_f = to_float_matrix(select_rows(X_clean, train_idx));
    auto y_train_f = to_float_vec_double(select_rows(y_clean, train_idx));
    
    // Use validation set for predictions instead of test set
    auto X_pred_f = val_idx.empty() ? to_float_matrix(select_rows(X_clean, test_idx)) : to_float_matrix(select_rows(X_clean, val_idx));
    auto y_pred_true = val_idx.empty() ? select_rows(y_clean, test_idx) : select_rows(y_clean, val_idx);
    auto returns_pred = val_idx.empty() ? select_rows(returns_clean, test_idx) : select_rows(returns_clean, val_idx);

    XGBoostModel model;
    model.fit(X_train_f, y_train_f, config.n_rounds, config.max_depth, config.nthread, config.objective);

    std::vector<float> y_pred_f = model.predict_proba(X_pred_f);
    std::vector<double> y_pred_double(y_pred_f.begin(), y_pred_f.end());

    PortfolioSimulation portfolio = simulate_portfolio(y_pred_double, returns_pred, false);
    
    std::vector<double> uncertainties(y_pred_double.size(), 0.0);
    std::map<std::string, double> empty_importances;
    return {y_pred_double, uncertainties, empty_importances, portfolio};
}

RegressionPipelineResult runPipelineRegressionWithTuning(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<double>& y,
    const std::vector<double>& returns,
    PipelineConfig config
) {
    std::vector<std::map<std::string, double>> X_clean;
    std::vector<double> y_clean;
    std::vector<double> returns_clean;
    
    for (size_t i = 0; i < X.size(); ++i) {
        bool valid = true;
        for (const auto& kv : X[i]) {
            if (std::isnan(kv.second) || std::isinf(kv.second)) {
                valid = false;
                break;
            }
        }
        if (valid) {
            X_clean.push_back(X[i]);
            y_clean.push_back(y[i]);
            returns_clean.push_back(returns[i]);
        }
    }

    std::vector<size_t> train_idx, val_idx, test_idx;
    if (config.split_type == Chronological) {
        size_t N = X_clean.size();
        size_t n_train = size_t(N * config.train_ratio);
        size_t n_val = size_t(N * config.val_ratio);
        size_t n_test = N - n_train - n_val;
        for (size_t i = 0; i < n_train; ++i) train_idx.push_back(i);
        for (size_t i = n_train; i < n_train + n_val; ++i) val_idx.push_back(i);
        for (size_t i = n_train + n_val; i < N; ++i) test_idx.push_back(i);
    } else {
        auto folds = MLSplitUtils::purgedKFoldSplit(X_clean.size(), config.n_splits, config.embargo);
        if (!folds.empty()) {
            train_idx = folds[0].train_indices;
            val_idx = folds[0].val_indices;
            test_idx = folds.back().val_indices;
        }
    }

    auto y_val = select_rows(y_clean, val_idx);
    
    std::vector<int> n_rounds_grid = {100, 200, 500, 800, 1000};
    std::vector<int> max_depth_grid = {2, 3, 4};
    std::vector<int> nthread_grid = {4};                         
    std::vector<std::string> objective_grid = {"reg:squarederror"};
    
    double best_score = -1e9;
    RegressionPipelineResult best_result;
    
    for (int n_rounds : n_rounds_grid) {
        for (int max_depth : max_depth_grid) {
            for (int nthread : nthread_grid) {
                for (const std::string& obj : objective_grid) {
                    config.n_rounds = n_rounds;
                    config.max_depth = max_depth;
                    config.nthread = nthread;
                    config.objective = obj;
                    
                    auto to_float_matrix = [](const std::vector<std::map<std::string, double>>& X) {
                        std::vector<std::vector<float>> Xf;
                        for (const auto& row : X) {
                            std::vector<float> v;
                            for (const auto& kv : row) v.push_back(static_cast<float>(kv.second));
                            Xf.push_back(v);
                        }
                        return Xf;
                    };
                    auto to_float_vec_double = [](const std::vector<double>& y) {
                        std::vector<float> yf(y.begin(), y.end());
                        return yf;
                    };

                    auto X_train_f = to_float_matrix(select_rows(X_clean, train_idx));
                    auto y_train_f = to_float_vec_double(select_rows(y_clean, train_idx));
                    auto X_val_f = to_float_matrix(select_rows(X_clean, val_idx));

                    XGBoostModel model;
                    model.fit(X_train_f, y_train_f, config.n_rounds, config.max_depth, config.nthread, config.objective);
                    
                    std::vector<float> y_pred_val_f = model.predict_proba(X_val_f);
                    std::vector<double> y_pred_val(y_pred_val_f.begin(), y_pred_val_f.end());
                    
                    double score = calculate_r2_score(y_val, y_pred_val);
                    if (score > best_score) {
                        best_score = score;
                        best_result = runPipelineRegression(X_clean, y_clean, returns_clean, config);
                    }
                }
            }
        }
    }
    return best_result;
}

}
