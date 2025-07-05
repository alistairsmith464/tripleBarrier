#include "MLPipeline.h"
#include "../data/DataCleaningUtils.h"
#include "MLSplits.h"
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace MLPipeline;

namespace {
// Helper: compute classification metrics
MetricsResult compute_classification_metrics(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1 && y_pred[i] == 1) tp++;
        if (y_true[i] == 0 && y_pred[i] == 0) tn++;
        if (y_true[i] == 0 && y_pred[i] == 1) fp++;
        if (y_true[i] == 1 && y_pred[i] == 0) fn++;
    }
    double accuracy = (tp + tn) / static_cast<double>(y_true.size());
    double precision = tp + fp == 0 ? 0 : tp / static_cast<double>(tp + fp);
    double recall = tp + fn == 0 ? 0 : tp / static_cast<double>(tp + fn);
    double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
    MetricsResult result{accuracy, precision, recall, f1, 0, 0, 0, tp, tn, fp, fn, static_cast<int>(y_true.size())};
    return result;
}

// Helper: compute financial metrics
void compute_financial_metrics(const std::vector<int>& y_true, const std::vector<int>& y_pred, const std::vector<double>& returns, MetricsResult& metrics) {
    double total_return = 0, total_squared = 0;
    int hits = 0, n = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] == 1) {
            total_return += returns[i];
            total_squared += returns[i] * returns[i];
            if (returns[i] > 0) hits++;
            n++;
        }
    }
    metrics.avg_return = n ? total_return / n : 0;
    double stddev = n > 1 ? std::sqrt((total_squared - (total_return * total_return) / n) / (n - 1)) : 0;
    metrics.sharpe_ratio = stddev > 0 ? metrics.avg_return / stddev : 0;
    metrics.hit_ratio = n ? hits / static_cast<double>(n) : 0;
}

// Helper: extract feature matrix for given indices
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
}

PipelineResult MLPipeline::runPipeline(
    const std::vector<std::map<std::string, double>>& X,
    const std::vector<int>& y,
    const std::vector<double>& returns,
    const PipelineConfig& config
) {
    // 1. Clean data (remove rows with NaN/Inf) and align y/returns
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

    // 2. Split data
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

    // Convert features to float vectors for XGBoost
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
    auto X_test_f = to_float_matrix(select_rows(X_clean, test_idx));
    auto y_test = select_rows(y_clean, test_idx);
    auto returns_test = select_rows(returns_clean, test_idx);

    // 3. Train XGBoost model
    XGBoostModel model;
    model.fit(X_train_f, y_train_f, config.n_rounds, config.max_depth, config.nthread, config.objective);

    // 4. Predict on test set
    std::vector<int> y_pred = model.predict(X_test_f);
    std::vector<float> y_prob = model.predict_proba(X_test_f);

    // 5. Compute metrics
    MetricsResult metrics = compute_classification_metrics(y_test, y_pred);
    compute_financial_metrics(y_test, y_pred, returns_test, metrics);

    // 6. Feature importances
    auto importances = model.feature_importances();
    std::map<std::string, double> importances_d(importances.begin(), importances.end());

    // Convert y_prob to double
    std::vector<double> y_prob_d(y_prob.begin(), y_prob.end());
    return {y_pred, y_prob_d, importances_d, metrics};
}
