#include "MetricsCalculator.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <set>
#include <map>

namespace MLPipeline {

bool MetricsCalculator::validate_inputs(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }
    return true;
}

bool MetricsCalculator::validate_inputs(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }
    return true;
}

void MetricsCalculator::check_binary_classification(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    std::set<int> unique_values;
    for (int val : y_true) unique_values.insert(val);
    for (int val : y_pred) unique_values.insert(val);
    
    if (unique_values.size() > 2) {
        throw std::invalid_argument("Binary classification metrics require at most 2 unique classes");
    }
}

double MetricsCalculator::calculateF1Score(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    validate_inputs(y_true, y_pred);
    check_binary_classification(y_true, y_pred);
    
    int tp = 0, fp = 0, fn = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1 && y_pred[i] == 1) tp++;
        else if (y_true[i] == 0 && y_pred[i] == 1) fp++;
        else if (y_true[i] == 1 && y_pred[i] == 0) fn++;
    }
    
    if (tp + fp == 0 || tp + fn == 0) return 0.0;
    
    double precision = static_cast<double>(tp) / (tp + fp);
    double recall = static_cast<double>(tp) / (tp + fn);
    
    if (precision + recall == 0.0) return 0.0;
    
    return 2.0 * (precision * recall) / (precision + recall);
}

double MetricsCalculator::calculateR2Score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    validate_inputs(y_true, y_pred);
    
    double mean_true = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();
    
    double ss_res = 0.0;
    double ss_tot = 0.0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        ss_tot += (y_true[i] - mean_true) * (y_true[i] - mean_true);
    }
    
    if (ss_tot == 0.0) return 1.0;
    
    return 1.0 - (ss_res / ss_tot);
}

double MetricsCalculator::calculateAccuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    validate_inputs(y_true, y_pred);
    
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) correct++;
    }
    
    return static_cast<double>(correct) / y_true.size();
}

double MetricsCalculator::calculatePrecision(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    validate_inputs(y_true, y_pred);
    check_binary_classification(y_true, y_pred);
    
    int tp = 0, fp = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1 && y_pred[i] == 1) tp++;
        else if (y_true[i] == 0 && y_pred[i] == 1) fp++;
    }
    
    if (tp + fp == 0) return 0.0;
    return static_cast<double>(tp) / (tp + fp);
}

double MetricsCalculator::calculateRecall(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    validate_inputs(y_true, y_pred);
    check_binary_classification(y_true, y_pred);
    
    int tp = 0, fn = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1 && y_pred[i] == 1) tp++;
        else if (y_true[i] == 1 && y_pred[i] == 0) fn++;
    }
    
    if (tp + fn == 0) return 0.0;
    return static_cast<double>(tp) / (tp + fn);
}

std::vector<std::vector<int>> MetricsCalculator::calculateConfusionMatrix(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    validate_inputs(y_true, y_pred);
    
    std::set<int> unique_classes;
    for (int val : y_true) unique_classes.insert(val);
    for (int val : y_pred) unique_classes.insert(val);
    
    std::vector<int> classes(unique_classes.begin(), unique_classes.end());
    int n_classes = classes.size();
    
    std::vector<std::vector<int>> matrix(n_classes, std::vector<int>(n_classes, 0));
    
    std::map<int, int> class_to_index;
    for (int i = 0; i < n_classes; ++i) {
        class_to_index[classes[i]] = i;
    }
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        int true_idx = class_to_index[y_true[i]];
        int pred_idx = class_to_index[y_pred[i]];
        matrix[true_idx][pred_idx]++;
    }
    
    return matrix;
}

double MetricsCalculator::calculateAUCROC(const std::vector<double>& y_true, const std::vector<double>& y_prob) {
    validate_inputs(y_true, y_prob);
    
    std::vector<std::pair<double, double>> prob_true_pairs;
    for (size_t i = 0; i < y_true.size(); ++i) {
        prob_true_pairs.emplace_back(y_prob[i], y_true[i]);
    }
    
    std::sort(prob_true_pairs.begin(), prob_true_pairs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    int positive_count = std::accumulate(y_true.begin(), y_true.end(), 0);
    int negative_count = y_true.size() - positive_count;
    
    if (positive_count == 0 || negative_count == 0) return 0.5;
    
    double auc = 0.0;
    int tp = 0, fp = 0;
    
    for (const auto& pair : prob_true_pairs) {
        if (pair.second == 1.0) {
            tp++;
        } else {
            auc += tp;
            fp++;
        }
    }
    
    return auc / (positive_count * negative_count);
}

double MetricsCalculator::calculateMAE(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    validate_inputs(y_true, y_pred);
    
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum_abs_diff += std::abs(y_true[i] - y_pred[i]);
    }
    
    return sum_abs_diff / y_true.size();
}

double MetricsCalculator::calculateRMSE(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    validate_inputs(y_true, y_pred);
    
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        sum_sq_diff += diff * diff;
    }
    
    return std::sqrt(sum_sq_diff / y_true.size());
}

double MetricsCalculator::calculateMAPE(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    validate_inputs(y_true, y_pred);
    
    double sum_abs_pct_error = 0.0;
    int valid_count = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (std::abs(y_true[i]) > 1e-10) {
            sum_abs_pct_error += std::abs((y_true[i] - y_pred[i]) / y_true[i]);
            valid_count++;
        }
    }
    
    if (valid_count == 0) return 0.0;
    
    return (sum_abs_pct_error / valid_count) * 100.0;
}

}
