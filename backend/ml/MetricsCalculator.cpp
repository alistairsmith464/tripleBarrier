#include "MetricsCalculator.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <set>

namespace MLPipeline {

// Input validation
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

}
