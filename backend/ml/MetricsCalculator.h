#pragma once
#include <vector>
#include <map>
#include <string>

namespace MLPipeline {

// Comprehensive metric calculator class
class MetricsCalculator {
public:
    
private:
    static bool validate_inputs(const std::vector<int>& y_true, const std::vector<int>& y_pred);
    static bool validate_inputs(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    static void check_binary_classification(const std::vector<int>& y_true, const std::vector<int>& y_pred);
};

}
