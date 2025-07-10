#pragma once
#include <vector>
#include <map>
#include <string>

namespace MLPipeline {

class MetricsCalculator {
public:
    // Classification metrics
    static double calculateF1Score(const std::vector<int>& y_true, const std::vector<int>& y_pred);
    static double calculateAccuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred);
    static double calculatePrecision(const std::vector<int>& y_true, const std::vector<int>& y_pred);
    static double calculateRecall(const std::vector<int>& y_true, const std::vector<int>& y_pred);
    static std::vector<std::vector<int>> calculateConfusionMatrix(const std::vector<int>& y_true, const std::vector<int>& y_pred);
    static double calculateAUCROC(const std::vector<double>& y_true, const std::vector<double>& y_prob);
    
    // Regression metrics
    static double calculateR2Score(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    static double calculateMAE(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    static double calculateRMSE(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    static double calculateMAPE(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    
private:
    static bool validate_inputs(const std::vector<int>& y_true, const std::vector<int>& y_pred);
    static bool validate_inputs(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    static void check_binary_classification(const std::vector<int>& y_true, const std::vector<int>& y_pred);
};

}
