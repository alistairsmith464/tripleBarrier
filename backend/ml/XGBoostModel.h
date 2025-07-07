#pragma once
#include <vector>
#include <string>
#include <map>
#include <xgboost/c_api.h>

class XGBoostModel {
public:
    XGBoostModel();
    ~XGBoostModel();
    void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y, int n_rounds = 10, int max_depth = 3, int nthread = 4, const std::string& objective = "binary:logistic");
    std::vector<int> predict(const std::vector<std::vector<float>>& X) const;
    std::vector<float> predict_proba(const std::vector<std::vector<float>>& X) const;
    std::vector<float> predict_regression(const std::vector<std::vector<float>>& X) const;
    std::map<std::string, float> feature_importances() const;
    void clear();
private:
    BoosterHandle booster_ = nullptr;
    int n_features_ = 0;
    std::vector<std::string> feature_names_;
    void free_booster();
};
