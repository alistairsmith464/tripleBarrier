#pragma once
#include <vector>
#include <string>
#include <memory>
#include <map>

namespace MLPipeline {

struct XGBoostConfig {
    int n_rounds = 20;
    int max_depth = 3;
    int nthread = 4;
    std::string objective = "binary:logistic";
    double learning_rate = 0.1;
    double subsample = 0.8;
    double colsample_bytree = 0.8;
    double reg_alpha = 0.1;
    double reg_lambda = 1.0;
    double min_child_weight = 1.0;
    double binary_threshold = 0.5;
    int num_class = 0;
};

class IMLModel {
public:
    virtual ~IMLModel() = default;
    virtual void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y, const XGBoostConfig& config) = 0;
    virtual std::vector<int> predict(const std::vector<std::vector<float>>& X) const = 0;
    virtual std::vector<float> predict_raw(const std::vector<std::vector<float>>& X) const = 0;
    virtual void clear() = 0;
    virtual bool is_trained() const = 0;
};

class XGBoostModel : public IMLModel {
public:
    XGBoostModel();
    ~XGBoostModel() override;
    
    XGBoostModel(const XGBoostModel&) = delete;
    XGBoostModel& operator=(const XGBoostModel&) = delete;
    
    XGBoostModel(XGBoostModel&& other) noexcept;
    XGBoostModel& operator=(XGBoostModel&& other) noexcept;
    
    void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y, const XGBoostConfig& config) override;
    std::vector<int> predict(const std::vector<std::vector<float>>& X) const override;
    std::vector<float> predict_raw(const std::vector<std::vector<float>>& X) const override;
    
    void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y, 
             int n_rounds = 10, int max_depth = 3, int nthread = 4, const std::string& objective = "binary:logistic");
    std::vector<float> predict_proba(const std::vector<std::vector<float>>& X) const;
    
    void clear() override;
    bool is_trained() const override;
    
    int get_num_features() const { return n_features_; }
    void set_feature_names(const std::vector<std::string>& names);
    const std::vector<std::string>& get_feature_names() const { return feature_names_; }
    
private:
    void* booster_ = nullptr;
    int n_features_ = 0;
    std::vector<std::string> feature_names_;
    XGBoostConfig config_;
    bool trained_ = false;
    
    std::map<float, int> label_mapping_;
    std::map<int, float> reverse_label_mapping_;
    
    void free_booster();
    void validate_input_dimensions(const std::vector<std::vector<float>>& X) const;
    void set_xgboost_parameters(const XGBoostConfig& config);
};

}
