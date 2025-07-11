#include "XGBoostModel.h"
#include <xgboost/c_api.h>
#include <cassert>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <set>

namespace MLPipeline {

XGBoostModel::XGBoostModel() = default;

XGBoostModel::~XGBoostModel() { 
    clear(); 
}

XGBoostModel::XGBoostModel(XGBoostModel&& other) noexcept 
    : booster_(other.booster_), n_features_(other.n_features_), 
      feature_names_(std::move(other.feature_names_)), config_(other.config_), trained_(other.trained_) {
    other.booster_ = nullptr;
    other.booster_ = nullptr;
    other.n_features_ = 0;
    other.trained_ = false;
}

XGBoostModel& XGBoostModel::operator=(XGBoostModel&& other) noexcept {
    if (this != &other) {
        clear();
        booster_ = other.booster_;
        n_features_ = other.n_features_;
        feature_names_ = std::move(other.feature_names_);
        config_ = other.config_;
        trained_ = other.trained_;
        
        other.booster_ = nullptr;
        other.n_features_ = 0;
        other.trained_ = false;
    }
    return *this;
}

void XGBoostModel::clear() { 
    free_booster(); 
    trained_ = false;
    n_features_ = 0;
    feature_names_.clear();
    label_mapping_.clear();
    reverse_label_mapping_.clear();
}

void XGBoostModel::free_booster() {
    if (booster_) {
        XGBoosterFree(static_cast<BoosterHandle>(booster_));
        booster_ = nullptr;
    }
}

bool XGBoostModel::is_trained() const {
    return trained_ && booster_ != nullptr;
}

void XGBoostModel::validate_input_dimensions(const std::vector<std::vector<float>>& X) const {
    if (X.empty()) {
        throw std::invalid_argument("Input feature matrix cannot be empty");
    }
    
    if (trained_ && n_features_ > 0) {
        for (const auto& row : X) {
            if (static_cast<int>(row.size()) != n_features_) {
                throw std::invalid_argument("Input feature dimensions do not match training dimensions. Expected: " + 
                                          std::to_string(n_features_) + ", got: " + std::to_string(row.size()));
            }
        }
    }
}

void XGBoostModel::set_xgboost_parameters(const XGBoostConfig& config) {
    int ret;
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "objective", config.objective.c_str());
    if (ret != 0) throw std::runtime_error("Failed to set objective parameter");
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "max_depth", std::to_string(config.max_depth).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set max_depth parameter");
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "nthread", std::to_string(config.nthread).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set nthread parameter");
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "learning_rate", std::to_string(config.learning_rate).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set learning_rate parameter");
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "subsample", std::to_string(config.subsample).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set subsample parameter");
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "colsample_bytree", std::to_string(config.colsample_bytree).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set colsample_bytree parameter");
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "reg_alpha", std::to_string(config.reg_alpha).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set reg_alpha parameter");
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "reg_lambda", std::to_string(config.reg_lambda).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set reg_lambda parameter");
    
    ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "min_child_weight", std::to_string(config.min_child_weight).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set min_child_weight parameter");
    
    if (config.num_class > 0 && (config.objective.find("multi:") == 0)) {
        ret = XGBoosterSetParam(static_cast<BoosterHandle>(booster_), "num_class", std::to_string(config.num_class).c_str());
        if (ret != 0) throw std::runtime_error("Failed to set num_class parameter");
    }
}

void XGBoostModel::fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y, const XGBoostConfig& config) {
    if (X.empty() || y.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    
    if (X.size() != y.size()) {
        throw std::invalid_argument("Feature matrix and target vector must have the same number of samples");
    }
    
    std::cout << "[DEBUG] XGBoostModel::fit called" << std::endl;
    std::cout << "  - Samples: " << X.size() << std::endl;
    std::cout << "  - Features: " << (X.empty() ? 0 : X[0].size()) << std::endl;
    std::cout << "  - Objective: " << config.objective << std::endl;
    std::cout << "  - n_rounds: " << config.n_rounds << std::endl;
    
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            if (std::isnan(X[i][j])) nan_count++;
            if (std::isinf(X[i][j])) inf_count++;
        }
        if (std::isnan(y[i]) || std::isinf(y[i])) {
            std::cout << "  - WARNING: Invalid label at index " << i << ": " << y[i] << std::endl;
        }
    }
    
    std::cout << "  - NaN features: " << nan_count << std::endl;
    std::cout << "  - Inf features: " << inf_count << std::endl;
    
    std::cout << "  - Label range check:" << std::endl;
    float min_label = *std::min_element(y.begin(), y.end());
    float max_label = *std::max_element(y.begin(), y.end());
    std::cout << "    Min label: " << min_label << std::endl;
    std::cout << "    Max label: " << max_label << std::endl;
    
    std::set<float> unique_labels(y.begin(), y.end());
    std::cout << "    Unique labels: ";
    for (float label : unique_labels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;
    
    XGBoostConfig adjusted_config = config;
    std::vector<float> adjusted_y = y;
    
    if (unique_labels.size() > 2 && config.objective == "binary:logistic") {
        adjusted_config.objective = "multi:softmax";
        std::cout << "  - WARNING: Detected " << unique_labels.size() << " classes but objective is binary:logistic" << std::endl;
        std::cout << "  - Switching to multi:softmax objective" << std::endl;
        
        std::vector<float> sorted_labels(unique_labels.begin(), unique_labels.end());
        std::sort(sorted_labels.begin(), sorted_labels.end());
        
        std::cout << "  - Internal mapping: ";
        for (size_t i = 0; i < sorted_labels.size(); ++i) {
            std::cout << sorted_labels[i] << "->" << i << " ";
        }
        std::cout << std::endl;
        
        label_mapping_.clear();
        reverse_label_mapping_.clear();
        for (size_t i = 0; i < sorted_labels.size(); ++i) {
            label_mapping_[sorted_labels[i]] = i;
            reverse_label_mapping_[i] = sorted_labels[i];
        }
        
        for (size_t i = 0; i < adjusted_y.size(); ++i) {
            adjusted_y[i] = label_mapping_[y[i]];
        }
        
        adjusted_config.num_class = static_cast<int>(unique_labels.size());
    }
    
    std::cout << "  - Using objective: " << adjusted_config.objective << std::endl;
    std::cout << "  - Starting XGBoost training..." << std::endl;
    
    free_booster();
    trained_ = false;
    n_features_ = 0;
    feature_names_.clear();
    
    int n_samples = static_cast<int>(X.size());
    n_features_ = static_cast<int>(X[0].size());
    config_ = adjusted_config;
    
    for (const auto& row : X) {
        if (static_cast<int>(row.size()) != n_features_) {
            throw std::invalid_argument("All feature vectors must have the same dimensions");
        }
    }
    
    std::vector<float> flat_X;
    flat_X.reserve(n_samples * n_features_);
    for (const auto& row : X) {
        flat_X.insert(flat_X.end(), row.begin(), row.end());
    }
    
    DMatrixHandle dtrain;
    int ret = XGDMatrixCreateFromMat(flat_X.data(), n_samples, n_features_, -1, &dtrain);
    if (ret != 0) {
        throw std::runtime_error("Failed to create XGBoost DMatrix");
    }
    
    try {
        ret = XGDMatrixSetFloatInfo(dtrain, "label", adjusted_y.data(), n_samples); 
        if (ret != 0) throw std::runtime_error("Failed to set labels");
        
        BoosterHandle temp_booster;
        ret = XGBoosterCreate(&dtrain, 1, &temp_booster);
        if (ret != 0) throw std::runtime_error("Failed to create XGBoost booster");
        booster_ = temp_booster;
        
        set_xgboost_parameters(adjusted_config); 
        
        std::cout << "  - Starting XGBoost training..." << std::endl;
        
        for (int iter = 0; iter < config.n_rounds; ++iter) {
            ret = XGBoosterUpdateOneIter(static_cast<BoosterHandle>(booster_), iter, dtrain);
            if (ret != 0) {
                const char* error_msg = XGBGetLastError();
                std::string full_error = "Training failed at iteration " + std::to_string(iter) + 
                                       ". XGBoost error: " + std::string(error_msg);
                std::cout << "  - ERROR: " << full_error << std::endl;
                throw std::runtime_error(full_error);
            }
        }
        
        std::cout << "  - Training completed successfully!" << std::endl;
        
        trained_ = true;
        
    } catch (...) {
        XGDMatrixFree(dtrain);
        throw;
    }
    
    XGDMatrixFree(dtrain);
}

std::vector<int> XGBoostModel::predict(const std::vector<std::vector<float>>& X) const {
    if (!is_trained()) {
        throw std::runtime_error("Model must be trained before making predictions");
    }
    
    std::vector<float> raw_predictions = predict_raw(X);
    std::vector<int> predictions;
    predictions.reserve(raw_predictions.size());
    
    if (config_.objective == "multi:softmax") {
        for (float pred : raw_predictions) {
            int xgb_index = static_cast<int>(pred);
            if (reverse_label_mapping_.count(xgb_index)) {
                predictions.push_back(static_cast<int>(reverse_label_mapping_.at(xgb_index)));
            } else {
                predictions.push_back(xgb_index);
            }
        }
    } else {
        for (float prob : raw_predictions) {
            predictions.push_back((prob > config_.binary_threshold) ? 1 : 0);
        }
    }
    
    return predictions;
}

std::vector<float> XGBoostModel::predict_raw(const std::vector<std::vector<float>>& X) const {
    if (!is_trained()) {
        throw std::runtime_error("Model must be trained before making predictions");
    }
    
    validate_input_dimensions(X);
    
    int n_samples = static_cast<int>(X.size());
    
    std::vector<float> flat_X;
    flat_X.reserve(n_samples * n_features_);
    for (const auto& row : X) {
        flat_X.insert(flat_X.end(), row.begin(), row.end());
    }
    
    DMatrixHandle dtest;
    int ret = XGDMatrixCreateFromMat(flat_X.data(), n_samples, n_features_, -1, &dtest);
    if (ret != 0) {
        throw std::runtime_error("Failed to create prediction DMatrix");
    }
    
    std::vector<float> predictions;
    try {
        bst_ulong out_len;
        const float* out_result;
        
        ret = XGBoosterPredict(static_cast<BoosterHandle>(booster_), dtest, 0, 0, 0, &out_len, &out_result);
        if (ret != 0) {
            throw std::runtime_error("Prediction failed");
        }
        
        predictions.assign(out_result, out_result + out_len);
        
    } catch (...) {
        XGDMatrixFree(dtest);
        throw;
    }
    
    XGDMatrixFree(dtest);
    return predictions;
}

void XGBoostModel::set_feature_names(const std::vector<std::string>& names) {
    if (trained_ && !names.empty() && static_cast<int>(names.size()) != n_features_) {
        throw std::invalid_argument("Number of feature names must match number of features");
    }
    feature_names_ = names;
}

void XGBoostModel::fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y, 
                      int n_rounds, int max_depth, int nthread, const std::string& objective) {
    XGBoostConfig config;
    config.n_rounds = n_rounds;
    config.max_depth = max_depth;
    config.nthread = nthread;
    config.objective = objective;
    
    fit(X, y, config);
}

std::vector<float> XGBoostModel::predict_proba(const std::vector<std::vector<float>>& X) const {
    return predict_raw(X);
}

}
