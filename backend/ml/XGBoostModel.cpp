#include "XGBoostModel.h"
#include "../utils/Exceptions.h"
#include "../utils/ErrorHandling.h"
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
    using namespace TripleBarrier;
    
    Validation::validateNotEmpty(X, "training_features");
    Validation::validateNotEmpty(y, "training_labels");
    Validation::validateSizeMatch(X, y, "features", "labels");
    
    if (config.n_rounds <= 0) {
        throw HyperparameterException("n_rounds must be positive", "n_rounds");
    }
    if (config.max_depth <= 0) {
        throw HyperparameterException("max_depth must be positive", "max_depth");
    }
    if (config.learning_rate <= 0.0) {
        throw HyperparameterException("learning_rate must be positive", "learning_rate");
    }
    
    if (!X.empty()) {
        size_t expected_features = X[0].size();
        for (size_t i = 1; i < X.size(); ++i) {
            if (X[i].size() != expected_features) {
                throw DataValidationException(
                    "Inconsistent feature dimensions at row " + std::to_string(i) + 
                    ": expected " + std::to_string(expected_features) + 
                    ", got " + std::to_string(X[i].size())
                );
            }
        }
    }
    
    if (X.size() != y.size()) {
        throw DataValidationException(
            "Size mismatch: features (" + std::to_string(X.size()) + 
            ") vs labels (" + std::to_string(y.size()) + ")"
        );
    }
    
    ErrorAccumulator dataErrors;
    int nan_count = 0, inf_count = 0;
    
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            if (std::isnan(X[i][j])) {
                nan_count++;
                if (dataErrors.errorCount() < 5) {
                    dataErrors.addError("NaN value in features", 
                                       "row " + std::to_string(i) + ", col " + std::to_string(j));
                }
            }
            if (std::isinf(X[i][j])) {
                inf_count++;
                if (dataErrors.errorCount() < 5) {
                    dataErrors.addError("Infinite value in features", 
                                       "row " + std::to_string(i) + ", col " + std::to_string(j));
                }
            }
        }
        if (std::isnan(y[i]) || std::isinf(y[i])) {
            dataErrors.addError("Invalid label value: " + std::to_string(y[i]), 
                               "row " + std::to_string(i));
        }
    }
    
    if (dataErrors.hasErrors()) {
        throw DataValidationException("Data quality issues detected", dataErrors.getAllErrors());
    }
    
    XGBoostConfig adjusted_config = config;
    std::vector<float> adjusted_y = y;
    std::set<float> unique_labels(y.begin(), y.end());
    if (unique_labels.size() > 2 && config.objective == "binary:logistic") {
        adjusted_config.objective = "multi:softmax";
        
        std::vector<float> sorted_labels(unique_labels.begin(), unique_labels.end());
        std::sort(sorted_labels.begin(), sorted_labels.end());
        
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
        
        for (int iter = 0; iter < config.n_rounds; ++iter) {
            ret = XGBoosterUpdateOneIter(static_cast<BoosterHandle>(booster_), iter, dtrain);
            if (ret != 0) {
                const char* error_msg = XGBGetLastError();
                std::string full_error = "Training failed at iteration " + std::to_string(iter) + 
                                       ". XGBoost error: " + std::string(error_msg);
                throw std::runtime_error(full_error);
            }
        }
        
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
    predictions.reserve(X.size());

    if (config_.objective == "multi:softmax") {
        for (float pred : raw_predictions) {
            int xgb_index = static_cast<int>(pred);
            if (reverse_label_mapping_.count(xgb_index)) {
                predictions.push_back(static_cast<int>(reverse_label_mapping_.at(xgb_index)));
            } else {
                predictions.push_back(xgb_index);
            }
        }
    } else if (config_.objective == "multi:softprob") {
        int n_classes = config_.num_class;
        int n_samples = static_cast<int>(X.size());
        for (int i = 0; i < n_samples; ++i) {
            int best_class = 0;
            float best_prob = raw_predictions[i * n_classes];
            for (int c = 1; c < n_classes; ++c) {
                float prob = raw_predictions[i * n_classes + c];
                if (prob > best_prob) {
                    best_prob = prob;
                    best_class = c;
                }
            }
            if (reverse_label_mapping_.count(best_class)) {
                predictions.push_back(static_cast<int>(reverse_label_mapping_.at(best_class)));
            } else {
                predictions.push_back(best_class);
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
