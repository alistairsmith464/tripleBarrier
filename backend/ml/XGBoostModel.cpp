#include "XGBoostModel.h"
#include <cassert>
#include <cstring>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>

namespace MLPipeline {

XGBoostModel::XGBoostModel() = default;

XGBoostModel::~XGBoostModel() { 
    clear(); 
}

XGBoostModel::XGBoostModel(XGBoostModel&& other) noexcept 
    : booster_(other.booster_), n_features_(other.n_features_), 
      feature_names_(std::move(other.feature_names_)), config_(other.config_), trained_(other.trained_) {
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
}

void XGBoostModel::free_booster() {
    if (booster_) {
        XGBoosterFree(booster_);
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
    
    ret = XGBoosterSetParam(booster_, "objective", config.objective.c_str());
    if (ret != 0) throw std::runtime_error("Failed to set objective parameter");
    
    ret = XGBoosterSetParam(booster_, "max_depth", std::to_string(config.max_depth).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set max_depth parameter");
    
    ret = XGBoosterSetParam(booster_, "nthread", std::to_string(config.nthread).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set nthread parameter");
    
    ret = XGBoosterSetParam(booster_, "learning_rate", std::to_string(config.learning_rate).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set learning_rate parameter");
    
    ret = XGBoosterSetParam(booster_, "subsample", std::to_string(config.subsample).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set subsample parameter");
    
    ret = XGBoosterSetParam(booster_, "colsample_bytree", std::to_string(config.colsample_bytree).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set colsample_bytree parameter");
    
    ret = XGBoosterSetParam(booster_, "reg_alpha", std::to_string(config.reg_alpha).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set reg_alpha parameter");
    
    ret = XGBoosterSetParam(booster_, "reg_lambda", std::to_string(config.reg_lambda).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set reg_lambda parameter");
    
    ret = XGBoosterSetParam(booster_, "min_child_weight", std::to_string(config.min_child_weight).c_str());
    if (ret != 0) throw std::runtime_error("Failed to set min_child_weight parameter");
}

void XGBoostModel::fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y, const XGBoostConfig& config) {
    if (X.empty() || y.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    
    if (X.size() != y.size()) {
        throw std::invalid_argument("Feature matrix and target vector must have the same number of samples");
    }
    
    clear();
    
    int n_samples = static_cast<int>(X.size());
    n_features_ = static_cast<int>(X[0].size());
    config_ = config;
    
    // Validate feature dimensions are consistent
    for (const auto& row : X) {
        if (static_cast<int>(row.size()) != n_features_) {
            throw std::invalid_argument("All feature vectors must have the same dimensions");
        }
    }
    
    // Flatten feature matrix
    std::vector<float> flat_X;
    flat_X.reserve(n_samples * n_features_);
    for (const auto& row : X) {
        flat_X.insert(flat_X.end(), row.begin(), row.end());
    }
    
    // Create DMatrix with proper error handling
    DMatrixHandle dtrain;
    int ret = XGDMatrixCreateFromMat(flat_X.data(), n_samples, n_features_, -1, &dtrain);
    if (ret != 0) {
        throw std::runtime_error("Failed to create XGBoost DMatrix");
    }
    
    try {
        ret = XGDMatrixSetFloatInfo(dtrain, "label", y.data(), n_samples);
        if (ret != 0) throw std::runtime_error("Failed to set labels");
        
        ret = XGBoosterCreate(&dtrain, 1, &booster_);
        if (ret != 0) throw std::runtime_error("Failed to create XGBoost booster");
        
        set_xgboost_parameters(config);
        
        // Train the model
        for (int iter = 0; iter < config.n_rounds; ++iter) {
            ret = XGBoosterUpdateOneIter(booster_, iter, dtrain);
            if (ret != 0) {
                throw std::runtime_error("Training failed at iteration " + std::to_string(iter));
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
    predictions.reserve(raw_predictions.size());
    
    for (float prob : raw_predictions) {
        predictions.push_back((prob > config_.binary_threshold) ? 1 : 0);
    }
    
    return predictions;
}

std::vector<float> XGBoostModel::predict_raw(const std::vector<std::vector<float>>& X) const {
    if (!is_trained()) {
        throw std::runtime_error("Model must be trained before making predictions");
    }
    
    validate_input_dimensions(X);
    
    int n_samples = static_cast<int>(X.size());
    
    // Flatten feature matrix
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
        
        ret = XGBoosterPredict(booster_, dtest, 0, 0, 0, &out_len, &out_result);
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

void XGBoostModel::save_model(const std::string& filename) const {
    if (!is_trained()) {
        throw std::runtime_error("Cannot save untrained model");
    }
    
    int ret = XGBoosterSaveModel(booster_, filename.c_str());
    if (ret != 0) {
        throw std::runtime_error("Failed to save model to " + filename);
    }
}

void XGBoostModel::load_model(const std::string& filename) {
    clear();
    
    // Check if file exists
    std::ifstream file(filename);
    if (!file.good()) {
        throw std::runtime_error("Model file does not exist: " + filename);
    }
    file.close();
    
    int ret = XGBoosterCreate(nullptr, 0, &booster_);
    if (ret != 0) {
        throw std::runtime_error("Failed to create booster for loading");
    }
    
    ret = XGBoosterLoadModel(booster_, filename.c_str());
    if (ret != 0) {
        free_booster();
        throw std::runtime_error("Failed to load model from " + filename);
    }
    
    trained_ = true;
    // Note: n_features_ and config_ would need to be saved/loaded separately for full restoration
}

void XGBoostModel::set_feature_names(const std::vector<std::string>& names) {
    if (trained_ && !names.empty() && static_cast<int>(names.size()) != n_features_) {
        throw std::invalid_argument("Number of feature names must match number of features");
    }
    feature_names_ = names;
}

// Legacy methods for backward compatibility
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
