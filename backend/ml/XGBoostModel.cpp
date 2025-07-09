#include "XGBoostModel.h"
#include <cassert>
#include <cstring>
#include <sstream>
#include <iostream>

XGBoostModel::XGBoostModel() {}

XGBoostModel::~XGBoostModel() { clear(); }

void XGBoostModel::clear() { free_booster(); }

void XGBoostModel::free_booster() {
    if (booster_) {
        XGBoosterFree(booster_);
        booster_ = nullptr;
    }
}

void XGBoostModel::fit(const std::vector<std::vector<float>>& X, const std::vector<float>& y, int n_rounds, int max_depth, int nthread, const std::string& objective) {
    clear();
    int n_samples = (int)X.size();
    if (n_samples == 0) return;
    n_features_ = (int)X[0].size();
    std::vector<float> flat_X;
    flat_X.reserve(n_samples * n_features_);
    for (const auto& row : X) flat_X.insert(flat_X.end(), row.begin(), row.end());
    DMatrixHandle dtrain;
    XGDMatrixCreateFromMat(flat_X.data(), n_samples, n_features_, -1, &dtrain);
    XGDMatrixSetFloatInfo(dtrain, "label", y.data(), n_samples);
    XGBoosterCreate(&dtrain, 1, &booster_);
    XGBoosterSetParam(booster_, "objective", objective.c_str());
    XGBoosterSetParam(booster_, "max_depth", std::to_string(max_depth).c_str());
    XGBoosterSetParam(booster_, "nthread", std::to_string(nthread).c_str());
    
    XGBoosterSetParam(booster_, "learning_rate", "0.1"); 
    XGBoosterSetParam(booster_, "subsample", "0.8");        
    XGBoosterSetParam(booster_, "colsample_bytree", "0.8"); 
    XGBoosterSetParam(booster_, "reg_alpha", "0.1");   
    XGBoosterSetParam(booster_, "reg_lambda", "1.0");    
    XGBoosterSetParam(booster_, "min_child_weight", "1"); 
    
    for (int iter = 0; iter < n_rounds; ++iter)
        XGBoosterUpdateOneIter(booster_, iter, dtrain);
    XGDMatrixFree(dtrain);
}

std::vector<int> XGBoostModel::predict(const std::vector<std::vector<float>>& X) const {
    std::vector<float> probas = predict_proba(X);
    std::vector<int> preds(probas.size());
    for (size_t i = 0; i < probas.size(); ++i)
        preds[i] = (probas[i] > 0.5f) ? 1 : 0;
    return preds;
}

std::vector<float> XGBoostModel::predict_proba(const std::vector<std::vector<float>>& X) const {
    int n_samples = (int)X.size();
    if (!booster_ || n_samples == 0) return {};
    int n_features = n_features_;
    std::vector<float> flat_X;
    flat_X.reserve(n_samples * n_features);
    for (const auto& row : X) flat_X.insert(flat_X.end(), row.begin(), row.end());
    DMatrixHandle dtest;
    XGDMatrixCreateFromMat(flat_X.data(), n_samples, n_features, -1, &dtest);
    bst_ulong out_len;
    const float* out_result;
    XGBoosterPredict(booster_, dtest, 0, 0, 0, &out_len, &out_result);
    std::vector<float> probas(out_result, out_result + out_len);
    XGDMatrixFree(dtest);
    return probas;
}
