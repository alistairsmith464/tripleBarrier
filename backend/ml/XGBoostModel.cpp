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

std::vector<float> XGBoostModel::predict_regression(const std::vector<std::vector<float>>& X) const {
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
    std::vector<float> preds(out_result, out_result + out_len);
    XGDMatrixFree(dtest);
    return preds;
}

std::map<std::string, float> XGBoostModel::feature_importances() const {
    std::map<std::string, float> importances;
    if (!booster_) return importances;
    bst_ulong out_len = 0;
    const char** model_dump = nullptr;
    // Dump the model as text
    if (XGBoosterDumpModelEx(booster_, nullptr, 0, "text", &out_len, &model_dump) != 0) {
        return importances;
    }
    // Count feature occurrences in the dump (proxy for importance)
    for (bst_ulong i = 0; i < out_len; ++i) {
        std::string dump(model_dump[i]);
        std::istringstream iss(dump);
        std::string line;
        while (std::getline(iss, line)) {
            size_t pos = line.find("[");
            if (pos != std::string::npos) {
                size_t end = line.find("<", pos);
                if (end == std::string::npos) end = line.find("]", pos);
                if (end != std::string::npos) {
                    std::string feat = line.substr(pos + 1, end - pos - 1);
                    // Feature name is before < or ]
                    size_t eq = feat.find("<");
                    if (eq != std::string::npos) feat = feat.substr(0, eq);
                    importances[feat] += 1.0f;
                }
            }
        }
    }
    // Free the model dump memory if needed (depends on XGBoost version)
    // XGBoost C API may not require explicit free for model_dump
    return importances;
}
