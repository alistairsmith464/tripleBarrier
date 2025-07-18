#pragma once
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <algorithm>
#include <stdexcept>

namespace MLSplitUtils {
    struct SplitResult {
        std::vector<std::map<std::string, double>> X_train, X_val, X_test;
        std::vector<int> y_train, y_val, y_test;
    };

    struct PurgedFold {
        std::vector<size_t> train_indices;
        std::vector<size_t> val_indices;
    };

    inline SplitResult chronologicalSplit(
        const std::vector<std::map<std::string, double>>& X,
        const std::vector<int>& y,
        double train_ratio = 0.6,
        double val_ratio = 0.2,
        double test_ratio = 0.2
    ) {
        size_t N = X.size();
        size_t n_train = size_t(N * train_ratio);
        size_t n_val = size_t(N * val_ratio);
        size_t n_test = N - n_train - n_val;
        SplitResult result;
        result.X_train.assign(X.begin(), X.begin() + n_train);
        result.y_train.assign(y.begin(), y.begin() + n_train);
        result.X_val.assign(X.begin() + n_train, X.begin() + n_train + n_val);
        result.y_val.assign(y.begin() + n_train, y.begin() + n_train + n_val);
        result.X_test.assign(X.begin() + n_train + n_val, X.end());
        result.y_test.assign(y.begin() + n_train + n_val, y.end());
        return result;
    }

    inline std::vector<PurgedFold> purgedKFoldSplit(
        size_t N,
        int n_splits = 5,
        int embargo = 0
    ) {
        if (N == 0 || n_splits <= 0) {
            throw std::invalid_argument("Invalid parameters for purged K-fold split");
        }
        
        std::vector<PurgedFold> folds;
        size_t fold_size = N / n_splits;
        
        for (int k = 0; k < n_splits; ++k) {
            size_t val_start = k * fold_size;
            size_t val_end = (k == n_splits - 1) ? N : (val_start + fold_size);
            
            std::vector<size_t> val_indices, train_indices;
            
            for (size_t i = val_start; i < val_end; ++i) {
                val_indices.push_back(i);
            }
            
            for (size_t i = 0; i < N; ++i) {
                if (i >= val_start && i < val_end) {
                    continue;
                }
                
                bool in_embargo = false;
                if (embargo > 0) {
                    if (i >= val_start - embargo && i < val_start) {
                        in_embargo = true;
                    }
                    
                    if (i >= val_end && i < val_end + embargo) {
                        in_embargo = true;
                    }
                }
                
                if (!in_embargo) {
                    train_indices.push_back(i);
                }
            }
            
            folds.push_back({train_indices, val_indices});
        }
        
        return folds;
    }
}
