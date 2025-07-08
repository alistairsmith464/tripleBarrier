#pragma once
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>

namespace DataCleaningUtils {
    inline void cleanFeatureRows(std::vector<std::map<std::string, double>>& features) {
        features.erase(
            std::remove_if(features.begin(), features.end(), [](const std::map<std::string, double>& row) {
                for (const auto& kv : row) {
                    if (std::isnan(kv.second) || std::isinf(kv.second)) return true;
                }
                return false;
            }),
            features.end()
        );
    }
}
