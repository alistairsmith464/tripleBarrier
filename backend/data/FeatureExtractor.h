#pragma once
#include <vector>
#include <map>
#include <string>
#include <set>
#include "PreprocessedRow.h"
#include "LabeledEvent.h"

class FeatureExtractor {
public:
    struct FeatureExtractionResult {
        std::vector<std::map<std::string, double>> features;
        std::vector<int> labels;
        std::vector<double> labels_double;  // For TTBM regression
        std::vector<double> returns;
    };

    // Feature mapping from user-friendly names to backend IDs
    static std::map<std::string, std::string> getFeatureMapping();

    // Extract features for classification (hard barriers)
    static FeatureExtractionResult extractFeaturesForClassification(
        const std::set<std::string>& selectedFeatures,
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents
    );

    // Extract features for regression (TTBM)
    static FeatureExtractionResult extractFeaturesForRegression(
        const std::set<std::string>& selectedFeatures,
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents
    );

private:
    static std::vector<int> findEventIndices(
        const std::vector<PreprocessedRow>& rows,
        const std::vector<LabeledEvent>& labeledEvents
    );

    static std::map<std::string, double> enhanceFeatures(
        const std::map<std::string, double>& baseFeatures,
        const PreprocessedRow& row
    );

    static void applyRobustScaling(std::vector<std::map<std::string, double>>& features);
};
