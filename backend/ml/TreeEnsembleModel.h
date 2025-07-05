#pragma once
#include <vector>
#include <map>
#include <string>
#include <random>
#include <algorithm>

// Simple tree ensemble (Random Forest-like) placeholder for modular ML pipeline
// Replace with XGBoost/LightGBM C++ API for production

class TreeEnsembleModel {
public:
    TreeEnsembleModel(int n_trees = 10, int max_depth = 3, int min_samples_split = 2);
    void fit(const std::vector<std::map<std::string, double>>& X, const std::vector<int>& y);
    std::vector<int> predict(const std::vector<std::map<std::string, double>>& X) const;
    std::map<std::string, double> feature_importances() const;
    void clear();
private:
    struct Node {
        bool is_leaf;
        int prediction;
        std::string feature;
        double threshold;
        Node* left;
        Node* right;
        Node() : is_leaf(true), prediction(0), threshold(0), left(nullptr), right(nullptr) {}
    };
    struct Tree {
        Node* root;
        std::map<std::string, double> importances;
        Tree() : root(nullptr) {}
    };
    int n_trees;
    int max_depth;
    int min_samples_split;
    std::vector<Tree> trees;
    std::vector<std::string> feature_names;
    void build_tree(Tree& tree, Node*& node, const std::vector<std::map<std::string, double>>& X, const std::vector<int>& y, int depth, std::mt19937& rng);
    int predict_tree(const Node* node, const std::map<std::string, double>& x) const;
    void collect_importances(const Node* node, std::map<std::string, double>& importances, double weight) const;
    void delete_tree(Node* node);
};
