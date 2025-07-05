#include "TreeEnsembleModel.h"
#include <numeric>
#include <set>

TreeEnsembleModel::TreeEnsembleModel(int n_trees_, int max_depth_, int min_samples_split_)
    : n_trees(n_trees_), max_depth(max_depth_), min_samples_split(min_samples_split_) {}

void TreeEnsembleModel::fit(const std::vector<std::map<std::string, double>>& X, const std::vector<int>& y) {
    clear();
    if (X.empty()) return;
    // Collect feature names
    feature_names.clear();
    for (const auto& kv : X[0]) feature_names.push_back(kv.first);
    std::mt19937 rng(42);
    for (int t = 0; t < n_trees; ++t) {
        // Bootstrap sample
        std::vector<std::map<std::string, double>> Xb;
        std::vector<int> yb;
        std::uniform_int_distribution<size_t> dist(0, X.size() - 1);
        for (size_t i = 0; i < X.size(); ++i) {
            size_t idx = dist(rng);
            Xb.push_back(X[idx]);
            yb.push_back(y[idx]);
        }
        Tree tree;
        build_tree(tree, tree.root, Xb, yb, 0, rng);
        trees.push_back(tree);
    }
}

std::vector<int> TreeEnsembleModel::predict(const std::vector<std::map<std::string, double>>& X) const {
    std::vector<int> preds(X.size(), 0);
    if (trees.empty()) return preds;
    for (size_t i = 0; i < X.size(); ++i) {
        std::map<int, int> votes;
        for (const auto& tree : trees) {
            int p = predict_tree(tree.root, X[i]);
            votes[p]++;
        }
        // Majority vote
        int best = 0, best_count = 0;
        for (const auto& kv : votes) {
            if (kv.second > best_count) {
                best = kv.first;
                best_count = kv.second;
            }
        }
        preds[i] = best;
    }
    return preds;
}

std::map<std::string, double> TreeEnsembleModel::feature_importances() const {
    std::map<std::string, double> importances;
    for (const auto& tree : trees) {
        collect_importances(tree.root, importances, 1.0);
    }
    // Normalize
    double total = 0.0;
    for (const auto& kv : importances) total += kv.second;
    if (total > 0) for (auto& kv : importances) kv.second /= total;
    return importances;
}

void TreeEnsembleModel::clear() {
    for (auto& tree : trees) delete_tree(tree.root);
    trees.clear();
}

void TreeEnsembleModel::build_tree(Tree& tree, Node*& node, const std::vector<std::map<std::string, double>>& X, const std::vector<int>& y, int depth, std::mt19937& rng) {
    node = new Node();
    // Stopping conditions
    std::set<int> unique_y(y.begin(), y.end());
    if (unique_y.size() == 1 || depth >= max_depth || X.size() < (size_t)min_samples_split) {
        node->is_leaf = true;
        node->prediction = *unique_y.begin();
        return;
    }
    // Find best split (greedy, single feature, threshold)
    double best_gini = 1.0;
    std::string best_feat;
    double best_thresh = 0;
    std::vector<size_t> best_left, best_right;
    for (const auto& feat : feature_names) {
        std::vector<double> values;
        for (const auto& row : X) values.push_back(row.at(feat));
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        for (size_t j = 1; j < sorted.size(); ++j) {
            double thresh = (sorted[j - 1] + sorted[j]) / 2.0;
            std::vector<size_t> left, right;
            for (size_t i = 0; i < X.size(); ++i) {
                if (X[i].at(feat) <= thresh) left.push_back(i);
                else right.push_back(i);
            }
            if (left.empty() || right.empty()) continue;
            // Gini impurity
            auto gini = [&](const std::vector<size_t>& idxs) {
                std::map<int, int> counts;
                for (size_t i : idxs) counts[y[i]]++;
                double impurity = 1.0;
                for (const auto& kv : counts) {
                    double p = double(kv.second) / idxs.size();
                    impurity -= p * p;
                }
                return impurity;
            };
            double g = (left.size() * gini(left) + right.size() * gini(right)) / X.size();
            if (g < best_gini) {
                best_gini = g;
                best_feat = feat;
                best_thresh = thresh;
                best_left = left;
                best_right = right;
            }
        }
    }
    if (best_left.empty() || best_right.empty()) {
        node->is_leaf = true;
        // Majority class
        std::map<int, int> counts;
        for (int v : y) counts[v]++;
        int maj = y[0], cnt = 0;
        for (const auto& kv : counts) if (kv.second > cnt) { maj = kv.first; cnt = kv.second; }
        node->prediction = maj;
        return;
    }
    node->is_leaf = false;
    node->feature = best_feat;
    node->threshold = best_thresh;
    // Feature importance: count splits
    tree.importances[best_feat] += 1.0;
    // Split data
    std::vector<std::map<std::string, double>> X_left, X_right;
    std::vector<int> y_left, y_right;
    for (size_t i : best_left) { X_left.push_back(X[i]); y_left.push_back(y[i]); }
    for (size_t i : best_right) { X_right.push_back(X[i]); y_right.push_back(y[i]); }
    build_tree(tree, node->left, X_left, y_left, depth + 1, rng);
    build_tree(tree, node->right, X_right, y_right, depth + 1, rng);
}

int TreeEnsembleModel::predict_tree(const Node* node, const std::map<std::string, double>& x) const {
    if (!node) return 0;
    if (node->is_leaf) return node->prediction;
    if (x.at(node->feature) <= node->threshold) return predict_tree(node->left, x);
    else return predict_tree(node->right, x);
}

void TreeEnsembleModel::collect_importances(const Node* node, std::map<std::string, double>& importances, double weight) const {
    if (!node || node->is_leaf) return;
    importances[node->feature] += weight;
    collect_importances(node->left, importances, weight * 0.5);
    collect_importances(node->right, importances, weight * 0.5);
}

void TreeEnsembleModel::delete_tree(Node* node) {
    if (!node) return;
    delete_tree(node->left);
    delete_tree(node->right);
    delete node;
}
