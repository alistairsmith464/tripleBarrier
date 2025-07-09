#pragma once
#include <QDialog>
#include <QLabel>
#include <QCheckBox>
#include <vector>
#include <map>
#include <QString>
#include <QSet>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"
#include <memory>
#include "../backend/ml/MLPipeline.h"

// Portfolio simulation results structure
struct PortfolioResults {
    double starting_capital = 100000.0;
    double final_value = 0.0;
    double total_return = 0.0;
    double annualized_return = 0.0;
    double max_drawdown = 0.0;
    double sharpe_ratio = 0.0;
    int total_trades = 0;
    int winning_trades = 0;
    int losing_trades = 0;
    double win_rate = 0.0;
    double avg_trade_return = 0.0;
    double best_trade = 0.0;
    double worst_trade = 0.0;
    std::vector<double> portfolio_values;
    std::vector<double> trade_returns;
};

class FeaturePreviewDialog : public QDialog {
    Q_OBJECT
public:
    FeaturePreviewDialog(const QSet<QString>& selectedFeatures,
                        const std::vector<PreprocessedRow>& rows,
                        const std::vector<LabeledEvent>& labeledEvents,
                        QWidget* parent = nullptr);
private slots:
    void onRunMLClicked();
private:
    QSet<QString> m_selectedFeatures;
    std::vector<PreprocessedRow> m_rows;
    std::vector<LabeledEvent> m_labeledEvents;
    QPushButton* m_runMLButton;
    QLabel* m_metricsLabel;
    QLabel* m_importancesLabel;
    QCheckBox* m_tuneHyperparamsCheckBox;
    QLabel* m_dataInfoLabel;
    QLabel* m_debugInfoLabel;
    std::vector<std::map<std::string, double>> m_features;
    std::vector<int> m_labels;
    std::vector<double> m_labels_double;  // For TTBM regression
    std::vector<double> m_returns;
    void extractFeaturesAndLabels(const QSet<QString>& selectedFeatures,
                                 const std::vector<PreprocessedRow>& rows,
                                 const std::vector<LabeledEvent>& labeledEvents,
                                 std::vector<std::map<std::string, double>>& features,
                                 std::vector<int>& labels,
                                 std::vector<double>& returns);
    void extractFeaturesAndLabelsRegression(const QSet<QString>& selectedFeatures,
                                           const std::vector<PreprocessedRow>& rows,
                                           const std::vector<LabeledEvent>& labeledEvents,
                                           std::vector<std::map<std::string, double>>& features,
                                           std::vector<double>& labels,
                                           std::vector<double>& returns);
    void showMLResults(const MLPipeline::PipelineResult& result);
    void showMLRegressionResults(const MLPipeline::RegressionPipelineResult& result);
    
    // Portfolio simulation functions
    PortfolioResults runPortfolioSimulation(const std::vector<double>& predictions,
                                           const std::vector<LabeledEvent>& events,
                                           bool is_ttbm);
    double calculateSharpeRatio(const std::vector<double>& returns);
    double calculateMaxDrawdown(const std::vector<double>& portfolio_values);
};
