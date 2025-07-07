#pragma once
#include <QDialog>
#include <QLabel>
#include <vector>
#include <map>
#include <QString>
#include <QSet>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"
#include <memory>
#include "../backend/ml/MLPipeline.h"

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
    std::vector<std::map<std::string, double>> m_features;
    std::vector<int> m_labels;
    std::vector<double> m_returns;
    void extractFeaturesAndLabels(const QSet<QString>& selectedFeatures,
                                 const std::vector<PreprocessedRow>& rows,
                                 const std::vector<LabeledEvent>& labeledEvents,
                                 std::vector<std::map<std::string, double>>& features,
                                 std::vector<int>& labels,
                                 std::vector<double>& returns);
    void extractFeaturesAndLabelsSoft(const QSet<QString>& selectedFeatures,
                                 const std::vector<PreprocessedRow>& rows,
                                 const std::vector<LabeledEvent>& labeledEvents,
                                 std::vector<std::map<std::string, double>>& features,
                                 std::vector<double>& soft_labels,
                                 std::vector<double>& returns);
    void showMLResults(const MLPipeline::PipelineResult& result);
};
