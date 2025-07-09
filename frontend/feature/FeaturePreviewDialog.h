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
#include "../backend/data/FeatureExtractor.h"
#include "../backend/data/PortfolioSimulator.h"
#include "../backend/ml/MLPipeline.h"
#include <memory>

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
    // Data members
    QSet<QString> m_selectedFeatures;
    std::vector<PreprocessedRow> m_rows;
    std::vector<LabeledEvent> m_labeledEvents;
    
    // UI components
    QPushButton* m_runMLButton;
    QLabel* m_metricsLabel;
    QLabel* m_importancesLabel;
    QCheckBox* m_tuneHyperparamsCheckBox;
    QLabel* m_dataInfoLabel;
    QLabel* m_debugInfoLabel;
    
    // Helper methods
    void setupUI();
    void createFeatureTable();
    void updateDataInfo();
    void updateBarrierDiagnostics();
    void showMLClassificationResults(const MLPipeline::PipelineResult& result);
    void showMLRegressionResults(const MLPipeline::RegressionPipelineResult& result);
};
