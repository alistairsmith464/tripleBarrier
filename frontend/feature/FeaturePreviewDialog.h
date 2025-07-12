#pragma once
#include <QDialog>
#include <QLabel>
#include <QCheckBox>
#include <QTableWidget>
#include <vector>
#include <map>
#include <QString>
#include <QSet>
#include <QVBoxLayout>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"
#include "../backend/data/FeatureExtractor.h"
#include "../backend/ml/PortfolioSimulator.h"
#include "../backend/ml/MLPipeline.h"
#include "../services/MLService.h"
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
    QSet<QString> m_selectedFeatures;
    std::vector<PreprocessedRow> m_rows;
    std::vector<LabeledEvent> m_labeledEvents;
    
    QPushButton* m_runMLButton;
    QLabel* m_metricsLabel;
    QLabel* m_importancesLabel;
    QCheckBox* m_tuneHyperparamsCheckBox;
    QLabel* m_dataInfoLabel;
    QLabel* m_debugInfoLabel;
    QTableWidget* m_tradeLogTable;
    
    void setupUI();
    void createFeatureTable(QVBoxLayout* vbox);
    void updateDataInfo();
    void updateBarrierDiagnostics();
    void showMLClassificationResults(const MLResults& results);
    void showMLRegressionResults(const MLResults& results);
    void displayTradeLog(const std::vector<MLPipeline::TradeLogEntry>& tradeLog);
};
