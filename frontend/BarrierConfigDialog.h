#pragma once
#include <QDialog>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QCheckBox>
#include <QComboBox>
#include "../backend/data/BarrierConfig.h"

class BarrierConfigDialog : public QDialog {
    Q_OBJECT
public:
    BarrierConfigDialog(QWidget* parent = nullptr) : QDialog(parent) {
        setWindowTitle("Set Barrier Config");
        auto* layout = new QFormLayout(this);
        
        // Labeling Type Selection
        labelingTypeBox = new QComboBox(this);
        labelingTypeBox->addItem("Hard Barrier");
        labelingTypeBox->addItem("TTBM (Time-to-Barrier Modification)");
        layout->addRow("Labeling Type:", labelingTypeBox);
        
        profitBox = new QDoubleSpinBox(this);
        profitBox->setRange(0.01, 10.0);
        profitBox->setValue(2.0);
        stopBox = new QDoubleSpinBox(this);
        stopBox->setRange(0.01, 10.0);
        stopBox->setValue(1.0);
        vertBox = new QSpinBox(this);
        vertBox->setRange(1, 1000);
        vertBox->setValue(20);
        cusumCheck = new QCheckBox("Use CUSUM Event Detection", this);
        cusumThresholdBox = new QDoubleSpinBox(this);
        cusumThresholdBox->setRange(0.1, 20.0);
        cusumThresholdBox->setValue(5.0);
        cusumThresholdBox->setEnabled(false);
        
        // TTBM Options
        ttbmDecayTypeBox = new QComboBox(this);
        ttbmDecayTypeBox->addItem("Exponential Decay");
        ttbmDecayTypeBox->addItem("Linear Decay");
        ttbmDecayTypeBox->addItem("Hyperbolic Decay");
        ttbmDecayTypeBox->setEnabled(false);
        
        ttbmLambdaBox = new QDoubleSpinBox(this);
        ttbmLambdaBox->setRange(0.1, 10.0);
        ttbmLambdaBox->setValue(1.0);
        ttbmLambdaBox->setDecimals(2);
        ttbmLambdaBox->setEnabled(false);
        
        ttbmAlphaBox = new QDoubleSpinBox(this);
        ttbmAlphaBox->setRange(0.0, 1.0);
        ttbmAlphaBox->setValue(0.5);
        ttbmAlphaBox->setDecimals(2);
        ttbmAlphaBox->setEnabled(false);
        
        ttbmBetaBox = new QDoubleSpinBox(this);
        ttbmBetaBox->setRange(0.1, 10.0);
        ttbmBetaBox->setValue(1.0);
        ttbmBetaBox->setDecimals(2);
        ttbmBetaBox->setEnabled(false);
        
        volWinBox = new QSpinBox(this);
        volWinBox->setRange(1, 1000);
        volWinBox->setValue(20);
        volWinBox = new QSpinBox(this);
        volWinBox->setRange(1, 1000);
        volWinBox->setValue(20);
        layout->addRow("Profit Multiple:", profitBox);
        layout->addRow(new QLabel("• Multiplier for the profit-taking barrier (e.g., 2.0 = 2x volatility above entry).", this));
        layout->addRow("Stop Multiple:", stopBox);
        layout->addRow(new QLabel("• Multiplier for the stop-loss barrier (e.g., 1.0 = 1x volatility below entry).", this));
        layout->addRow("Vertical Window:", vertBox);
        layout->addRow(new QLabel("• Maximum holding period in bars (time steps) before exit.", this));
        layout->addRow(cusumCheck);
        layout->addRow(new QLabel("• Enable CUSUM event detection for volatility-based event filtering.", this));
        layout->addRow("CUSUM Threshold:", cusumThresholdBox);
        layout->addRow(new QLabel("• Sensitivity for CUSUM filter (higher = fewer events).", this));
        
        // TTBM Options
        layout->addRow("TTBM Decay Type:", ttbmDecayTypeBox);
        layout->addRow(new QLabel("• How quickly label confidence decays with time.", this));
        layout->addRow("Lambda (λ) - Exponential:", ttbmLambdaBox);
        layout->addRow(new QLabel("• Exponential decay rate (higher = faster decay).", this));
        layout->addRow("Alpha (α) - Linear:", ttbmAlphaBox);
        layout->addRow(new QLabel("• Linear decay factor (0-1, higher = more decay).", this));
        layout->addRow("Beta (β) - Hyperbolic:", ttbmBetaBox);
        layout->addRow(new QLabel("• Hyperbolic decay steepness (higher = faster decay).", this));
        
        layout->addRow("Volatility Window:", volWinBox);
        layout->addRow(new QLabel("• Window size for volatility calculation (e.g., 20 = 20 bars).", this));
        errorLabel = new QLabel(this);
        errorLabel->setStyleSheet("color: #e74c3c; font-size: 12px;");
        layout->addRow(errorLabel);
        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::accepted, this, &BarrierConfigDialog::onAccept);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        connect(cusumCheck, &QCheckBox::toggled, cusumThresholdBox, &QDoubleSpinBox::setEnabled);
        
        // Enable/disable TTBM options based on labeling type
        connect(labelingTypeBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
            bool isTTBM = (index == 1);
            ttbmDecayTypeBox->setEnabled(isTTBM);
            ttbmLambdaBox->setEnabled(isTTBM);
            ttbmAlphaBox->setEnabled(isTTBM);
            ttbmBetaBox->setEnabled(isTTBM);
        });
    }
    BarrierConfig getConfig() const {
        BarrierConfig cfg;
        cfg.profit_multiple = profitBox->value();
        cfg.stop_multiple = stopBox->value();
        cfg.vertical_window = vertBox->value();
        cfg.use_cusum = cusumCheck->isChecked();
        cfg.cusum_threshold = cusumThresholdBox->value();
        
        // Set labeling type
        cfg.labeling_type = (labelingTypeBox->currentIndex() == 0) ? 
                           BarrierConfig::Hard : BarrierConfig::TTBM;
        
        // TTBM parameters
        cfg.ttbm_decay_type = static_cast<BarrierConfig::TTBMDecayType>(ttbmDecayTypeBox->currentIndex());
        cfg.ttbm_lambda = ttbmLambdaBox->value();
        cfg.ttbm_alpha = ttbmAlphaBox->value();
        cfg.ttbm_beta = ttbmBetaBox->value();
        
        return cfg;
    }
private slots:
    void onAccept() {
        try {
            getConfig().validate();
            accept();
        } catch (const std::exception& ex) {
            errorLabel->setText(ex.what());
        }
    }
private:
    QDoubleSpinBox *profitBox;
    QDoubleSpinBox *stopBox;
    QSpinBox *vertBox;
    QCheckBox *cusumCheck;
    QDoubleSpinBox *cusumThresholdBox;
    QLabel *errorLabel;
    QComboBox* labelingTypeBox;
    QSpinBox* volWinBox;
    
    // TTBM Controls
    QComboBox* ttbmDecayTypeBox;
    QDoubleSpinBox* ttbmLambdaBox;
    QDoubleSpinBox* ttbmAlphaBox;
    QDoubleSpinBox* ttbmBetaBox;
public:
    int volatilityWindow() const { return volWinBox->value(); }
};
