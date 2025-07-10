#pragma once
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include "../backend/data/BarrierConfig.h"
#include "frontend/config/VisualizationConfig.h"
#include "ui/BaseDialog.h"
#include "ui/UIStrings.h"
#include "utils/InputValidator.h"

class BarrierConfigDialog : public BaseDialog {
    Q_OBJECT
    
public:
    BarrierConfigDialog(QWidget* parent = nullptr);
    BarrierConfig getConfig() const;

protected:
    ValidationResult validateInput() override;
    void onAccept() override;

private:
    void setupControls();
    void setupValidation();
    void onLabelingTypeChanged(int index);
    void onCusumToggled(bool enabled);
    
    // UI Controls
    QComboBox* labelingTypeBox;
    QDoubleSpinBox* profitBox;
    QDoubleSpinBox* stopBox;
    QSpinBox* vertBox;
    QCheckBox* cusumCheck;
    QDoubleSpinBox* cusumThresholdBox;
    QComboBox* ttbmDecayTypeBox;
    QSpinBox* volWinBox;
};
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
        
        ttbmDecayTypeBox = new QComboBox(this);
        ttbmDecayTypeBox->addItem("Exponential Decay (Fixed Parameters)");
        ttbmDecayTypeBox->addItem("Linear Decay (Fixed Parameters)");
        ttbmDecayTypeBox->addItem("Hyperbolic Decay (Fixed Parameters)");
        ttbmDecayTypeBox->setEnabled(false);
        
        // Parameter controls removed - using fixed optimal values from configuration
        ttbmLambdaBox = nullptr;
        ttbmAlphaBox = nullptr;
        ttbmBetaBox = nullptr;
        
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
        
        // TTBM Options (parameters now fixed for regression optimization)
        layout->addRow("TTBM Decay Type:", ttbmDecayTypeBox);
        layout->addRow(new QLabel("• Decay parameters are now fixed and optimized for regression performance.", this));
        
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
            // Individual parameter controls are now fixed (no longer user-editable)
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
        
        cfg.ttbm_decay_type = static_cast<BarrierConfig::TTBMDecayType>(ttbmDecayTypeBox->currentIndex());
        cfg.ttbm_lambda = VisualizationConfig::getOptimalLambda();
        cfg.ttbm_alpha = VisualizationConfig::getOptimalAlpha();
        cfg.ttbm_beta = VisualizationConfig::getOptimalBeta();
        
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
    
    // TTBM controls - fixed parameters loaded from configuration
    QComboBox* ttbmDecayTypeBox;
    QDoubleSpinBox* ttbmLambdaBox;  // Set to nullptr - using fixed configuration values
    QDoubleSpinBox* ttbmAlphaBox;   // Set to nullptr - using fixed configuration values
    QDoubleSpinBox* ttbmBetaBox;    // Set to nullptr - using fixed configuration values
public:
    int volatilityWindow() const { return volWinBox->value(); }
};
