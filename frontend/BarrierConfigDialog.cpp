#include "BarrierConfigDialog.h"

BarrierConfigDialog::BarrierConfigDialog(QWidget* parent) 
    : BaseDialog(parent, UIStrings::BARRIER_CONFIG_TITLE) {
    setupControls();
    setupValidation();
    setMinimumDialogSize(QSize(500, 600));
}

void BarrierConfigDialog::setupControls() {
    // Labeling Type Selection
    addSection(UIStrings::LABELING_CONFIG_SECTION);
    
    labelingTypeBox = new QComboBox(this);
    labelingTypeBox->addItem(UIStrings::HARD_BARRIER);
    labelingTypeBox->addItem(UIStrings::TTBM_BARRIER);
    addFormRow(UIStrings::LABELING_TYPE, labelingTypeBox);
    
    addSection(UIStrings::BARRIER_PARAMS_SECTION);
    
    // Profit Multiple
    profitBox = new QDoubleSpinBox(this);
    profitBox->setRange(0.01, 10.0);
    profitBox->setValue(2.0);
    profitBox->setSingleStep(0.1);
    addFormRow(UIStrings::PROFIT_MULTIPLE, profitBox, UIStrings::PROFIT_TOOLTIP);
    
    // Stop Multiple
    stopBox = new QDoubleSpinBox(this);
    stopBox->setRange(0.01, 10.0);
    stopBox->setValue(1.0);
    stopBox->setSingleStep(0.1);
    addFormRow(UIStrings::STOP_MULTIPLE, stopBox, UIStrings::STOP_TOOLTIP);
    
    // Vertical Window
    vertBox = new QSpinBox(this);
    vertBox->setRange(1, 1000);
    vertBox->setValue(20);
    addFormRow(UIStrings::VERTICAL_WINDOW, vertBox, UIStrings::VERTICAL_TOOLTIP);
    
    addSection(UIStrings::EVENT_DETECTION_SECTION);
    
    // CUSUM Configuration
    cusumCheck = new QCheckBox(UIStrings::USE_CUSUM, this);
    addFormRow(cusumCheck, nullptr, UIStrings::CUSUM_TOOLTIP);
    
    cusumThresholdBox = new QDoubleSpinBox(this);
    cusumThresholdBox->setRange(0.1, 20.0);
    cusumThresholdBox->setValue(5.0);
    cusumThresholdBox->setEnabled(false);
    addFormRow(UIStrings::CUSUM_THRESHOLD, cusumThresholdBox, UIStrings::CUSUM_THRESHOLD_TOOLTIP);
    
    addSection(UIStrings::TTBM_CONFIG_SECTION);
    
    // TTBM Decay Type
    ttbmDecayTypeBox = new QComboBox(this);
    ttbmDecayTypeBox->addItem(UIStrings::EXPONENTIAL_DECAY);
    ttbmDecayTypeBox->addItem(UIStrings::LINEAR_DECAY);
    ttbmDecayTypeBox->addItem(UIStrings::HYPERBOLIC_DECAY);
    ttbmDecayTypeBox->setEnabled(false);
    addFormRow(UIStrings::TTBM_DECAY_TYPE, ttbmDecayTypeBox);
    
    addSection(UIStrings::VOLATILITY_CALC_SECTION);
    
    // Volatility Window
    volWinBox = new QSpinBox(this);
    volWinBox->setRange(1, 1000);
    volWinBox->setValue(20);
    addFormRow(UIStrings::VOLATILITY_WINDOW, volWinBox, UIStrings::VOLATILITY_WINDOW_TOOLTIP);
    
    // Setup connections
    connect(cusumCheck, &QCheckBox::toggled, this, &BarrierConfigDialog::onCusumToggled);
    connect(labelingTypeBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &BarrierConfigDialog::onLabelingTypeChanged);
    
    setupStandardButtons();
}

void BarrierConfigDialog::setupValidation() {
    // Add validation for barrier configuration
    addValidator([this]() {
        BarrierConfig config = getConfig();
        return InputValidator::validateBarrierConfig(config);
    });
    
    // Add validation for window sizes
    addValidator([this]() {
        return InputValidator::validateRange(profitBox->value(), 0.01, 10.0, "Profit Multiple");
    });
    
    addValidator([this]() {
        return InputValidator::validateRange(stopBox->value(), 0.01, 10.0, "Stop Multiple");
    });
    
    addValidator([this]() {
        return InputValidator::validateRange(vertBox->value(), 1, 1000, "Vertical Window");
    });
}

void BarrierConfigDialog::onLabelingTypeChanged(int index) {
    bool isTTBM = (index == 1);
    ttbmDecayTypeBox->setEnabled(isTTBM);
}

void BarrierConfigDialog::onCusumToggled(bool enabled) {
    cusumThresholdBox->setEnabled(enabled);
}

ValidationResult BarrierConfigDialog::validateInput() {
    // First run base validation
    ValidationResult baseResult = BaseDialog::validateInput();
    if (!baseResult.isValid) {
        return baseResult;
    }
    
    // Additional custom validation
    BarrierConfig config = getConfig();
    return InputValidator::validateBarrierConfig(config);
}

void BarrierConfigDialog::onAccept() {
    // Custom acceptance logic if needed
    setStatusMessage(UIStrings::CONFIG_VALIDATED);
}

BarrierConfig BarrierConfigDialog::getConfig() const {
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
    
    // Use fixed optimal parameters for TTBM (from configuration)
    cfg.ttbm_lambda = VisualizationConfig::TTBM_LAMBDA;
    cfg.ttbm_alpha = VisualizationConfig::TTBM_ALPHA;
    cfg.ttbm_beta = VisualizationConfig::TTBM_BETA;
    
    cfg.vol_window = volWinBox->value();
    
    return cfg;
}
