#include "BarrierConfigDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QLabel>

BarrierConfigDialog::BarrierConfigDialog(QWidget* parent) 
    : QDialog(parent) {
    setWindowTitle("Barrier Configuration");
    setModal(true);
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    
    // Simple form layout
    profitBox = new QDoubleSpinBox(this);
    profitBox->setRange(0.01, 10.0);
    profitBox->setValue(2.0);
    profitBox->setSingleStep(0.1);
    
    stopBox = new QDoubleSpinBox(this);
    stopBox->setRange(0.01, 10.0);
    stopBox->setValue(1.0);
    stopBox->setSingleStep(0.1);
    
    vertBox = new QSpinBox(this);
    vertBox->setRange(1, 1000);
    vertBox->setValue(20);
    
    cusumCheck = new QCheckBox("Use CUSUM", this);
    cusumThresholdBox = new QDoubleSpinBox(this);
    cusumThresholdBox->setRange(0.1, 20.0);
    cusumThresholdBox->setValue(5.0);
    cusumThresholdBox->setEnabled(false);
    
    labelingTypeBox = new QComboBox(this);
    labelingTypeBox->addItem("Hard Barrier");
    labelingTypeBox->addItem("TTBM Barrier");
    
    ttbmDecayTypeBox = new QComboBox(this);
    ttbmDecayTypeBox->addItem("Exponential");
    ttbmDecayTypeBox->addItem("Linear");
    ttbmDecayTypeBox->addItem("Hyperbolic");
    ttbmDecayTypeBox->setEnabled(false);
    
    volWinBox = new QSpinBox(this);
    volWinBox->setRange(1, 1000);
    volWinBox->setValue(20);
    
    // Add widgets to layout
    layout->addWidget(new QLabel("Profit Multiple:"));
    layout->addWidget(profitBox);
    layout->addWidget(new QLabel("Stop Multiple:"));
    layout->addWidget(stopBox);
    layout->addWidget(new QLabel("Vertical Window:"));
    layout->addWidget(vertBox);
    layout->addWidget(cusumCheck);
    layout->addWidget(new QLabel("CUSUM Threshold:"));
    layout->addWidget(cusumThresholdBox);
    layout->addWidget(new QLabel("Labeling Type:"));
    layout->addWidget(labelingTypeBox);
    layout->addWidget(new QLabel("TTBM Decay Type:"));
    layout->addWidget(ttbmDecayTypeBox);
    layout->addWidget(new QLabel("Volatility Window:"));
    layout->addWidget(volWinBox);
    
    // Buttons
    QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttonBox);
    
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(cusumCheck, &QCheckBox::toggled, cusumThresholdBox, &QDoubleSpinBox::setEnabled);
    connect(labelingTypeBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
        ttbmDecayTypeBox->setEnabled(index == 1);
    });
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
    
    // Use fixed parameters for TTBM
    cfg.ttbm_lambda = 2.0;
    cfg.ttbm_alpha = 0.8;
    cfg.ttbm_beta = 3.0;
    
    return cfg;
}

int BarrierConfigDialog::volatilityWindow() const {
    return volWinBox->value();
}
