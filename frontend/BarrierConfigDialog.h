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
        volWinBox = new QSpinBox(this);
        volWinBox->setRange(1, 1000);
        volWinBox->setValue(20);
        evtIntBox = new QSpinBox(this);
        evtIntBox->setRange(1, 1000);
        evtIntBox->setValue(10);
        layout->addRow("Profit Multiple:", profitBox);
        layout->addRow(new QLabel("\u2022 Multiplier for the profit-taking barrier (e.g., 2.0 = 2x volatility above entry).", this));
        layout->addRow("Stop Multiple:", stopBox);
        layout->addRow(new QLabel("\u2022 Multiplier for the stop-loss barrier (e.g., 1.0 = 1x volatility below entry).", this));
        layout->addRow("Vertical Window:", vertBox);
        layout->addRow(new QLabel("\u2022 Maximum holding period in bars (time steps) before exit.", this));
        layout->addRow(cusumCheck);
        layout->addRow(new QLabel("\u2022 Enable CUSUM event detection for volatility-based event filtering.", this));
        layout->addRow("CUSUM Threshold:", cusumThresholdBox);
        layout->addRow(new QLabel("\u2022 Sensitivity for CUSUM filter (higher = fewer events).", this));
        layout->addRow("Volatility Window:", volWinBox);
        layout->addRow(new QLabel("\u2022 Window size for volatility calculation (e.g., 20 = 20 bars).", this));
        layout->addRow("Event Interval:", evtIntBox);
        layout->addRow(new QLabel("\u2022 Minimum interval between detected events (in bars).", this));
        errorLabel = new QLabel(this);
        errorLabel->setStyleSheet("color: #e74c3c; font-size: 12px;");
        layout->addRow(errorLabel);
        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::accepted, this, &BarrierConfigDialog::onAccept);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        connect(cusumCheck, &QCheckBox::toggled, cusumThresholdBox, &QDoubleSpinBox::setEnabled);

        QComboBox* labelingTypeBox = new QComboBox(this);
        labelingTypeBox->addItem("Hard Barrier");
        labelingTypeBox->addItem("Probabilistic Barrier");
        layout->insertRow(0, "Labeling Type:", labelingTypeBox);
        this->labelingTypeBox = labelingTypeBox;
    }
    BarrierConfig getConfig() const {
        BarrierConfig cfg;
        cfg.profit_multiple = profitBox->value();
        cfg.stop_multiple = stopBox->value();
        cfg.vertical_window = vertBox->value();
        cfg.use_cusum = cusumCheck->isChecked();
        cfg.cusum_threshold = cusumThresholdBox->value();
        if (labelingTypeBox->currentIndex() == 0) {
            cfg.labeling_type = BarrierConfig::Hard;
        } else {
            cfg.labeling_type = BarrierConfig::Probabilistic;
        }
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
    QSpinBox* evtIntBox;
public:
    int volatilityWindow() const { return volWinBox->value(); }
    int eventInterval() const { return evtIntBox->value(); }
};
