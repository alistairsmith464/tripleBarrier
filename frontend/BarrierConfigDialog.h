#pragma once
#include <QDialog>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QCheckBox>
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
        stopBox->setRange(0.0, 1.0);
        stopBox->setValue(1.0);
        vertBox = new QSpinBox(this);
        vertBox->setRange(1, 1000);
        vertBox->setValue(20);
        cusumCheck = new QCheckBox("Use CUSUM Event Detection", this);
        cusumThresholdBox = new QDoubleSpinBox(this);
        cusumThresholdBox->setRange(0.1, 20.0);
        cusumThresholdBox->setValue(5.0);
        cusumThresholdBox->setEnabled(false);
        layout->addRow("Profit Multiple:", profitBox);
        layout->addRow("Stop Multiple:", stopBox);
        layout->addRow("Vertical Window:", vertBox);
        layout->addRow(cusumCheck);
        layout->addRow("CUSUM Threshold:", cusumThresholdBox);
        errorLabel = new QLabel(this);
        errorLabel->setStyleSheet("color: #e74c3c; font-size: 12px;");
        layout->addRow(errorLabel);
        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::accepted, this, &BarrierConfigDialog::onAccept);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        connect(cusumCheck, &QCheckBox::toggled, cusumThresholdBox, &QDoubleSpinBox::setEnabled);
    }
    BarrierConfig getConfig() const {
        BarrierConfig cfg;
        cfg.profit_multiple = profitBox->value();
        cfg.stop_multiple = stopBox->value();
        cfg.vertical_window = vertBox->value();
        cfg.use_cusum = cusumCheck->isChecked();
        cfg.cusum_threshold = cusumThresholdBox->value();
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
};
