#pragma once
#include <QDialog>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include "../backend/data/BarrierConfig.h"

class BarrierConfigDialog : public QDialog {
    Q_OBJECT
    
public:
    BarrierConfigDialog(QWidget* parent = nullptr);
    BarrierConfig getConfig() const;
    int volatilityWindow() const;

private:
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
