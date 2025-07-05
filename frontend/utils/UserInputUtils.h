#ifndef USERINPUTUTILS_H
#define USERINPUTUTILS_H

#include <QString>
#include "../BarrierConfigDialog.h"
#include "../../backend/data/BarrierConfig.h"
#include "../../backend/data/DataPreprocessor.h"
#include <QWidget>

class UserInputUtils {
public:
    static bool getBarrierConfig(QWidget* parent, BarrierConfig& cfg);
    static bool getPreprocessingParams(QWidget* parent, DataPreprocessor::Params& params, const BarrierConfig& cfg);
    static bool getLabelingConfig(QWidget* parent, BarrierConfig& cfg, DataPreprocessor::Params& params);
};

#endif // USERINPUTUTILS_H
