#include "UserInputUtils.h"
#include <QInputDialog>

bool UserInputUtils::getBarrierConfig(QWidget* parent, BarrierConfig& cfg) {
    BarrierConfigDialog dialog(parent);
    if (dialog.exec() == QDialog::Accepted) {
        cfg = dialog.getConfig();
        return true;
    }
    return false;
}

bool UserInputUtils::getPreprocessingParams(QWidget* parent, DataPreprocessor::Params& params, const BarrierConfig& cfg, int volWin) {
    params.volatility_window = volWin;
    params.barrier_multiple = cfg.profit_multiple;
    params.vertical_barrier = cfg.vertical_window;
    params.use_cusum = cfg.use_cusum;
    params.cusum_threshold = cfg.cusum_threshold;
    return true;
}

bool UserInputUtils::getLabelingConfig(QWidget* parent, BarrierConfig& cfg, DataPreprocessor::Params& params) {
    BarrierConfigDialog dialog(parent);
    if (dialog.exec() != QDialog::Accepted) return false;
    cfg = dialog.getConfig();
    return getPreprocessingParams(parent, params, cfg, dialog.volatilityWindow());
}
