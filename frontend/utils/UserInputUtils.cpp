#include "UserInputUtils.h"
#include "ValidationFramework.h"
#include <QInputDialog>

bool UserInputUtils::getBarrierConfig(QWidget* parent, BarrierConfig& cfg) {
    BarrierConfigDialog dialog(parent);
    if (dialog.exec() == QDialog::Accepted) {
        cfg = dialog.getConfig();
        ValidationFramework::CoreValidator::validatePositive(cfg.profit_multiple, "Profit multiple");
        ValidationFramework::CoreValidator::validatePositive(cfg.stop_multiple, "Stop multiple");
        ValidationFramework::CoreValidator::validatePositive(cfg.vertical_window, "Vertical window");
        if (cfg.use_cusum) {
            ValidationFramework::CoreValidator::validatePositive(cfg.cusum_threshold, "CUSUM threshold");
        }
        return true;
    }
    return false;
}

bool UserInputUtils::getPreprocessingParams(QWidget* parent, DataPreprocessor::Params& params, const BarrierConfig& cfg, int volWin) {
    ValidationFramework::CoreValidator::validatePositive(volWin, "Volatility window");
    ValidationFramework::CoreValidator::validatePositive(cfg.profit_multiple, "Profit multiple");
    ValidationFramework::CoreValidator::validatePositive(cfg.vertical_window, "Vertical window");
    params.volatility_window = volWin;
    params.barrier_multiple = cfg.profit_multiple;
    params.vertical_barrier = cfg.vertical_window;
    params.use_cusum = cfg.use_cusum;
    params.cusum_threshold = cfg.cusum_threshold;
    if (params.use_cusum) {
        ValidationFramework::CoreValidator::validatePositive(params.cusum_threshold, "CUSUM threshold");
    }
    return true;
}

bool UserInputUtils::getLabelingConfig(QWidget* parent, BarrierConfig& cfg, DataPreprocessor::Params& params) {
    BarrierConfigDialog dialog(parent);
    if (dialog.exec() != QDialog::Accepted) return false;
    cfg = dialog.getConfig();
    ValidationFramework::CoreValidator::validatePositive(cfg.profit_multiple, "Profit multiple");
    ValidationFramework::CoreValidator::validatePositive(cfg.stop_multiple, "Stop multiple");
    ValidationFramework::CoreValidator::validatePositive(cfg.vertical_window, "Vertical window");
    if (cfg.use_cusum) {
        ValidationFramework::CoreValidator::validatePositive(cfg.cusum_threshold, "CUSUM threshold");
    }
    return getPreprocessingParams(parent, params, cfg, dialog.volatilityWindow());
}
