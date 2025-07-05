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

bool UserInputUtils::getPreprocessingParams(QWidget* parent, DataPreprocessor::Params& params, const BarrierConfig& cfg) {
    bool ok = false;
    int volWin = QInputDialog::getInt(parent, "Volatility Window", "Enter volatility window:", 20, 1, 1000, 1, &ok);
    if (!ok) return false;
    int evtInt = QInputDialog::getInt(parent, "Event Interval", "Enter event interval:", 10, 1, 1000, 1, &ok);
    if (!ok) return false;
    // Use values from BarrierConfig for the rest
    params.volatility_window = volWin;
    params.event_interval = evtInt;
    params.barrier_multiple = cfg.profit_multiple;
    params.vertical_barrier = cfg.vertical_window;
    params.use_cusum = cfg.use_cusum;
    params.cusum_threshold = cfg.cusum_threshold;
    return true;
}

bool UserInputUtils::getLabelingConfig(QWidget* parent, BarrierConfig& cfg, DataPreprocessor::Params& params) {
    if (!getBarrierConfig(parent, cfg)) return false;
    if (!getPreprocessingParams(parent, params, cfg)) return false;
    return true;
}
