#include "LabelingUtils.h"
#include "../../backend/data/HardBarrierLabeler.h"
#include "../../backend/data/TTBMLabeler.h"
#include <iostream>

namespace LabelingUtils {
    std::vector<LabeledEvent> labelEvents(
        const std::vector<PreprocessedRow>& processed,
        const std::vector<size_t>& event_indices,
        const BarrierConfig& cfg
    ) {
        if (cfg.labeling_type == BarrierConfig::TTBM) {
            TTBMLabeler labeler(cfg.ttbm_decay_type, cfg.ttbm_lambda, cfg.ttbm_alpha, cfg.ttbm_beta);
            return labeler.label(
                processed,
                event_indices,
                cfg.profit_multiple,
                cfg.stop_multiple,
                cfg.vertical_window
            );
        } else {
            HardBarrierLabeler labeler;
            return labeler.label(
                processed,
                event_indices,
                cfg.profit_multiple,
                cfg.stop_multiple,
                cfg.vertical_window
            );
        }
    }
}
