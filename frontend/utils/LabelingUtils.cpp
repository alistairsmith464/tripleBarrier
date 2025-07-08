#include "LabelingUtils.h"
#include "../../backend/data/HardBarrierLabeler.h"
#include <iostream>

namespace LabelingUtils {
    std::vector<LabeledEvent> labelEvents(
        const std::vector<PreprocessedRow>& processed,
        const std::vector<size_t>& event_indices,
        const BarrierConfig& cfg
    ) {
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
