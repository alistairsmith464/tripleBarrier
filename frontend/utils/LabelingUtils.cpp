#include "LabelingUtils.h"
#include "../../backend/data/HardBarrierLabeler.h"
#include "../../backend/data/ProbabilisticBarrierLabeler.h"

namespace LabelingUtils {
    std::vector<LabeledEvent> labelEvents(
        const std::vector<PreprocessedRow>& processed,
        const std::vector<size_t>& event_indices,
        const BarrierConfig& cfg
    ) {
        if (cfg.labeling_type == BarrierConfig::Hard) {
            HardBarrierLabeler labeler;
            return labeler.label(
                processed,
                event_indices,
                cfg.profit_multiple,
                cfg.stop_multiple,
                cfg.vertical_window
            );
        } else {
            ProbabilisticBarrierLabeler labeler;
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
