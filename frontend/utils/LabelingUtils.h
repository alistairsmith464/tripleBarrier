#pragma once
#include <vector>
#include "../../backend/data/PreprocessedRow.h"
#include "../../backend/data/LabeledEvent.h"
#include "../../backend/data/BarrierConfig.h"

namespace LabelingUtils {
    std::vector<LabeledEvent> labelEvents(
        const std::vector<PreprocessedRow>& processed,
        const std::vector<size_t>& event_indices,
        const BarrierConfig& cfg
    );
}
