#pragma once
#include "IBarrierLabeler.h"
#include "PreprocessedRow.h"
#include "LabeledEvent.h"
#include <vector>

class ProbabilisticBarrierLabeler : public IBarrierLabeler {
public:
    std::vector<LabeledEvent> label(
        const std::vector<PreprocessedRow>& data,
        const std::vector<size_t>& event_indices,
        double profit_multiple,
        double stop_multiple,
        int vertical_barrier
    ) const override;
    // Optionally: add parameters for sigmoid steepness, etc.
};
