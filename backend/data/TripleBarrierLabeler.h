#pragma once
#include <vector>
#include "PreprocessedRow.h"
#include "LabeledEvent.h"

class TripleBarrierLabeler {
public:
    static std::vector<LabeledEvent> label(
        const std::vector<PreprocessedRow>& data,
        const std::vector<size_t>& event_indices,
        double profit_multiple,
        double stop_multiple,
        int vertical_barrier
    );
};
