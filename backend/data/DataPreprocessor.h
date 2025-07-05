#pragma once
#include <vector>
#include "../backend/data/DataRow.h"
#include "PreprocessedRow.h"
#include "VolatilityCalculator.h"
#include "EventSelector.h"
#include "BarrierConfig.h"

class DataPreprocessor {
public:
    struct Params {
        int volatility_window = 20;
        int event_interval = 10;
        double barrier_multiple = 2.0;
        int vertical_barrier = 20;
        // CUSUM options
        bool use_cusum = false;
        double cusum_threshold = 5.0;
    };

    static std::vector<PreprocessedRow> preprocess(const std::vector<DataRow>& rows, const Params& params);
};
