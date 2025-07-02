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
        int volatility_window;
        int event_interval;
        double barrier_multiple;
        int vertical_barrier;
        Params(int volatility_window_ = 20, int event_interval_ = 10, double barrier_multiple_ = 2.0, int vertical_barrier_ = 20)
            : volatility_window(volatility_window_), event_interval(event_interval_), barrier_multiple(barrier_multiple_), vertical_barrier(vertical_barrier_) {}
    };

    static std::vector<PreprocessedRow> preprocess(const std::vector<DataRow>& rows, const Params& params = Params());
};
