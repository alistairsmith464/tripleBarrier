#include "DataPreprocessor.h"
#include <cmath>

std::vector<PreprocessedRow> DataPreprocessor::preprocess(const std::vector<DataRow>& rows, const Params& params) {
    std::vector<PreprocessedRow> out;
    
    if (rows.size() < 2) return out;

    std::vector<double> logReturns(rows.size(), 0.0);
    for (size_t i = 1; i < rows.size(); ++i) {
        logReturns[i] = std::log(rows[i].price / rows[i-1].price);
    }

    std::vector<double> vol = VolatilityCalculator::rollingStdDev(logReturns, params.volatility_window);

    std::vector<Event> events;
    if (params.use_cusum) {
        events = EventSelector::selectCUSUMEvents(rows, vol, params.cusum_threshold);
    } else {
        if (params.barrier_config.labeling_type == BarrierConfig::Hard) {
            events = EventSelector::selectDynamicEvents(rows, params.vertical_barrier);
        } else {
            events = EventSelector::selectEvents(rows, params.vertical_barrier);
        }
    }

    std::vector<bool> is_event(rows.size(), false);
    for (const auto& e : events) is_event[e.index] = true;

    for (size_t i = 0; i < rows.size(); ++i) {
        PreprocessedRow p;
        p.timestamp = rows[i].timestamp;
        p.price = rows[i].price;
        p.open = rows[i].open;
        p.high = rows[i].high;
        p.low = rows[i].low;
        p.close = rows[i].close;
        p.volume = rows[i].volume;
        p.log_return = logReturns[i];
        p.volatility = vol[i];
        p.is_event = is_event[i];
        out.push_back(p);
    }
    return out;
}
