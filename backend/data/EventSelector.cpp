#include "EventSelector.h"
#include <algorithm>

std::vector<Event> EventSelector::selectEvents(const std::vector<DataRow>& rows, int interval) {
    std::vector<Event> events;
    for (size_t i = 0; i < rows.size(); i += interval) {
        events.push_back(Event{i, rows[i].timestamp});
    }
    return events;
}

std::vector<Event> EventSelector::selectCUSUMEvents(const std::vector<DataRow>& rows, const std::vector<double>& volatility, double threshold) {
    std::vector<double> prices;
    for (const auto& r : rows) prices.push_back(r.price);
    std::vector<size_t> indices = CUSUMFilter::detect(prices, volatility, threshold);
    std::vector<Event> events;
    for (size_t idx : indices) {
        if (idx < rows.size()) events.push_back(Event{idx, rows[idx].timestamp});
    }
    return events;
}

std::vector<Event> EventSelector::selectDynamicEvents(const std::vector<DataRow>& rows, int vertical_barrier) {
    std::vector<Event> events;
    int dynamic_interval = std::max(1, vertical_barrier / 3);
    for (size_t i = 0; i < rows.size(); i += dynamic_interval) {
        events.push_back(Event{i, rows[i].timestamp});
    }
    return events;
}
