#pragma once
#include <vector>
#include <string>
#include "DataRow.h"
#include "CUSUMFilter.h"

struct Event {
    size_t index;
    std::string timestamp;
};

class EventSelector {
public:
    static std::vector<Event> selectEvents(const std::vector<DataRow>& rows, int interval) {
        std::vector<Event> events;

        for (size_t i = 0; i < rows.size(); i += interval) {
            events.push_back(Event{i, rows[i].timestamp});
        }
        
        return events;
    }
    static std::vector<Event> selectCUSUMEvents(const std::vector<DataRow>& rows, const std::vector<double>& volatility, double threshold) {
        std::vector<double> prices;
        for (const auto& r : rows) prices.push_back(r.price);
        std::vector<size_t> indices = CUSUMFilter::detect(prices, volatility, threshold);
        std::vector<Event> events;
        for (size_t idx : indices) {
            if (idx < rows.size()) events.push_back(Event{idx, rows[idx].timestamp});
        }
        return events;
    }
};
