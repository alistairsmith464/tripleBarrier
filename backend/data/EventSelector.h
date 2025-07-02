#pragma once
#include <vector>
#include <string>
#include "DataRow.h"

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
};
