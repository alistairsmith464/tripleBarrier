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
    static std::vector<Event> selectEvents(const std::vector<DataRow>& rows, int interval);
    static std::vector<Event> selectCUSUMEvents(const std::vector<DataRow>& rows, const std::vector<double>& volatility, double threshold);
    static std::vector<Event> selectDynamicEvents(const std::vector<DataRow>& rows, int vertical_barrier);
    
    static std::vector<Event> selectEventsWithGap(const std::vector<DataRow>& rows, int interval, int min_gap);
    static std::vector<Event> selectCUSUMEventsWithGap(const std::vector<DataRow>& rows, 
                                                       const std::vector<double>& volatility, 
                                                       double threshold, int min_gap);

private:
    static std::vector<Event> enforceMinimumGap(const std::vector<Event>& events, int min_gap);
};
