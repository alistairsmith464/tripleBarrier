#pragma once
#include <vector>
#include <cstddef>

class OverlapPurger {
public:
    static std::vector<size_t> purgeOverlappingEvents(
        const std::vector<size_t>& event_indices,
        int vertical_barrier,
        int min_gap = -1
    );
    
private:
    static bool hasOverlap(size_t event1_start, size_t event1_end, 
                          size_t event2_start, size_t event2_end);
};
