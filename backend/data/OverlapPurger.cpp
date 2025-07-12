#include "OverlapPurger.h"
#include <algorithm>
#include <iostream>

std::vector<size_t> OverlapPurger::purgeOverlappingEvents(
    const std::vector<size_t>& event_indices,
    int vertical_barrier,
    int min_gap
) {
    if (event_indices.empty()) return event_indices;
    
    int effective_min_gap = (min_gap == -1) ? vertical_barrier : min_gap;
    
    std::vector<size_t> sorted_indices = event_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    
    std::vector<size_t> purged;
    purged.reserve(sorted_indices.size());
    
    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        size_t current_start = sorted_indices[i];
        size_t current_end = current_start + vertical_barrier;
        
        bool has_overlap = false;
        for (const auto& existing_start : purged) {
            size_t existing_end = existing_start + vertical_barrier;
            
            if (hasOverlap(existing_start, existing_end, current_start, current_end)) {
                has_overlap = true;
                break;
            }
            
            if (current_start < existing_start + effective_min_gap) {
                has_overlap = true;
                break;
            }
        }
        
        if (!has_overlap) {
            purged.push_back(current_start);
        }
    }
    
    return purged;
}

bool OverlapPurger::hasOverlap(size_t event1_start, size_t event1_end, 
                              size_t event2_start, size_t event2_end) {
    return !(event1_end <= event2_start || event2_end <= event1_start);
}
