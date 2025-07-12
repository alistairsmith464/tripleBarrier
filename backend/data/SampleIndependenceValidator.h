#pragma once
#include "LabeledEvent.h"
#include "PreprocessedRow.h"
#include <vector>

class SampleIndependenceValidator {
public:
    struct IndependenceReport {
        size_t total_samples;
        size_t overlapping_samples;
        size_t gap_violations;
        double overlap_percentage;
        double avg_gap_size;
        double min_gap_size;
        double max_gap_size;
        bool independence_violated;
    };
    
    static IndependenceReport validateSampleIndependence(
        const std::vector<LabeledEvent>& events,
        int vertical_barrier,
        int min_gap_requirement = -1
    );
    
    static std::vector<size_t> findOverlappingEventPairs(
        const std::vector<LabeledEvent>& events,
        int vertical_barrier
    );
    
    static std::vector<size_t> findGapViolations(
        const std::vector<LabeledEvent>& events,
        int min_gap_requirement
    );
    
private:
    static bool hasTemporalOverlap(const LabeledEvent& event1, const LabeledEvent& event2, int vertical_barrier);
    static double calculateGapSize(const LabeledEvent& event1, const LabeledEvent& event2);
};
