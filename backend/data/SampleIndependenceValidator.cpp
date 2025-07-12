#include "SampleIndependenceValidator.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>

SampleIndependenceValidator::IndependenceReport 
SampleIndependenceValidator::validateSampleIndependence(
    const std::vector<LabeledEvent>& events,
    int vertical_barrier,
    int min_gap_requirement
) {
    IndependenceReport report;
    report.total_samples = events.size();
    report.overlapping_samples = 0;
    report.gap_violations = 0;
    report.overlap_percentage = 0.0;
    report.avg_gap_size = 0.0;
    report.min_gap_size = std::numeric_limits<double>::max();
    report.max_gap_size = 0.0;
    report.independence_violated = false;
    
    if (events.size() < 2) {
        return report;
    }
    
    std::vector<double> gaps;
    gaps.reserve(events.size() - 1);
    
    if (min_gap_requirement == -1) {
        min_gap_requirement = vertical_barrier;
    }
    
    for (size_t i = 0; i < events.size(); ++i) {
        for (size_t j = i + 1; j < events.size(); ++j) {
            if (hasTemporalOverlap(events[i], events[j], vertical_barrier)) {
                report.overlapping_samples++;
                report.independence_violated = true;
            }
            
            double gap = calculateGapSize(events[i], events[j]);
            gaps.push_back(gap);
            
            if (gap < min_gap_requirement) {
                report.gap_violations++;
                report.independence_violated = true;
            }
            
            report.min_gap_size = std::min(report.min_gap_size, gap);
            report.max_gap_size = std::max(report.max_gap_size, gap);
        }
    }
    
    if (!gaps.empty()) {
        report.avg_gap_size = std::accumulate(gaps.begin(), gaps.end(), 0.0) / gaps.size();
    }
    
    report.overlap_percentage = (100.0 * report.overlapping_samples) / report.total_samples;
    
    return report;
}

std::vector<size_t> SampleIndependenceValidator::findOverlappingEventPairs(
    const std::vector<LabeledEvent>& events,
    int vertical_barrier
) {
    std::vector<size_t> overlapping_pairs;
    
    for (size_t i = 0; i < events.size(); ++i) {
        for (size_t j = i + 1; j < events.size(); ++j) {
            if (hasTemporalOverlap(events[i], events[j], vertical_barrier)) {
                overlapping_pairs.push_back(i);
                overlapping_pairs.push_back(j);
            }
        }
    }
    
    return overlapping_pairs;
}

std::vector<size_t> SampleIndependenceValidator::findGapViolations(
    const std::vector<LabeledEvent>& events,
    int min_gap_requirement
) {
    std::vector<size_t> violations;
    
    for (size_t i = 0; i < events.size(); ++i) {
        for (size_t j = i + 1; j < events.size(); ++j) {
            double gap = calculateGapSize(events[i], events[j]);
            if (gap < min_gap_requirement) {
                violations.push_back(i);
                violations.push_back(j);
            }
        }
    }
    
    return violations;
}

bool SampleIndependenceValidator::hasTemporalOverlap(
    const LabeledEvent& event1, 
    const LabeledEvent& event2, 
    int vertical_barrier
) {
    return (abs(event1.periods_to_exit - event2.periods_to_exit) < vertical_barrier);
}

double SampleIndependenceValidator::calculateGapSize(
    const LabeledEvent& event1, 
    const LabeledEvent& event2
) {
    return abs(event1.periods_to_exit - event2.periods_to_exit);
}
