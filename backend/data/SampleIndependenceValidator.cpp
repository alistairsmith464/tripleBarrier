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

void SampleIndependenceValidator::logIndependenceReport(const IndependenceReport& report) {
    std::cout << "==================== SAMPLE INDEPENDENCE REPORT ====================" << std::endl;
    std::cout << "Total samples: " << report.total_samples << std::endl;
    std::cout << "Overlapping samples: " << report.overlapping_samples << std::endl;
    std::cout << "Gap violations: " << report.gap_violations << std::endl;
    std::cout << "Overlap percentage: " << report.overlap_percentage << "%" << std::endl;
    std::cout << "Average gap size: " << report.avg_gap_size << std::endl;
    std::cout << "Min gap size: " << report.min_gap_size << std::endl;
    std::cout << "Max gap size: " << report.max_gap_size << std::endl;
    std::cout << "Independence violated: " << (report.independence_violated ? "YES" : "NO") << std::endl;
    
    if (report.independence_violated) {
        std::cout << "[WARNING] Sample independence assumption VIOLATED!" << std::endl;
        std::cout << "[WARNING] This may lead to overfitting and poor generalization!" << std::endl;
    } else {
        std::cout << "[INFO] Sample independence assumption satisfied." << std::endl;
    }
    std::cout << "======================================================================" << std::endl;
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
