#pragma once
#include <QString>
#include <QSet>
#include <vector>
#include <string>
#include <set>
#include "../backend/data/FeatureExtractor.h"
#include "../backend/data/PortfolioSimulator.h"

// Utility class for formatting display text in feature preview dialogs
class FeaturePreviewUtils {
public:
    // Convert Qt types to std types for backend
    static std::set<std::string> convertQSetToStdSet(const QSet<QString>& qset);
    
    // Generate formatted display text for barrier diagnostics
    static QString formatBarrierDiagnostics(
        const BarrierDiagnostics& diagnostics,
        const std::vector<LabeledEvent>& labeledEvents
    );
    
    // Generate formatted display text for portfolio results
    static QString formatPortfolioResults(
        const PortfolioResults& results,
        bool is_ttbm
    );
    
    // Generate formatted display text for trading strategy description
    static QString formatTradingStrategy(bool is_ttbm);
    
    // Generate formatted display text for model information
    static QString formatModelInfo(
        bool is_ttbm,
        bool tune_enabled,
        const std::vector<LabeledEvent>& labeledEvents
    );
    
    // Generate sample trading decisions debug text
    static QString formatSampleTradingDecisions(
        const std::vector<double>& predictions,
        bool is_ttbm,
        int max_samples = 10
    );
};
