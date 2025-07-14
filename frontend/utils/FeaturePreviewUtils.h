#pragma once
#include <QSet>
#include <vector>
#include <string>
#include <set>
#include "../backend/data/FeatureExtractor.h"
#include "../backend/data/LabeledEvent.h"
#include "../backend/ml/PortfolioSimulator.h"

class FeaturePreviewUtils {
public:
    static std::set<std::string> convertQSetToStdSet(const QSet<QString>& qset);
    
    static QString formatBarrierDiagnostics(
        const MLPipeline::BarrierDiagnostics& diagnostics,
        const std::vector<LabeledEvent>& labeledEvents
    );
    
    static QString formatPortfolioResults(
        const MLPipeline::PortfolioResults& results,
        bool is_ttbm
    );
    
    static QString formatTradingStrategy(bool is_ttbm);
    
    static QString formatModelInfo(
        bool is_ttbm,
        bool tune_enabled,
        const std::vector<LabeledEvent>& labeledEvents
    );
};
