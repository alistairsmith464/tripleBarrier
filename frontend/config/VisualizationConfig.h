#pragma once

#include <QColor>

// Centralized configuration for visualization parameters
class VisualizationConfig {
public:
    // Chart colors
    static QColor getProfitColor() { return Qt::green; }
    static QColor getStopColor() { return Qt::red; }
    static QColor getNeutralColor() { return Qt::white; }
    static QColor getVerticalBarrierColor() { return Qt::blue; }
    
    // TTBM visualization
    static int getTTBMColorSeriesCount() { return 15; }
    static int getTTBMBinCount() { return 30; }
    static int getMinMarkerSize() { return 6; }
    static int getMaxMarkerSize() { return 15; }
    
    // Trading strategy parameters
    static double getTTBMPositionMultiplier() { return 3.0; }
    static double getHardBarrierPositionSize() { return 2.0; }
    static double getTradingThreshold() { return 0.1; }
    
    // Chart formatting
    static QString getDateTimeFormat() { return "yyyy-MM-dd HH:mm"; }
    
    // TTBM Parameters (moved from UI)
    static double getOptimalLambda() { return 2.0; }
    static double getOptimalAlpha() { return 0.8; }
    static double getOptimalBeta() { return 3.0; }
};
