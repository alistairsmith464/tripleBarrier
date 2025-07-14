#pragma once

#include <QColor>

class VisualizationConfig {
public:
    static QColor getProfitColor() { return Qt::green; }
    static QColor getStopColor() { return Qt::red; }
    static QColor getNeutralColor() { return Qt::white; }
    static QColor getVerticalBarrierColor() { return Qt::blue; }
    
    static int getTTBMColorSeriesCount() { return 15; }
    static int getTTBMBinCount() { return 30; }
    static int getMinMarkerSize() { return 6; }
    static int getMaxMarkerSize() { return 15; }
    
    static double getTTBMPositionMultiplier() { return 5.0; }
    static double getHardBarrierPositionSize() { return 25.0; }
    static double getTradingThreshold() { return 0.25; }
    
    static QString getDateTimeFormat() { return "yyyy-MM-dd HH:mm"; }
    
    static double getOptimalLambda() { return 2.0; }
    static double getOptimalAlpha() { return 0.8; }
    static double getOptimalBeta() { return 3.0; }
};
