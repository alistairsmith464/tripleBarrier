#pragma once
#include <vector>
#include <string>
#include <map>
#include <set>

class FeatureCalculator {
public:
    static const std::string CLOSE_TO_CLOSE_RETURN_1D;
    static const std::string RETURN_5D;
    static const std::string RETURN_10D;
    static const std::string ROLLING_STD_5D;
    static const std::string EWMA_VOL_10D;
    static const std::string SMA_5D;
    static const std::string SMA_10D;
    static const std::string SMA_20D;
    static const std::string DIST_TO_SMA_5D;
    static const std::string ROC_5D;
    static const std::string RSI_14D;
    static const std::string PRICE_RANGE_5D;
    static const std::string CLOSE_OVER_HIGH_5D;
    static const std::string SLOPE_LR_10D;
    static const std::string DAY_OF_WEEK;
    static const std::string DAYS_SINCE_LAST_EVENT;

    static std::map<std::string, double> calculateFeatures(
        const std::vector<double>& prices,
        const std::vector<std::string>& timestamps,
        const std::vector<int>& eventIndices,
        int eventIdx,
        const std::set<std::string>& selectedFeatures,
        const std::vector<int>* eventStarts = nullptr
    );

    static double closeToCloseReturn1D(const std::vector<double>& prices, int idx);
    static double returnND(const std::vector<double>& prices, int idx, int n);
    static double rollingStdND(const std::vector<double>& prices, int idx, int n);
    static double ewmaVolND(const std::vector<double>& prices, int idx, int n, double alpha=0.94);
    static double smaND(const std::vector<double>& prices, int idx, int n);
    static double distToSMA(const std::vector<double>& prices, int idx, int n);
    static double rocND(const std::vector<double>& prices, int idx, int n);
    static double rsiND(const std::vector<double>& prices, int idx, int n);
    static double priceRangeND(const std::vector<double>& prices, int idx, int n);
    static double closeOverHighND(const std::vector<double>& prices, int idx, int n);
    static double slopeLRND(const std::vector<double>& prices, int idx, int n);
    static int dayOfWeek(const std::vector<std::string>& timestamps, int idx);
    static int daysSinceLastEvent(const std::vector<int>& eventIndices, int idx);
};
