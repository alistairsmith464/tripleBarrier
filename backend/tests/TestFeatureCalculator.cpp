#include <gtest/gtest.h>
#include "../data/FeatureCalculator.h"
#include <vector>
#include <string>
#include <cmath>

TEST(FeatureCalculatorTest, CloseToCloseReturn1D) {
    std::vector<double> prices = {100, 105};
    EXPECT_NEAR(FeatureCalculator::closeToCloseReturn1D(prices, 1), 0.05, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::closeToCloseReturn1D(prices, 0)));
}

TEST(FeatureCalculatorTest, ReturnND) {
    std::vector<double> prices = {100, 102, 104, 108};
    EXPECT_NEAR(FeatureCalculator::returnND(prices, 3, 3), 0.08, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::returnND(prices, 2, 3)));
}

TEST(FeatureCalculatorTest, RollingStdND) {
    std::vector<double> prices = {1, 2, 3, 4, 5};
    double std = FeatureCalculator::rollingStdND(prices, 5, 5);
    EXPECT_NEAR(std, std::sqrt(2.0), 1e-6);
}

TEST(FeatureCalculatorTest, EWMA) {
    std::vector<double> prices = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    double ewma = FeatureCalculator::ewmaVolND(prices, 10, 10, 0.94);
    EXPECT_TRUE(ewma > 0);
}

TEST(FeatureCalculatorTest, SMA) {
    std::vector<double> prices = {1, 2, 3, 4, 5};
    EXPECT_NEAR(FeatureCalculator::smaND(prices, 4, 5), 3.0, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::smaND(prices, 2, 5)));
}

TEST(FeatureCalculatorTest, DistToSMA) {
    std::vector<double> prices = {1, 2, 3, 4, 5};
    EXPECT_NEAR(FeatureCalculator::distToSMA(prices, 4, 5), 2.0, 1e-6);
}

TEST(FeatureCalculatorTest, ROC) {
    std::vector<double> prices = {100, 105};
    EXPECT_NEAR(FeatureCalculator::rocND(prices, 1, 1), 5.0, 1e-6);
}

TEST(FeatureCalculatorTest, RSI) {
    std::vector<double> prices = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114};
    double rsi = FeatureCalculator::rsiND(prices, 14, 14);
    EXPECT_TRUE(rsi > 50);
}

TEST(FeatureCalculatorTest, PriceRange) {
    std::vector<double> prices = {1, 2, 3, 4, 5};
    EXPECT_NEAR(FeatureCalculator::priceRangeND(prices, 4, 5), 4.0, 1e-6);
}

TEST(FeatureCalculatorTest, CloseOverHigh) {
    std::vector<double> prices = {1, 2, 3, 4, 5};
    EXPECT_NEAR(FeatureCalculator::closeOverHighND(prices, 4, 5), 1.0, 1e-6);
}

TEST(FeatureCalculatorTest, SlopeLR) {
    std::vector<double> prices = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double slope = FeatureCalculator::slopeLRND(prices, 9, 10);
    EXPECT_NEAR(slope, 1.0, 1e-6);
}

TEST(FeatureCalculatorTest, DayOfWeek) {
    std::vector<std::string> timestamps = {"2023-07-03", "2023-07-04"};
    EXPECT_GE(FeatureCalculator::dayOfWeek(timestamps, 0), 0);
}

TEST(FeatureCalculatorTest, CalculateFeatures) {
    std::vector<double> prices = {100, 101, 102, 103, 104, 105};
    std::vector<std::string> timestamps = {"2023-07-01", "2023-07-02", "2023-07-03", "2023-07-04", "2023-07-05", "2023-07-06"};
    std::vector<int> eventIndices = {5};
    std::set<std::string> feats = {FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D, FeatureCalculator::SMA_5D};
    auto result = FeatureCalculator::calculateFeatures(prices, timestamps, eventIndices, 0, feats);
    EXPECT_TRUE(result.count(FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D));
    EXPECT_TRUE(result.count(FeatureCalculator::SMA_5D));
}

TEST(FeatureCalculatorTest, CloseToCloseReturn1D_Detailed) {
    std::vector<double> prices = {100, 105, 110};
    EXPECT_NEAR(FeatureCalculator::closeToCloseReturn1D(prices, 1), 0.05, 1e-6);
    EXPECT_NEAR(FeatureCalculator::closeToCloseReturn1D(prices, 2), 0.047619, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::closeToCloseReturn1D(prices, 0)));
}

TEST(FeatureCalculatorTest, ReturnND_Detailed) {
    std::vector<double> prices = {100, 102, 104, 108, 112};
    EXPECT_NEAR(FeatureCalculator::returnND(prices, 4, 4), 0.12, 1e-6);
    EXPECT_NEAR(FeatureCalculator::returnND(prices, 3, 2), 0.0384615, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::returnND(prices, 1, 3)));
}

TEST(FeatureCalculatorTest, RollingStdND_Detailed) {
    std::vector<double> prices = {1, 2, 3, 4, 5, 6};
    EXPECT_NEAR(FeatureCalculator::rollingStdND(prices, 5, 5), std::sqrt(2.0), 1e-6);
    EXPECT_NEAR(FeatureCalculator::rollingStdND(prices, 6, 3), std::sqrt(2.0/3.0), 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::rollingStdND(prices, 2, 5)));
}

TEST(FeatureCalculatorTest, EWMA_Detailed) {
    std::vector<double> prices = {1,2,3,4,5,6,7,8,9,10,11};
    double ewma1 = FeatureCalculator::ewmaVolND(prices, 10, 10, 0.94);
    double ewma2 = FeatureCalculator::ewmaVolND(prices, 10, 5, 0.90);
    EXPECT_TRUE(ewma1 > 0);
    EXPECT_TRUE(ewma2 > 0);
    EXPECT_TRUE(std::isnan(FeatureCalculator::ewmaVolND(prices, 3, 5)));
}

TEST(FeatureCalculatorTest, SMA_Detailed) {
    std::vector<double> prices = {1,2,3,4,5,6};
    EXPECT_NEAR(FeatureCalculator::smaND(prices, 5, 5), 4.0, 1e-6);
    EXPECT_NEAR(FeatureCalculator::smaND(prices, 4, 3), 4.0, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::smaND(prices, 2, 5)));
}

TEST(FeatureCalculatorTest, DistToSMA_Detailed) {
    std::vector<double> prices = {1,2,3,4,5,6};
    EXPECT_NEAR(FeatureCalculator::distToSMA(prices, 5, 5), 2.0, 1e-6);
    EXPECT_NEAR(FeatureCalculator::distToSMA(prices, 4, 3), 2.0, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::distToSMA(prices, 2, 5)));
}

TEST(FeatureCalculatorTest, ROC_Detailed) {
    std::vector<double> prices = {100, 105, 110};
    EXPECT_NEAR(FeatureCalculator::rocND(prices, 1, 1), 5.0, 1e-6);
    EXPECT_NEAR(FeatureCalculator::rocND(prices, 2, 2), 10.0, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::rocND(prices, 0, 1)));
}

TEST(FeatureCalculatorTest, RSI_Detailed) {
    std::vector<double> prices = {100,101,102,103,104,105,106,107,108,109,110,111,112,113,114};
    double rsi = FeatureCalculator::rsiND(prices, 14, 14);
    EXPECT_TRUE(rsi > 50);
    std::vector<double> flat = {100,100,100,100,100,100,100,100,100,100,100,100,100,100,100};
    EXPECT_NEAR(FeatureCalculator::rsiND(flat, 14, 14), 50.0, 1e-6);
    std::vector<double> down = {100,99,98,97,96,95,94,93,92,91,90,89,88,87,86};
    EXPECT_TRUE(FeatureCalculator::rsiND(down, 14, 14) < 50);
}

TEST(FeatureCalculatorTest, PriceRange_Detailed) {
    std::vector<double> prices = {1,2,3,4,5,6};
    EXPECT_NEAR(FeatureCalculator::priceRangeND(prices, 5, 5), 4.0, 1e-6);
    EXPECT_NEAR(FeatureCalculator::priceRangeND(prices, 4, 3), 2.0, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::priceRangeND(prices, 2, 5)));
}

TEST(FeatureCalculatorTest, CloseOverHigh_Detailed) {
    std::vector<double> prices = {1,2,3,4,5,6};
    EXPECT_NEAR(FeatureCalculator::closeOverHighND(prices, 5, 5), 1.0, 1e-6);
    EXPECT_NEAR(FeatureCalculator::closeOverHighND(prices, 4, 3), 1.0, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::closeOverHighND(prices, 2, 5)));
}

TEST(FeatureCalculatorTest, SlopeLR_Detailed) {
    std::vector<double> prices = {1,2,3,4,5,6,7,8,9,10};
    double slope = FeatureCalculator::slopeLRND(prices, 9, 10);
    EXPECT_NEAR(slope, 1.0, 1e-6);
    std::vector<double> flat = {5,5,5,5,5,5,5,5,5,5};
    EXPECT_NEAR(FeatureCalculator::slopeLRND(flat, 9, 10), 0.0, 1e-6);
    EXPECT_TRUE(std::isnan(FeatureCalculator::slopeLRND(prices, 2, 5)));
}

TEST(FeatureCalculatorTest, DayOfWeek_Detailed) {
    std::vector<std::string> timestamps = {"2023-07-03", "2023-07-04", "2023-07-05"};
    int dow0 = FeatureCalculator::dayOfWeek(timestamps, 0);
    int dow1 = FeatureCalculator::dayOfWeek(timestamps, 1);
    int dow2 = FeatureCalculator::dayOfWeek(timestamps, 2);
    EXPECT_GE(dow0, 0);
    EXPECT_GE(dow1, 0);
    EXPECT_GE(dow2, 0);
}

TEST(FeatureCalculatorTest, CalculateFeatures_Detailed) {
    std::vector<double> prices = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    std::vector<std::string> timestamps = {"2023-07-01", "2023-07-02", "2023-07-03", "2023-07-04", "2023-07-05", "2023-07-06", "2023-07-07", "2023-07-08", "2023-07-09", "2023-07-10"};
    std::vector<int> eventIndices = {5, 7, 9};
    std::set<std::string> feats = {FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D, FeatureCalculator::SMA_5D, FeatureCalculator::RETURN_5D};
    for (int i = 0; i < 3; ++i) {
        auto result = FeatureCalculator::calculateFeatures(prices, timestamps, eventIndices, i, feats);
        EXPECT_TRUE(result.count(FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D));
        EXPECT_TRUE(result.count(FeatureCalculator::SMA_5D));
        EXPECT_TRUE(result.count(FeatureCalculator::RETURN_5D));
    }
}
