#include <gtest/gtest.h>
#include "../data/VolatilityCalculator.h"

TEST(VolatilityCalculatorTest, RollingStdDevBasic) {
    std::vector<double> logReturns = {0, 1, 2, 3, 4};
    int window = 3;
    auto result = VolatilityCalculator::rollingStdDev(logReturns, window);
    EXPECT_EQ(result.size(), logReturns.size());
    // First window-1 should be NaN
    for (int i = 0; i < window-1; ++i) {
        EXPECT_TRUE(std::isnan(result[i]));
    }
    // The rest should be valid numbers
    for (size_t i = window-1; i < result.size(); ++i) {
        EXPECT_FALSE(std::isnan(result[i]));
    }
}

TEST(VolatilityCalculatorTest, WindowLargerThanData) {
    std::vector<double> logReturns = {1, 2};
    int window = 5;
    auto result = VolatilityCalculator::rollingStdDev(logReturns, window);
    EXPECT_EQ(result.size(), logReturns.size());
    for (auto v : result) EXPECT_TRUE(std::isnan(v));
}

TEST(VolatilityCalculatorTest, WindowIsOne) {
    std::vector<double> logReturns = {1, 2, 3};
    int window = 1;
    auto result = VolatilityCalculator::rollingStdDev(logReturns, window);
    for (auto v : result) EXPECT_TRUE(std::isnan(v));
}

TEST(VolatilityCalculatorTest, AllZeros) {
    std::vector<double> logReturns(10, 0.0);
    int window = 5;
    auto result = VolatilityCalculator::rollingStdDev(logReturns, window);
    for (size_t i = window-1; i < result.size(); ++i) {
        EXPECT_EQ(result[i], 0.0);
    }
}

TEST(VolatilityCalculatorTest, LargeDataSet) {
    const int N = 100000;
    std::vector<double> logReturns(N, 1.0);
    int window = 1000;
    auto result = VolatilityCalculator::rollingStdDev(logReturns, window);
    EXPECT_EQ(result.size(), N);
    for (int i = 0; i < window-1; ++i) EXPECT_TRUE(std::isnan(result[i]));
    for (int i = window-1; i < N; ++i) EXPECT_EQ(result[i], 0.0);
}
