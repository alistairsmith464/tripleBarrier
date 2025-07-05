#include <gtest/gtest.h>
#include "../data/CUSUMFilter.h"
#include <vector>

TEST(CUSUMFilterTest, DetectsEventsSimple) {
    std::vector<double> prices = {100, 101, 102, 103, 104, 105};
    std::vector<double> vol(prices.size(), 1.0);
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should trigger at index 2 (price diff = 2), then at 4 (next diff = 2)
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events[0], 2);
    EXPECT_EQ(events[1], 4);
}

TEST(CUSUMFilterTest, DetectsEventsNegative) {
    std::vector<double> prices = {100, 99, 98, 97, 96, 95};
    std::vector<double> vol(prices.size(), 1.0);
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should trigger at index 2 and 4 (cumulative negative move)
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events[0], 2);
    EXPECT_EQ(events[1], 4);
}

TEST(CUSUMFilterTest, NoEventsIfBelowThreshold) {
    std::vector<double> prices = {100, 100.5, 101, 101.4, 101.8};
    std::vector<double> vol(prices.size(), 1.0);
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    EXPECT_TRUE(events.empty());
}

TEST(CUSUMFilterTest, HandlesZeroVolatility) {
    std::vector<double> prices = {100, 101, 102};
    std::vector<double> vol = {1.0, 0.0, 0.0};
    double threshold = 1.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should not crash, and should still detect the first event
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events[0], 1);
}

TEST(CUSUMFilterTest, AlternatingUpDownMovements) {
    std::vector<double> prices = {100, 101, 100, 101, 100, 101};
    std::vector<double> vol(prices.size(), 1.0);
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should not trigger any events, as up/down cancels out
    EXPECT_TRUE(events.empty());
}

TEST(CUSUMFilterTest, MixedVolatility) {
    std::vector<double> prices = {100, 101, 103, 106, 110};
    std::vector<double> vol = {1.0, 1.0, 2.0, 2.0, 1.0};
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should trigger at index 2 (cumulative = 3), then at 4 (cumulative = 4)
    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events[0], 2);
    EXPECT_EQ(events[1], 4);
}

TEST(CUSUMFilterTest, ThresholdEdge) {
    std::vector<double> prices = {100, 101, 102};
    std::vector<double> vol(prices.size(), 1.0);
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should trigger at index 2 (cumulative = 2)
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0], 2);
}

TEST(CUSUMFilterTest, EventResetBehavior) {
    std::vector<double> prices = {100, 102, 104, 106};
    std::vector<double> vol(prices.size(), 1.0);
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should trigger at 1 (diff=2), reset, then at 2 (diff=2), reset, then at 3 (diff=2)
    ASSERT_EQ(events.size(), 3);
    EXPECT_EQ(events[0], 1);
    EXPECT_EQ(events[1], 2);
    EXPECT_EQ(events[2], 3);
}

TEST(CUSUMFilterTest, SingleLargeJump) {
    std::vector<double> prices = {100, 100.1, 105};
    std::vector<double> vol(prices.size(), 1.0);
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should trigger at index 2 (jump from 100.1 to 105)
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0], 2);
}

TEST(CUSUMFilterTest, MultipleEventsSameDirection) {
    std::vector<double> prices = {100, 102, 104, 106, 108};
    std::vector<double> vol(prices.size(), 1.0);
    double threshold = 2.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should trigger at 1, 2, 3, 4
    ASSERT_EQ(events.size(), 4);
    EXPECT_EQ(events[0], 1);
    EXPECT_EQ(events[1], 2);
    EXPECT_EQ(events[2], 3);
    EXPECT_EQ(events[3], 4);
}

TEST(CUSUMFilterTest, AllZeroVolatility) {
    std::vector<double> prices = {100, 101, 102};
    std::vector<double> vol(prices.size(), 0.0);
    double threshold = 1.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    // Should not crash, and should not trigger any events
    EXPECT_TRUE(events.empty());
}

TEST(CUSUMFilterTest, ShortInput) {
    std::vector<double> prices = {100};
    std::vector<double> vol = {1.0};
    double threshold = 1.0;
    auto events = CUSUMFilter::detect(prices, vol, threshold);
    EXPECT_TRUE(events.empty());
    prices = {100, 101};
    vol = {1.0, 1.0};
    events = CUSUMFilter::detect(prices, vol, threshold);
    // Should trigger at index 1
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0], 1);
}
