#include <gtest/gtest.h>
#include "../data/HardBarrierLabeler.h"
#include "../data/PreprocessedRow.h"
#include "../data/IBarrierLabeler.h"
#include <cmath>

TEST(HardBarrierLabelerTest, ProfitHitFirst) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    data[2].price = 102.1;
    std::vector<size_t> events = {0};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[0].exit_time, "2");
}

TEST(HardBarrierLabelerTest, StopHitFirst) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    data[3].price = 98.9;
    std::vector<size_t> events = {0};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, -1);
    EXPECT_EQ(result[0].exit_time, "3");
}

TEST(HardBarrierLabelerTest, VerticalBarrierOnly) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {0};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, 0);
    EXPECT_EQ(result[0].exit_time, "4");
}

TEST(HardBarrierLabelerTest, MultipleEvents) {
    std::vector<PreprocessedRow> data(10);
    for (int i = 0; i < 10; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    data[2].price = 102.1;
    data[7].price = 98.9;
    std::vector<size_t> events = {0, 5};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[1].label, -1);
}

TEST(HardBarrierLabelerTest, EdgeCases) {
    std::vector<PreprocessedRow> data(3);
    for (int i = 0; i < 3; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {2};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, 0);
    EXPECT_EQ(result[0].exit_time, "2");
}

TEST(HardBarrierLabelerTest, ProfitAndStopSameBar) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    data[2].price = 102.0;
    data[2].volatility = 1.0;
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, {0}, 2.0, -2.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[0].exit_time, "2");
}

TEST(HardBarrierLabelerTest, EventIndexOutOfBounds) {
    std::vector<PreprocessedRow> data(3);
    for (int i = 0; i < 3; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {5};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 0);
}

TEST(HardBarrierLabelerTest, ZeroVolatility) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 0.0;
    }
    data[3].price = 100.0;
    std::vector<size_t> events = {0};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[0].exit_time, "1");
}

TEST(HardBarrierLabelerTest, LargeDataSet) {
    const int N = 10000;
    std::vector<PreprocessedRow> data(N);
    for (int i = 0; i < N; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0 + i * 0.01;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {0, 100, 5000};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 1.0, 1.0, 100);
    ASSERT_EQ(result.size(), 3);
    for (const auto& r : result) {
        EXPECT_EQ(r.label, +1);
    }
}

TEST(HardBarrierLabelerTest, NegativePriceMovement) {
    std::vector<PreprocessedRow> data(10);
    for (int i = 0; i < 10; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0 - i * 2.0;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {0};
    HardBarrierLabeler labeler;
    auto result = labeler.label(data, events, 2.0, 1.0, 9);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, -1);
    EXPECT_EQ(result[0].exit_time, "1");
}

