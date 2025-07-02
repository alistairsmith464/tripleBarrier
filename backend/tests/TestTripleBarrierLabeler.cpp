#include <gtest/gtest.h>
#include "../data/TripleBarrierLabeler.h"
#include "../data/PreprocessedRow.h"

TEST(TripleBarrierLabelerTest, ProfitHitFirst) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }

    data[2].price = 102.1;
    std::vector<size_t> events = {0};
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[0].exit_time, "2");
}

TEST(TripleBarrierLabelerTest, StopHitFirst) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    data[3].price = 98.9;
    std::vector<size_t> events = {0};
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, -1);
    EXPECT_EQ(result[0].exit_time, "3");
}

TEST(TripleBarrierLabelerTest, VerticalBarrierOnly) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {0};
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, 0);
    EXPECT_EQ(result[0].exit_time, "4");
}

TEST(TripleBarrierLabelerTest, MultipleEvents) {
    std::vector<PreprocessedRow> data(10);
    for (int i = 0; i < 10; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    data[2].price = 102.1;
    data[7].price = 98.9;
    std::vector<size_t> events = {0, 5};
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[1].label, -1);
}

TEST(TripleBarrierLabelerTest, EdgeCases) {
    std::vector<PreprocessedRow> data(3);
    for (int i = 0; i < 3; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {2};
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, 0);
    EXPECT_EQ(result[0].exit_time, "2");
}

TEST(TripleBarrierLabelerTest, ProfitAndStopSameBar) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    // Both barriers hit on the same bar: price >= pt and <= sl
    // pt = 100 + 2*1 = 102, sl = 100 - 1*1 = 99
    // Set price to 102 (profit) and 99 (stop) on the same bar is not possible for a single price,
    // but we can set price to 102 (profit) and test that profit takes precedence if both are hit on the same bar.
    data[2].price = 102.0; // hits profit barrier
    data[2].volatility = 1.0;
    // To simulate both, we can set stop_multiple so that sl = 102, so price == pt == sl
    auto result = TripleBarrierLabeler::label(data, {0}, 2.0, -2.0, 4); // sl = 100 - (-2)*1 = 102
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[0].exit_time, "2");
}

TEST(TripleBarrierLabelerTest, EventIndexOutOfBounds) {
    std::vector<PreprocessedRow> data(3);
    for (int i = 0; i < 3; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {5}; // out of bounds
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 0);
}

TEST(TripleBarrierLabelerTest, ZeroVolatility) {
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 0.0;
    }
    data[3].price = 100.0;
    std::vector<size_t> events = {0};
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 4);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[0].exit_time, "1");
}

TEST(TripleBarrierLabelerTest, LargeDataSet) {
    const int N = 10000;
    std::vector<PreprocessedRow> data(N);
    for (int i = 0; i < N; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0 + i * 0.01;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {0, 100, 5000};
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 100);
    ASSERT_EQ(result.size(), 3);
    for (const auto& r : result) {
        EXPECT_EQ(r.label, +1);
    }
}

TEST(TripleBarrierLabelerTest, NegativePriceMovement) {
    std::vector<PreprocessedRow> data(10);
    for (int i = 0; i < 10; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0 - i * 2.0;
        data[i].volatility = 1.0;
    }
    std::vector<size_t> events = {0};
    auto result = TripleBarrierLabeler::label(data, events, 2.0, 1.0, 9);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, -1);
    EXPECT_EQ(result[0].exit_time, "1");
}
