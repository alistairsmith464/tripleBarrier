#include <gtest/gtest.h>
#include "../data/PreprocessedRow.h"

TEST(PreprocessedRowTest, DefaultValues) {
    PreprocessedRow row;
    EXPECT_EQ(row.log_return, 0.0);
    EXPECT_EQ(row.volatility, 0.0);
    EXPECT_FALSE(row.is_event);
    EXPECT_FALSE(row.open.has_value());
    EXPECT_FALSE(row.high.has_value());
    EXPECT_FALSE(row.low.has_value());
    EXPECT_FALSE(row.close.has_value());
    EXPECT_FALSE(row.volume.has_value());
}

TEST(PreprocessedRowTest, SetValues) {
    PreprocessedRow row;
    row.timestamp = "2025-07-02";
    row.price = 123.45;
    row.open = 120.0;
    row.high = 130.0;
    row.low = 119.0;
    row.close = 125.0;
    row.volume = 1000.0;
    row.log_return = 0.01;
    row.volatility = 0.02;
    row.is_event = true;
    EXPECT_EQ(row.timestamp, "2025-07-02");
    EXPECT_EQ(row.price, 123.45);
    EXPECT_TRUE(row.open.has_value());
    EXPECT_EQ(row.open.value(), 120.0);
    EXPECT_TRUE(row.high.has_value());
    EXPECT_EQ(row.high.value(), 130.0);
    EXPECT_TRUE(row.low.has_value());
    EXPECT_EQ(row.low.value(), 119.0);
    EXPECT_TRUE(row.close.has_value());
    EXPECT_EQ(row.close.value(), 125.0);
    EXPECT_TRUE(row.volume.has_value());
    EXPECT_EQ(row.volume.value(), 1000.0);
    EXPECT_EQ(row.log_return, 0.01);
    EXPECT_EQ(row.volatility, 0.02);
    EXPECT_TRUE(row.is_event);
}

TEST(PreprocessedRowTest, LargeDataSet) {
    const int N = 100000;
    std::vector<PreprocessedRow> rows(N);
    for (int i = 0; i < N; ++i) {
        rows[i].timestamp = std::to_string(i);
        rows[i].price = i * 1.0;
    }
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(rows[i].timestamp, std::to_string(i));
        EXPECT_EQ(rows[i].price, i * 1.0);
    }
}
