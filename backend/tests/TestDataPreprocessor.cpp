#include <gtest/gtest.h>
#include "../data/DataPreprocessor.h"
#include "../data/DataRow.h"

TEST(DataPreprocessorTest, PreprocessBasic) {
    std::vector<DataRow> rows(5);
    for (int i = 0; i < 5; ++i) {
        rows[i].timestamp = std::to_string(i);
        rows[i].price = 100 + i;
    }
    DataPreprocessor::Params params;
    auto result = DataPreprocessor::preprocess(rows, params);
    EXPECT_EQ(result.size(), rows.size());
    EXPECT_EQ(result[0].timestamp, "0");
    EXPECT_EQ(result[1].timestamp, "1");
    EXPECT_EQ(result[0].log_return, 0.0);
    EXPECT_TRUE(std::isnan(result[0].volatility));
}

TEST(DataPreprocessorTest, EmptyRows) {
    std::vector<DataRow> rows;
    DataPreprocessor::Params params;
    auto result = DataPreprocessor::preprocess(rows, params);
    EXPECT_TRUE(result.empty());
}

TEST(DataPreprocessorTest, OneRow) {
    std::vector<DataRow> rows(1);
    rows[0].timestamp = "t0";
    rows[0].price = 100.0;
    DataPreprocessor::Params params;
    auto result = DataPreprocessor::preprocess(rows, params);
    EXPECT_TRUE(result.empty());
}

TEST(DataPreprocessorTest, LargeDataSet) {
    const int N = 100000;
    std::vector<DataRow> rows(N);
    for (int i = 0; i < N; ++i) {
        rows[i].timestamp = std::to_string(i);
        rows[i].price = 100 + i;
    }
    DataPreprocessor::Params params;
    params.volatility_window = 10;
    auto result = DataPreprocessor::preprocess(rows, params);
    EXPECT_EQ(result.size(), N);
    EXPECT_EQ(result[0].timestamp, "0");
    EXPECT_EQ(result[N-1].timestamp, std::to_string(N-1));
}

TEST(DataPreprocessorTest, CustomParams) {
    std::vector<DataRow> rows(20);
    for (int i = 0; i < 20; ++i) {
        rows[i].timestamp = std::to_string(i);
        rows[i].price = 100 + i;
    }
    DataPreprocessor::Params params;
    params.volatility_window = 5;
    params.barrier_multiple = 3.0;
    params.vertical_barrier = 7;
    auto result = DataPreprocessor::preprocess(rows, params);
    EXPECT_EQ(result.size(), rows.size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(result[i].volatility));
    }
    EXPECT_FALSE(std::isnan(result[4].volatility));
}

TEST(DataPreprocessorTest, DynamicEventSelection) {
    std::vector<DataRow> rows(30);
    for (int i = 0; i < 30; ++i) {
        rows[i].timestamp = std::to_string(i);
        rows[i].price = 100 + i;
    }
    
    DataPreprocessor::Params params;
    params.volatility_window = 5;
    params.vertical_barrier = 12;
    params.barrier_config.labeling_type = BarrierConfig::Hard;
    auto result = DataPreprocessor::preprocess(rows, params);
    
    int events = 0;
    for (const auto& row : result) {
        if (row.is_event) events++;
    }
    
    EXPECT_GT(events, 3);
}
