#include <gtest/gtest.h>
#include "../data/FeatureCalculator.h"
#include "../data/FeatureExtractor.h"
#include "../data/PreprocessedRow.h"
#include "../data/LabeledEvent.h"
#include <set>
#include <string>
#include <vector>

using namespace std;

// Helper to create dummy data
PreprocessedRow makeRow(double price, const string& ts) {
    PreprocessedRow row;
    row.price = price;
    row.timestamp = ts;
    row.volatility = 1.0;
    return row;
}

LabeledEvent makeEvent(int label, const std::string& ts = "", double entry_price = 100.0, double exit_price = 110.0) {
    LabeledEvent e;
    e.label = label;
    e.entry_time = ts;
    e.entry_price = entry_price;
    e.exit_price = exit_price;
    return e;
}

// ------------------ CLOSE_TO_CLOSE_RETURN_1D ------------------
TEST(FeatureExtractorTest, CloseToCloseReturn1D_Positive) {
    vector<PreprocessedRow> rows = {makeRow(100, "2021-01-01"), makeRow(110, "2021-01-02")};
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-02", 110, 110)};
    set<string> features = {"Close-to-close return for the previous day"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D], 0.1, 0.01);
}

TEST(FeatureExtractorTest, CloseToCloseReturn1D_Negative) {
    vector<PreprocessedRow> rows = {makeRow(100, "2021-01-01"), makeRow(90, "2021-01-02")};
    vector<LabeledEvent> events = {makeEvent(-1, "2021-01-02", 90, 90)};
    set<string> features = {"Close-to-close return for the previous day"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D], -0.1, 0.01);
}

TEST(FeatureExtractorTest, CloseToCloseReturn1D_Zero) {
    vector<PreprocessedRow> rows = {makeRow(100, "2021-01-01"), makeRow(100, "2021-01-02")};
    vector<LabeledEvent> events = {makeEvent(0, "2021-01-02", 100, 100)};
    set<string> features = {"Close-to-close return for the previous day"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D], 0.0, 0.01);
}

// ------------------ SMA_5D ------------------
TEST(FeatureExtractorTest, SMA5D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) rows.push_back(makeRow(100 + i, "2021-01-0" + to_string(i+1)));
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 104, 104)};
    set<string> features = {"5-day simple moving average (SMA)"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::SMA_5D], 102.0);
}

TEST(FeatureExtractorTest, SMA5D_Constant) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) rows.push_back(makeRow(50, "2021-01-0" + to_string(i+1)));
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 50, 50)};
    set<string> features = {"5-day simple moving average (SMA)"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::SMA_5D], 50.0);
}

TEST(FeatureExtractorTest, SMA5D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) rows.push_back(makeRow(10 * (i+1), "2021-01-0" + to_string(i+1)));
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 50, 50)};
    set<string> features = {"5-day simple moving average (SMA)"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::SMA_5D], 30.0);
}

// ------------------ ROLLING_STD_5D ------------------
TEST(FeatureExtractorTest, RollingStd5D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 6; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + i * 2, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-06", 110, 110)};
    set<string> features = {"Rolling standard deviation of daily returns over the last 5 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // With prices: 100, 102, 104, 106, 108, 110, std of last 5 prices = sqrt(8) ≈ 2.8284271247461903
    EXPECT_NEAR(result.features[0][FeatureCalculator::ROLLING_STD_5D], 2.8284271247461903, 0.01);
}

TEST(FeatureExtractorTest, RollingStd5D_Constant) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 6; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-06", 100, 100)};
    set<string> features = {"Rolling standard deviation of daily returns over the last 5 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::ROLLING_STD_5D], 0.0, 0.01);
}

TEST(FeatureExtractorTest, RollingStd5D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 6; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(10 * (i+1), "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-06", 60, 60)};
    set<string> features = {"Rolling standard deviation of daily returns over the last 5 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // Prices: 10, 20, 30, 40, 50, 60. Std of last 5 prices = sqrt(200) ≈ 14.1421356237
    EXPECT_NEAR(result.features[0][FeatureCalculator::ROLLING_STD_5D], 14.1421356237, 0.01);
}

// ------------------ EWMA_VOL_10D ------------------
TEST(FeatureExtractorTest, EwmaVol10D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 11; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-11", 100, 100)};
    set<string> features = {"EWMA volatility over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // All prices are constant, so EWMA volatility should be 0
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::EWMA_VOL_10D], 0.0);
}

TEST(FeatureExtractorTest, EwmaVol10D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 11; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-11", 100, 100)};
    set<string> features = {"EWMA volatility over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::EWMA_VOL_10D], 0.0);
}

// ------------------ EWMA_VOL_10D (Additional Tests) ------------------
// Alternating prices (high volatility)
TEST(FeatureExtractorTest, EwmaVol10D_AlternatingPrices) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 10; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + (i % 2) * 10, "2021-01-" + day));
    }
    rows.push_back(makeRow(110, "2021-01-11"));
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-11", 110, 110)};
    set<string> features = {"EWMA volatility over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_GT(result.features[0][FeatureCalculator::EWMA_VOL_10D], 0.05);
}

// Single spike in price
TEST(FeatureExtractorTest, EwmaVol10D_SingleSpike) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 10; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    rows.push_back(makeRow(200, "2021-01-11")); // Big spike at end
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-11", 200, 200)};
    set<string> features = {"EWMA volatility over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_GT(result.features[0][FeatureCalculator::EWMA_VOL_10D], 0.5);
}

// Single drop in price
TEST(FeatureExtractorTest, EwmaVol10D_SingleDrop) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 10; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    rows.push_back(makeRow(50, "2021-01-11")); // Big drop at end
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-11", 50, 50)};
    set<string> features = {"EWMA volatility over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_GT(result.features[0][FeatureCalculator::EWMA_VOL_10D], 0.5);
}

// Mixed volatility: stable, then volatile
TEST(FeatureExtractorTest, EwmaVol10D_MixedVolatility) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    for (int i = 5; i < 10; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + (i % 2) * 20, "2021-01-" + day));
    }
    rows.push_back(makeRow(120, "2021-01-11"));
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-11", 120, 120)};
    set<string> features = {"EWMA volatility over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_GT(result.features[0][FeatureCalculator::EWMA_VOL_10D], 0.1);
}

// ------------------ DIST_TO_SMA_5D ------------------
TEST(FeatureExtractorTest, DistToSMA5D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + i, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 104, 104)};
    set<string> features = {"Distance between current close price and 5-day SMA"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::DIST_TO_SMA_5D], 2.0);
}

TEST(FeatureExtractorTest, DistToSMA5D_Zero) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 100, 100)};
    set<string> features = {"Distance between current close price and 5-day SMA"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::DIST_TO_SMA_5D], 0.0);
}

TEST(FeatureExtractorTest, DistToSMA5D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 3; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(10 * (i+1), "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-03", 30, 30)};
    set<string> features = {"Distance between current close price and 5-day SMA"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::DIST_TO_SMA_5D], 10.0);
}

// ------------------ ROC_5D ------------------
TEST(FeatureExtractorTest, Roc5D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 6; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + i * 2, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-06", 108, 108)};
    set<string> features = {"Rate of Change (ROC) over 5 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // ROC = (110 - 100) / 100 = 0.1
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::ROC_5D], 0.1);
}

TEST(FeatureExtractorTest, Roc5D_Zero) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 6; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-06", 100, 100)};
    set<string> features = {"Rate of Change (ROC) over 5 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::ROC_5D], 0.0);
}

TEST(FeatureExtractorTest, Roc5D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 3; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(10 * (i+1), "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-03", 30, 30)};
    set<string> features = {"Rate of Change (ROC) over 5 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_DOUBLE_EQ(result.features[0][FeatureCalculator::ROC_5D], 2.0);
}

// ------------------ RSI_14D ------------------
TEST(FeatureExtractorTest, RSI14D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 14; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + i, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-14", 113, 113)};
    set<string> features = {"Relative Strength Index (RSI) over 14 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::RSI_14D], 100.0, 0.01);
}

TEST(FeatureExtractorTest, RSI14D_Constant) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 14; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-01", 100, 100)};
    set<string> features = {"Relative Strength Index (RSI) over 14 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::RSI_14D], 0.0, 0.01);
}

TEST(FeatureExtractorTest, RSI14D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 7; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + i, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-07", 106, 106)};
    set<string> features = {"Relative Strength Index (RSI) over 14 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::RSI_14D], 100.0, 0.01);
}

TEST(FeatureExtractorTest, RSI14D_AlternatingUpDown) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 14; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + (i % 2), "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-14", 101, 101)};
    set<string> features = {"Relative Strength Index (RSI) over 14 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::RSI_14D], 50.0, 1.0);
}

TEST(FeatureExtractorTest, RSI14D_SingleDrop) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 13; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    rows.push_back(makeRow(90, "2021-01-14"));
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-14", 90, 90)};
    set<string> features = {"Relative Strength Index (RSI) over 14 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::RSI_14D], 0.0, 0.01);
}

TEST(FeatureExtractorTest, RSI14D_SingleRise) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 13; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    rows.push_back(makeRow(110, "2021-01-14"));
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-14", 110, 110)};
    set<string> features = {"Relative Strength Index (RSI) over 14 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::RSI_14D], 100.0, 0.01);
}

// ------------------ PRICE_RANGE_5D ------------------
TEST(FeatureExtractorTest, PriceRange5D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + i * 2, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 108, 108)};
    set<string> features = {"5-day high minus 5-day low (price range)"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::PRICE_RANGE_5D], 8.0, 0.01);
}

TEST(FeatureExtractorTest, PriceRange5D_Constant) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-01", 100, 100)};
    set<string> features = {"5-day high minus 5-day low (price range)"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::PRICE_RANGE_5D], 0.0, 0.01);
}

TEST(FeatureExtractorTest, PriceRange5D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 3; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(10 * (i+1), "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-03", 30, 30)};
    set<string> features = {"5-day high minus 5-day low (price range)"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::PRICE_RANGE_5D], 20.0, 0.01);
}

// ------------------ CLOSE_OVER_HIGH_5D ------------------
TEST(FeatureExtractorTest, CloseOverHigh5D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + i, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 104, 104)};
    set<string> features = {"Current close price relative to 5-day high"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_OVER_HIGH_5D], 1.0, 0.01);
}

TEST(FeatureExtractorTest, CloseOverHigh5D_Constant) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-01", 100, 100)};
    set<string> features = {"Current close price relative to 5-day high"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_OVER_HIGH_5D], 1.0, 0.01);
}

TEST(FeatureExtractorTest, CloseOverHigh5D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 3; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(10 * (i+1), "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-03", 30, 30)};
    set<string> features = {"Current close price relative to 5-day high"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_OVER_HIGH_5D], 1.0, 0.01);
}

// Case: High is at the start of the window
TEST(FeatureExtractorTest, CloseOverHigh5D_HighAtStart) {
    vector<PreprocessedRow> rows = {makeRow(200, "2021-01-01"), makeRow(100, "2021-01-02"), makeRow(100, "2021-01-03"), makeRow(100, "2021-01-04"), makeRow(100, "2021-01-05")};
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 100, 100)};
    set<string> features = {"Current close price relative to 5-day high"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // Last price is 100, high is 200, so ratio is 0.5
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_OVER_HIGH_5D], 0.5, 0.01);
}

// Case: High is in the middle of the window
TEST(FeatureExtractorTest, CloseOverHigh5D_HighInMiddle) {
    vector<PreprocessedRow> rows = {makeRow(100, "2021-01-01"), makeRow(150, "2021-01-02"), makeRow(200, "2021-01-03"), makeRow(150, "2021-01-04"), makeRow(100, "2021-01-05")};
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 100, 100)};
    set<string> features = {"Current close price relative to 5-day high"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // Last price is 100, high is 200, so ratio is 0.5
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_OVER_HIGH_5D], 0.5, 0.01);
}

// Case: All prices are different, high is at the end
TEST(FeatureExtractorTest, CloseOverHigh5D_HighAtEnd) {
    vector<PreprocessedRow> rows = {makeRow(100, "2021-01-01"), makeRow(120, "2021-01-02"), makeRow(140, "2021-01-03"), makeRow(160, "2021-01-04"), makeRow(180, "2021-01-05")};
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 180, 180)};
    set<string> features = {"Current close price relative to 5-day high"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // Last price is 180, high is 180, so ratio is 1.0
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_OVER_HIGH_5D], 1.0, 0.01);
}

// Case: Only one row (edge case)
TEST(FeatureExtractorTest, CloseOverHigh5D_OneRow) {
    vector<PreprocessedRow> rows = {makeRow(123, "2021-01-01")};
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-01", 123, 123)};
    set<string> features = {"Current close price relative to 5-day high"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // Last price is 123, high is 123, so ratio is 1.0
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_OVER_HIGH_5D], 1.0, 0.01);
}

// Case: High is lower than close (should not happen, but test for robustness)
TEST(FeatureExtractorTest, CloseOverHigh5D_CloseAboveHigh) {
    vector<PreprocessedRow> rows = {makeRow(100, "2021-01-01"), makeRow(100, "2021-01-02"), makeRow(100, "2021-01-03"), makeRow(100, "2021-01-04"), makeRow(200, "2021-01-05")};
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 200, 200)};
    set<string> features = {"Current close price relative to 5-day high"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // Last price is 200, high is 200, so ratio is 1.0
    EXPECT_NEAR(result.features[0][FeatureCalculator::CLOSE_OVER_HIGH_5D], 1.0, 0.01);
}

// ------------------ SLOPE_LR_10D ------------------
TEST(FeatureExtractorTest, SlopeLR10D_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 10; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100 + i, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-10", 109, 109)};
    set<string> features = {"Slope of linear regression of close prices over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::SLOPE_LR_10D], 1.0, 0.01);
}

TEST(FeatureExtractorTest, SlopeLR10D_Constant) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 10; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-01", 100, 100)};
    set<string> features = {"Slope of linear regression of close prices over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::SLOPE_LR_10D], 0.0, 0.01);
}

TEST(FeatureExtractorTest, SlopeLR10D_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 5; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(10 * (i+1), "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-05", 50, 50)};
    set<string> features = {"Slope of linear regression of close prices over 10 days"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    EXPECT_NEAR(result.features[0][FeatureCalculator::SLOPE_LR_10D], 10.0, 0.01);
}

// ------------------ DAY_OF_WEEK ------------------
TEST(FeatureExtractorTest, DayOfWeek_Basic) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 7; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-07", 100, 100)};
    set<string> features = {"Day of the week"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // 2021-01-07 is a Thursday (day 4), but this depends on implementation
    // For this test, let's expect 4
    EXPECT_EQ(result.features[0][FeatureCalculator::DAY_OF_WEEK], 4);
}

TEST(FeatureExtractorTest, DayOfWeek_Constant) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 7; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-01", 100, 100)};
    set<string> features = {"Day of the week"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // 2021-01-01 is a Friday (day 5)
    EXPECT_EQ(result.features[0][FeatureCalculator::DAY_OF_WEEK], 5);
}

TEST(FeatureExtractorTest, DayOfWeek_ShortWindow) {
    vector<PreprocessedRow> rows;
    for (int i = 0; i < 3; ++i) {
        string day = (i+1 < 10 ? "0" : "") + to_string(i+1);
        rows.push_back(makeRow(100, "2021-01-" + day));
    }
    vector<LabeledEvent> events = {makeEvent(1, "2021-01-03", 100, 100)};
    set<string> features = {"Day of the week"};
    auto result = FeatureExtractor::extractFeaturesForClassification(features, rows, events);
    // 2021-01-03 is a Sunday (day 0)
    EXPECT_EQ(result.features[0][FeatureCalculator::DAY_OF_WEEK], 0);
}
