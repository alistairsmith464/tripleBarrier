#include <gtest/gtest.h>
#include "../data/TTBMLabeler.h"
#include "../data/PreprocessedRow.h"
#include "../data/BarrierConfig.h"
#include <cmath>

TEST(TTBMLabelerTest, ExponentialDecayBasic) {
    TTBMLabeler labeler(BarrierConfig::Exponential, 1.0, 0.5, 1.0);
    
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    // Quick profit hit at time 1
    data[1].price = 102.1;
    
    std::vector<size_t> events = {0};
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);  // Hard label should be +1
    EXPECT_GT(result[0].ttbm_label, 0.0);  // TTBM label should be positive
    EXPECT_LT(result[0].ttbm_label, 1.0);  // But reduced due to time decay
    EXPECT_EQ(result[0].time_to_barrier_ratio, 0.25);  // 1 period out of 4
    
    // Check exponential decay: f(t) = e^(-λ * t) = e^(-1.0 * 0.25) ≈ 0.779
    EXPECT_NEAR(result[0].decay_factor, std::exp(-1.0 * 0.25), 0.01);
    EXPECT_NEAR(result[0].ttbm_label, result[0].decay_factor, 0.01);
}

TEST(TTBMLabelerTest, LinearDecayBasic) {
    TTBMLabeler labeler(BarrierConfig::Linear, 1.0, 0.5, 1.0);
    
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    // Stop hit at time 2
    data[2].price = 98.9;
    
    std::vector<size_t> events = {0};
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, -1);  // Hard label should be -1
    EXPECT_LT(result[0].ttbm_label, 0.0);  // TTBM label should be negative
    EXPECT_GT(result[0].ttbm_label, -1.0);  // But reduced in magnitude due to time decay
    EXPECT_EQ(result[0].time_to_barrier_ratio, 0.5);  // 2 periods out of 4
    
    // Check linear decay: f(t) = 1 - α * t = 1 - 0.5 * 0.5 = 0.75
    EXPECT_NEAR(result[0].decay_factor, 0.75, 0.01);
    EXPECT_NEAR(result[0].ttbm_label, -0.75, 0.01);  // -1 * 0.75
}

TEST(TTBMLabelerTest, HyperbolicDecayBasic) {
    TTBMLabeler labeler(BarrierConfig::Hyperbolic, 1.0, 0.5, 2.0);
    
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    // Profit hit at time 3
    data[3].price = 102.1;
    
    std::vector<size_t> events = {0};
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);  // Hard label should be +1
    EXPECT_GT(result[0].ttbm_label, 0.0);  // TTBM label should be positive
    EXPECT_EQ(result[0].time_to_barrier_ratio, 0.75);  // 3 periods out of 4
    
    // Check hyperbolic decay: f(t) = 1 / (1 + β * t) = 1 / (1 + 2.0 * 0.75) = 1 / 2.5 = 0.4
    EXPECT_NEAR(result[0].decay_factor, 0.4, 0.01);
    EXPECT_NEAR(result[0].ttbm_label, 0.4, 0.01);
}

TEST(TTBMLabelerTest, VerticalBarrierHit) {
    TTBMLabeler labeler(BarrierConfig::Exponential, 1.0, 0.5, 1.0);
    
    std::vector<PreprocessedRow> data(5);
    for (int i = 0; i < 5; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;  // No barrier hit
        data[i].volatility = 1.0;
    }
    
    std::vector<size_t> events = {0};
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, 0);  // Hard label should be 0 (vertical barrier)
    EXPECT_EQ(result[0].time_to_barrier_ratio, 1.0);  // Full vertical barrier period
    EXPECT_NEAR(result[0].ttbm_label, 0.0, 0.01);  // 0 * decay_factor = 0
}

TEST(TTBMLabelerTest, InstantBarrierHit) {
    TTBMLabeler labeler(BarrierConfig::Exponential, 1.0, 0.5, 1.0);
    
    std::vector<PreprocessedRow> data(3);
    for (int i = 0; i < 3; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    // Immediate profit hit
    data[1].price = 102.1;
    
    std::vector<size_t> events = {0};
    auto result = labeler.label(data, events, 2.0, 1.0, 2);
    
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].label, +1);
    EXPECT_EQ(result[0].time_to_barrier_ratio, 0.5);  // 1 period out of 2
    // For exponential decay with λ=1.0: f(0.5) = e^(-1.0 * 0.5) ≈ 0.606
    EXPECT_NEAR(result[0].decay_factor, std::exp(-0.5), 0.01);
    EXPECT_NEAR(result[0].ttbm_label, std::exp(-0.5), 0.01);
}

TEST(TTBMLabelerTest, MultipleEvents) {
    TTBMLabeler labeler(BarrierConfig::Exponential, 2.0, 0.5, 1.0);
    
    std::vector<PreprocessedRow> data(10);
    for (int i = 0; i < 10; ++i) {
        data[i].timestamp = std::to_string(i);
        data[i].price = 100.0;
        data[i].volatility = 1.0;
    }
    // First event: quick profit
    data[1].price = 102.1;
    // Second event: slower stop
    data[7].price = 98.9;
    
    std::vector<size_t> events = {0, 5};
    auto result = labeler.label(data, events, 2.0, 1.0, 4);
    
    ASSERT_EQ(result.size(), 2);
    
    // First event: profit at time 1
    EXPECT_EQ(result[0].label, +1);
    EXPECT_GT(result[0].ttbm_label, 0.0);
    EXPECT_EQ(result[0].time_to_barrier_ratio, 0.25);
    
    // Second event: stop at time 2 (relative to event start at 5)
    EXPECT_EQ(result[1].label, -1);
    EXPECT_LT(result[1].ttbm_label, 0.0);
    EXPECT_EQ(result[1].time_to_barrier_ratio, 0.5);
    
    // Quick event should have higher magnitude than slower event
    EXPECT_GT(std::abs(result[0].ttbm_label), std::abs(result[1].ttbm_label));
}
