#include <gtest/gtest.h>
#include "../data/BarrierConfig.h"

TEST(BarrierConfigTest, Validation) {
    BarrierConfig cfg{2.0, 0.5, 20};
    EXPECT_NO_THROW(cfg.validate());
    cfg.stop_multiple = 1.1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.stop_multiple = -0.1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.stop_multiple = 0.5;
    cfg.profit_multiple = 0.0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.profit_multiple = 2.0;
    cfg.vertical_window = 0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(BarrierConfigTest, EdgeCases) {
    BarrierConfig cfg{1e-9, 0.0, 1};
    EXPECT_NO_THROW(cfg.validate());
    cfg.profit_multiple = -1e-9;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.profit_multiple = 1.0;
    cfg.stop_multiple = 1.0;
    EXPECT_NO_THROW(cfg.validate());
    cfg.stop_multiple = 1.00001;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.stop_multiple = 0.0;
    cfg.vertical_window = 1;
    EXPECT_NO_THROW(cfg.validate());
    cfg.vertical_window = -1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(BarrierConfigTest, LargeValues) {
    BarrierConfig cfg{1e6, 1.0, 1000000};
    EXPECT_NO_THROW(cfg.validate());
    cfg.stop_multiple = 1.1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}
