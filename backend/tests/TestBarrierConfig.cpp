#include <gtest/gtest.h>
#include "../data/BarrierConfig.h"

TEST(BarrierConfigTest, Validation) {
    BarrierConfig cfg{2.0, 2.0, 20};
    EXPECT_NO_THROW(cfg.validate());

    cfg.profit_multiple = 2.0;
    cfg.stop_multiple = 0.5;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg.profit_multiple = 1.0;
    cfg.stop_multiple = 0.5;
    EXPECT_NO_THROW(cfg.validate());

    cfg.profit_multiple = 0.0;
    cfg.stop_multiple = 0.5;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg.profit_multiple = 1.0;
    cfg.stop_multiple = 0.0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg.profit_multiple = 1.0;
    cfg.stop_multiple = 1.0;
    cfg.vertical_window = 0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(BarrierConfigTest, EdgeCases) {
    BarrierConfig cfg{1e-9, 1e-9, 1};
    EXPECT_NO_THROW(cfg.validate());

    cfg.profit_multiple = -1e-9;
    cfg.stop_multiple = 1e-9;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg.profit_multiple = 1.0;
    cfg.stop_multiple = -1.0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg.profit_multiple = 1.0;
    cfg.stop_multiple = 1.0;
    EXPECT_NO_THROW(cfg.validate());

    cfg.profit_multiple = 1.0;
    cfg.stop_multiple = 2.1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg.profit_multiple = 1.0;
    cfg.stop_multiple = 1.0;
    cfg.vertical_window = -1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(BarrierConfigTest, LargeValues) {
    BarrierConfig cfg{1e6, 1e6, 1000000};
    EXPECT_NO_THROW(cfg.validate());

    cfg.profit_multiple = 1e6;
    cfg.stop_multiple = 1.0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}
