#pragma once

namespace Constants {
    namespace CSV {
        constexpr int EXTENSION_LENGTH = 4;
        constexpr const char* EXTENSION = ".csv";
    }
    
    namespace Validation {
        constexpr double EPSILON = 1e-6;
        constexpr double NUMERICAL_EPSILON = 1e-12;
        constexpr double MAX_HYPERBOLIC_BETA_TIME = 1e6;
    }
    
    namespace Defaults {
        constexpr double STARTING_CAPITAL = 100000.0;
        constexpr int DEFAULT_VOLATILITY_WINDOW = 20;
        constexpr double DEFAULT_CUSUM_THRESHOLD = 5.0;
        constexpr int DEFAULT_VERTICAL_BARRIER = 50;
        constexpr double DEFAULT_PROFIT_MULTIPLE = 2.0;
        constexpr double DEFAULT_STOP_MULTIPLE = 1.0;
    }
    
    namespace Portfolio {
        constexpr double TRADING_DAYS_PER_YEAR = 252.0;
        constexpr double MAX_POSITION_PCT = 0.05;
        constexpr double POSITION_THRESHOLD = 0.05;
    }
}
