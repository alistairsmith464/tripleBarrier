cmake_minimum_required(VERSION 3.16)

# Backend library
add_library(backend
    data/CSVDataSource.cpp
    data/CSVDataSource.h
    data/DataRow.h
    data/DataSource.h
    data/DataPreprocessor.cpp
    data/DataPreprocessor.h
    data/LabeledEvent.h
    data/CUSUMFilter.cpp
    data/CUSUMFilter.h
    data/FeatureCalculator.cpp
    data/FeatureCalculator.h
    data/HardBarrierLabeler.cpp
    data/HardBarrierLabeler.h
    data/TTBMLabeler.cpp
    data/TTBMLabeler.h
    data/IBarrierLabeler.h
    data/BarrierConfig.h
    data/FeatureExtractor.cpp
    data/FeatureExtractor.h
    data/PreprocessedRow.h
    data/EventSelector.cpp
    data/EventSelector.h
    data/OverlapPurger.cpp
    data/OverlapPurger.h
    data/SampleIndependenceValidator.cpp
    data/SampleIndependenceValidator.h
    data/VolatilityCalculator.h
    data/DataCleaningUtils.h
    data/Constants.h
    ml/MLPipeline.cpp
    ml/MLPipeline.h
    ml/MLSplits.h
    ml/XGBoostModel.cpp
    ml/XGBoostModel.h
    ml/PortfolioSimulator.cpp
    ml/PortfolioSimulator.h
    ml/DataUtils.cpp
    ml/DataUtils.h
    ml/ModelUtils.cpp
    ml/ModelUtils.h
    ml/MetricsCalculator.cpp
    ml/MetricsCalculator.h
    ml/BarrierMLStrategy.cpp
    ml/BarrierMLStrategy.h
    utils/Exceptions.h
    utils/ErrorHandling.h
)

target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link XGBoost using modern CMake target
target_link_libraries(backend PUBLIC xgboost)

# Only link Qt6::Core if building frontend (and Qt6 is available)
if(BUILD_FRONTEND)
    # No Qt6 linkage needed for backend
endif()

# GoogleTest
add_subdirectory(${CMAKE_SOURCE_DIR}/external/googletest ${CMAKE_BINARY_DIR}/googletest)

# Test executable
add_executable(TestCSVDataSource tests/TestCSVDataSource.cpp)
target_link_libraries(TestCSVDataSource backend gtest gtest_main)
target_include_directories(TestCSVDataSource PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_test(NAME CSVDataSourceTest COMMAND TestCSVDataSource WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Test executables for all backend files
add_executable(TestBarrierConfig tests/TestBarrierConfig.cpp)
target_link_libraries(TestBarrierConfig backend gtest gtest_main)
add_test(NAME BarrierConfigTest COMMAND TestBarrierConfig)

add_executable(TestPreprocessedRow tests/TestPreprocessedRow.cpp)
target_link_libraries(TestPreprocessedRow backend gtest gtest_main)
add_test(NAME PreprocessedRowTest COMMAND TestPreprocessedRow)

add_executable(TestVolatilityCalculator tests/TestVolatilityCalculator.cpp)
target_link_libraries(TestVolatilityCalculator backend gtest gtest_main)
add_test(NAME VolatilityCalculatorTest COMMAND TestVolatilityCalculator)

add_executable(TestEventSelector tests/TestEventSelector.cpp)
target_link_libraries(TestEventSelector backend gtest gtest_main)
add_test(NAME EventSelectorTest COMMAND TestEventSelector)

add_executable(TestDataPreprocessor tests/TestDataPreprocessor.cpp)
target_link_libraries(TestDataPreprocessor backend gtest gtest_main)
add_test(NAME DataPreprocessorTest COMMAND TestDataPreprocessor)

add_executable(TestTTBMLabeler tests/TestTTBMLabeler.cpp)
target_link_libraries(TestTTBMLabeler backend gtest gtest_main)
add_test(NAME TTBMLabelerTest COMMAND TestTTBMLabeler)
