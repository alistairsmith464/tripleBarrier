cmake_minimum_required(VERSION 3.16)

# Backend library
add_library(backend
    FileHandler.cpp
    FileHandler.h
    data/CSVDataSource.cpp
    data/CSVDataSource.h
    data/DataRow.h
    data/DataSource.h
)

target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Only link Qt6::Core if building frontend (and Qt6 is available)
if(BUILD_FRONTEND)
    target_link_libraries(backend Qt6::Core)
endif()

# GoogleTest
add_subdirectory(${CMAKE_SOURCE_DIR}/external/googletest ${CMAKE_BINARY_DIR}/googletest)

# Test executable
add_executable(TestCSVDataSource tests/TestCSVDataSource.cpp)
target_link_libraries(TestCSVDataSource backend gtest gtest_main)
target_include_directories(TestCSVDataSource PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_test(NAME CSVDataSourceTest COMMAND TestCSVDataSource)
