cmake_minimum_required(VERSION 3.16)

# Backend library
add_library(backend
    data/CSVDataSource.cpp
    data/CSVDataSource.h
    data/DataRow.h
    data/DataSource.h
)

target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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
