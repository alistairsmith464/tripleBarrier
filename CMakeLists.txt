cmake_minimum_required(VERSION 3.16)
project(TripleBarrierApp VERSION 1.0.0 LANGUAGES CXX)

option(BUILD_FRONTEND "Build the Qt6 frontend" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)

# Fetch XGBoost
FetchContent_Declare(
    xgboost
    GIT_REPOSITORY https://github.com/dmlc/xgboost.git
    GIT_TAG v2.1.1  # Use a stable version
)

# Configure XGBoost build options
set(BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(XGBOOST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XGBOOST_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Make XGBoost available
FetchContent_MakeAvailable(xgboost)

if(BUILD_FRONTEND)
    # Find Qt6 components
    find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

    # Enable automatic MOC, UIC, and RCC
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
endif()

# Add subdirectories
add_subdirectory(backend)

if(BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()
