cmake_minimum_required(VERSION 3.16)
project(TripleBarrierApp VERSION 1.0.0 LANGUAGES CXX)

option(BUILD_FRONTEND "Build the Qt6 frontend" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt6 components
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# Enable automatic MOC, UIC, and RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Add subdirectories
add_subdirectory(backend)

if(BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()
