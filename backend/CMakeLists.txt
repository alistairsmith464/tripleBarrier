cmake_minimum_required(VERSION 3.16)

# Backend library
add_library(backend
    FileHandler.cpp
    FileHandler.h
)

target_link_libraries(backend Qt6::Core)
target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
