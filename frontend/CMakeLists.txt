cmake_minimum_required(VERSION 3.16)

# Frontend executable
add_executable(TripleBarrierApp
    main.cpp
    MainWindow.cpp
    MainWindow.h
    MainWindow.ui
)

target_link_libraries(TripleBarrierApp 
    Qt6::Core 
    Qt6::Widgets 
    backend
)
