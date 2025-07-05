cmake_minimum_required(VERSION 3.16)

find_package(Qt6 COMPONENTS Charts REQUIRED)

# Frontend executable
add_executable(TripleBarrierApp
    main.cpp
    MainWindow.cpp
    MainWindow.h
    MainWindow.ui
    FileHandler.cpp
    FileHandler.h
    BarrierConfigDialog.cpp
    BarrierConfigDialog.h
    FeatureSelectionDialog.cpp
    FeatureSelectionDialog.h
)

target_link_libraries(TripleBarrierApp 
    Qt6::Core 
    Qt6::Widgets 
    Qt6::Charts
    backend
)
