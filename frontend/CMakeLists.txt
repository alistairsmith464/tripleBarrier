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
    MLHyperparamsDialog.cpp
    MLHyperparamsDialog.h
    utils/CSVExportUtils.cpp
    utils/DialogUtils.cpp
    utils/FileDialogUtils.cpp
    utils/UserInputUtils.cpp
    utils/LabelingUtils.cpp
    utils/FeaturePreviewUtils.cpp
    plot/LabeledEventPlotter.cpp
    feature/FeaturePreviewDialog.cpp
    ui/MainWindowUI.cpp
    ../backend/data/HardBarrierLabeler.cpp
)

target_link_libraries(TripleBarrierApp 
    Qt6::Core 
    Qt6::Widgets 
    Qt6::Charts
    backend
    "${CMAKE_SOURCE_DIR}/xgboost/lib/libxgboost.dll.a"
)

target_include_directories(TripleBarrierApp PRIVATE ${CMAKE_SOURCE_DIR})
