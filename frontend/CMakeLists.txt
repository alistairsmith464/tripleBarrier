cmake_minimum_required(VERSION 3.16)

find_package(Qt6 COMPONENTS Charts REQUIRED)

# Frontend executable
add_executable(TripleBarrierApp
    main.cpp
    MainWindow.cpp
    MainWindow.h
    MainWindow.ui
    BarrierConfigDialog.cpp
    BarrierConfigDialog.h
    FeatureSelectionDialog.cpp
    FeatureSelectionDialog.h
    MLHyperparamsDialog.cpp
    MLHyperparamsDialog.h
    utils/DialogUtils.cpp
    utils/FileDialogUtils.cpp
    utils/UserInputUtils.cpp
    utils/FeaturePreviewUtils.cpp
    utils/DateParsingUtils.cpp
    utils/ErrorHandler.cpp
    utils/InputValidator.cpp
    utils/AsyncTaskManager.cpp
    utils/ValidationFramework.cpp
    utils/ErrorHandlingStrategy.cpp
    services/DataService.cpp
    services/MLService.cpp
    state/ApplicationState.cpp
    config/ApplicationConfig.cpp
    plot/LabeledEventPlotter.cpp
    plot/PlotStrategy.cpp
    feature/FeaturePreviewDialog.cpp
    ui/MainWindowUI.cpp
    ui/UIStrings.cpp
    ../backend/data/HardBarrierLabeler.cpp
)

target_link_libraries(TripleBarrierApp 
    Qt6::Core 
    Qt6::Widgets 
    Qt6::Charts
    backend
)

target_include_directories(TripleBarrierApp PRIVATE ${CMAKE_SOURCE_DIR})
