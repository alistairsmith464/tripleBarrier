@echo off
setlocal enabledelayedexpansion

echo === Triple Barrier Application - Simple Build and Run ===

rem Set project paths
set PROJECT_ROOT=%~dp0
set BUILD_DIR=%PROJECT_ROOT%build
set QT_PATH=

rem Try to find Qt installation
echo Searching for Qt installation...

rem Check specific known path first
if exist "C:\Qt\6.9.1\mingw_64\bin\qtpaths.exe" (
    set QT_PATH=C:\Qt\6.9.1\mingw_64
    goto :found_qt
)

for /d %%d in ("C:\Qt\*") do (
    for /d %%v in ("%%d\msvc*") do (
        if exist "%%v\bin\qtpaths.exe" (
            set QT_PATH=%%v
            goto :found_qt
        )
    )
    for /d %%v in ("%%d\mingw*") do (
        if exist "%%v\bin\qtpaths.exe" (
            set QT_PATH=%%v
            goto :found_qt
        )
    )
)

for /d %%d in ("%USERPROFILE%\Qt\*") do (
    for /d %%v in ("%%d\msvc*") do (
        if exist "%%v\bin\qtpaths.exe" (
            set QT_PATH=%%v
            goto :found_qt
        )
    )
    for /d %%v in ("%%d\mingw*") do (
        if exist "%%v\bin\qtpaths.exe" (
            set QT_PATH=%%v
            goto :found_qt
        )
    )
)

:found_qt
if "%QT_PATH%"=="" (
    echo Error: Qt6 installation not found!
    echo Please install Qt6 or set QT_PATH manually in this script.
    pause
    exit /b 1
)

echo Found Qt6 at: %QT_PATH%

rem Add Qt, MinGW compiler tools, and CMake to PATH
set PATH=%QT_PATH%\bin;C:\Qt\Tools\mingw1310_64\bin;C:\Qt\Tools\CMake_64\bin;%PATH%

rem Check if CMake is available
cmake --version >nul 2>&1
if errorlevel 1 (
    echo Error: CMake not found! Please install CMake and add it to PATH.
    pause
    exit /b 1
)

rem Create build directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

rem Configure project
echo Configuring project...
cd /d "%BUILD_DIR%"
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="%QT_PATH%" "%PROJECT_ROOT%"
if errorlevel 1 (
    cd /d "%PROJECT_ROOT%"
    echo Error: CMake configuration failed!
    pause
    exit /b 1
)
cd /d "%PROJECT_ROOT%"

rem Build project
echo Building project...
cmake --build "%BUILD_DIR%"
if errorlevel 1 (
    echo Error: Build failed!
    pause
    exit /b 1
)

rem Copy xgboost.dll to frontend build output
call "%PROJECT_ROOT%copy_xgboost_dll.bat"

rem Run application
echo Starting application...
"%BUILD_DIR%\frontend\TripleBarrierApp.exe"

echo === Build and run completed ===
pause
