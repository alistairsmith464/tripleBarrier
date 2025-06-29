@echo off
echo Building Triple Barrier App...

rem Set paths for your specific setup
set QT_PATH=C:\Qt\6.9.1\mingw_64
set CMAKE_PATH=C:\Qt\Tools\CMake_64\bin
set MINGW_PATH=C:\Qt\Tools\mingw1310_64\bin

rem Add tools to PATH
set PATH=%QT_PATH%\bin;%CMAKE_PATH%;%MINGW_PATH%;%PATH%

rem Build
if not exist build mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH="%QT_PATH%" ..
cmake --build .
cd ..

echo Build complete!
pause
