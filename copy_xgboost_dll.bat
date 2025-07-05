@echo off
REM Copy xgboost.dll to the frontend build output directory after build
set SRC_DLL=%~dp0xgboost\lib\xgboost.dll
set DEST_DLL=%~dp0build\frontend\xgboost.dll
if exist "%SRC_DLL%" copy /Y "%SRC_DLL%" "%DEST_DLL%"
