@echo off
setlocal

title Depth Anything V2 Video Converter
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "PYTHONUTF8=1"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Project Python was not found:
    echo %PYTHON_EXE%
    echo.
    echo Check that the .venv folder still exists in this project.
    pause
    exit /b 1
)

if defined FFMPEG_PATH (
    if exist "%FFMPEG_PATH%" goto ffmpeg_ok
    echo [ERROR] FFMPEG_PATH points to a missing file:
    echo %FFMPEG_PATH%
    echo.
    pause
    exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [ERROR] FFmpeg was not found in PATH.
    echo Install FFmpeg and add its bin folder to PATH,
    echo or set FFMPEG_PATH to ffmpeg.exe before starting.
    echo.
    pause
    exit /b 1
)

:ffmpeg_ok

echo Starting Depth Anything V2 Video Converter...
echo The local web page will open in your browser automatically.
echo To stop the program, press Ctrl+C here or close this window.
echo.

"%PYTHON_EXE%" app.py

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] The program exited with code %EXIT_CODE%.
    echo Keep this window open if you need to inspect the error above.
    pause
)

exit /b %EXIT_CODE%
