@echo off
setlocal

title Depth Anything V2 Video Converter
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "FFMPEG_PATH=E:\AI\Tool\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
set "PYTHONUTF8=1"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Project Python was not found:
    echo %PYTHON_EXE%
    echo.
    echo Check that the .venv folder still exists in this project.
    pause
    exit /b 1
)

if not exist "%FFMPEG_PATH%" (
    echo [ERROR] FFmpeg was not found:
    echo %FFMPEG_PATH%
    echo.
    echo Check that FFmpeg is still installed at the configured path.
    pause
    exit /b 1
)

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
