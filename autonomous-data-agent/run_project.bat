@echo off
setlocal enabledelayedexpansion

REM Change to the directory containing this script so paths are relative and not hardcoded
pushd "%~dp0"

cls
echo.
echo ============================================================================
echo  AUTONOMOUS DATA CLEANING AGENT - COMPLETE STARTUP
echo ============================================================================
echo.

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo OK - Virtual environment created
    echo.
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo OK - Virtual environment activated
echo.

python -c "import fastapi, pandas, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies - first time only
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo WARNING: Some packages may have failed to install
    )
    echo OK - Dependencies installed
) else (
    echo OK - Dependencies already installed
)
echo.

echo Generating sample datasets...
python generate_samples.py >nul 2>&1
echo OK - Sample data ready

REM Script will keep current directory as the script directory while running
echo.

REM Start web server directly
cls
echo.
echo ============================================================================
echo  STARTING WEB SERVER
echo ============================================================================
echo.
echo  Frontend: http://localhost:8000
echo  API: http://localhost:8000/api/
echo  API Docs: http://localhost:8000/docs
echo.
echo  Press Ctrl+C to stop the server
echo.
echo ============================================================================
echo.

python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 120 --timeout-graceful-shutdown 120

if errorlevel 1 (
    echo.
    echo ERROR: Server failed to start
    pause
)
