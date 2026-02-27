@echo off
setlocal enabledelayedexpansion

cd /d "C:\Users\Praneeth P\OneDrive\Desktop\data\autonomous-data-agent"

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
echo.

:menu
cls
echo.
echo ============================================================================
echo  AUTONOMOUS DATA CLEANING AGENT - MAIN MENU
echo ============================================================================
echo.
echo  1. Start Web Server (http://localhost:8000)
echo  2. Prepare Dataset (Run 9-Step Data Preparation)
echo  3. Generate Sample Datasets
echo  4. Exit
echo.
echo ============================================================================
set /p choice="Select option (1-4): "

if "%choice%"=="1" goto start_server
if "%choice%"=="2" goto prepare_data
if "%choice%"=="3" goto generate_samples
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:start_server
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
goto menu

:prepare_data
cls
echo.
echo ============================================================================
echo  DATA PREPARATION - 9 STEP PIPELINE
echo ============================================================================
echo.
echo Enter CSV file path (or drag and drop file here):
set /p input_file="Input file: "

if not exist "%input_file%" (
    echo.
    echo ERROR: File not found: %input_file%
    pause
    goto menu
)

echo.
echo Processing: %input_file%
echo.

python prepare_data.py "%input_file%"

if errorlevel 1 (
    echo.
    echo ERROR: Data preparation failed
    pause
) else (
    echo.
    pause
)

goto menu

:generate_samples
echo.
echo Generating sample datasets...
python generate_samples.py
echo.
pause
goto menu

:exit
echo.
echo Goodbye!
echo.
