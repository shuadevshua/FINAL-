@echo off
REM Startup script to run Flask app with venv
echo ========================================
echo Starting Neural Toxicity Analyzer
echo ========================================
echo.

REM Get the directory where this script is located
cd /d "%~dp0"

REM Activate venv
echo Activating virtual environment (venv)...
call venv\Scripts\activate.bat

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] Flask not found in venv. Installing...
    pip install flask flask-cors
    echo.
)

REM Start the Flask server
echo.
echo Starting Flask server...
echo Frontend will be available at: http://localhost:5000/
echo.
python app.py

pause

