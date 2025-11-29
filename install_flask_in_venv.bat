@echo off
REM Script to install Flask in venv (our single virtual environment)
echo ========================================
echo Installing Flask in venv
echo ========================================
echo.
echo This will install Flask and flask-cors in venv.
echo venv is our single virtual environment for this project.
echo.

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv" (
    echo [ERROR] venv not found!
    echo Please create it first with: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install flask flask-cors

echo.
echo [SUCCESS] Flask has been installed in venv.
echo You can now run start_server.bat to start the application.
echo.
pause

