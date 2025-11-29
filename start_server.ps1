# PowerShell startup script to run Flask app with venv
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Neural Toxicity Analyzer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Activate venv
Write-Host "Activating virtual environment (venv)..." -ForegroundColor Yellow
& "$scriptDir\venv\Scripts\Activate.ps1"

# Check if Flask is installed
try {
    python -c "import flask" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[WARNING] Flask not found in venv. Installing..." -ForegroundColor Yellow
        pip install flask flask-cors
        Write-Host ""
    }
} catch {
    Write-Host "[WARNING] Flask not found in venv. Installing..." -ForegroundColor Yellow
    pip install flask flask-cors
    Write-Host ""
}

# Start the Flask server
Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Green
Write-Host "Frontend will be available at: http://localhost:5000/" -ForegroundColor Green
Write-Host ""
python app.py

