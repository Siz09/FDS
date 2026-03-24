# Start Face-Service for Production
# Uses Gunicorn with Uvicorn workers to bypass the Python GIL
# and allow true parallel face processing.

# Ensure we're in the right directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir\..

# Check available RAM before starting (needs ~2GB for 4 workers)
$freeRam = Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty FreePhysicalMemory
$freeRamMB = [math]::Round($freeRam / 1024)

if ($freeRamMB -lt 2048) {
    Write-Warning "WARNING: Only ${freeRamMB}MB RAM available. 4 Gunicorn workers may consume ~2GB."
    Write-Warning "Press Ctrl+C to abort, or wait 3 seconds to continue..."
    Start-Sleep -Seconds 3
}

Write-Host "Starting face-service with 4 Gunicorn workers..." -ForegroundColor Green
.venv\Scripts\activate
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
