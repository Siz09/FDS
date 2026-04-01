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

$workers = if ($env:FACE_SERVICE_WORKERS) { $env:FACE_SERVICE_WORKERS } else { 4 }
Write-Host "Starting face-service with $workers Gunicorn workers (OPENBLAS_NUM_THREADS=1)..." -ForegroundColor Green
$env:OPENBLAS_NUM_THREADS = "1"
.venv\Scripts\activate
gunicorn app.main:app -w $workers -k uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000
