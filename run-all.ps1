
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"

if (!(Test-Path $Backend) -or !(Test-Path $Frontend)) {
    Write-Host "Folder backend/frontend tidak ditemukan!" -ForegroundColor Red
    exit 1
}

if (!(Test-Path (Join-Path $Frontend "node_modules"))) {
    Write-Host "Menginstall dependency frontend..."
    Push-Location $Frontend
    npm install
    Pop-Location
}

Write-Host "Starting GERDGuard Backend..."

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$Backend'; py -3.11 -m uvicorn main:app --reload"
)

Start-Sleep -Seconds 2

Write-Host "Starting GERDGuard Frontend..."

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$Frontend'; npm run dev"
)

Write-Host ""
Write-Host "GERDGuard sedang dijalankan!" -ForegroundColor Green
Write-Host "Frontend : http://localhost:3000"
Write-Host "Backend  : http://127.0.0.1:8000"
Write-Host "API Docs : http://127.0.0.1:8000/docs"
