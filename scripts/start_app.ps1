# Launch the packaged QMLKit console; falls back to running from source.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/start_app.ps1 [-Browser]

param(
    [switch]$Browser
)

$root = Split-Path -Parent $PSScriptRoot

$exe = Join-Path $root "dist/QMLKitConsole/QMLKitConsole.exe"
if (Test-Path $exe) {
    Write-Host "Starting packaged app: $exe"
    Start-Process -FilePath $exe -WorkingDirectory (Split-Path -Parent $exe)
    exit 0
}

Write-Host "Packaged exe not found - falling back to source mode." -ForegroundColor Yellow
Set-Location $root
if (-not (Test-Path (Join-Path $root "frontend/out"))) {
    pnpm --filter frontend build
}
$arg = if ($Browser) { "--browser" } else { "" }
python qmlkit_desktop.py --port 8000 $arg
