# Build the QMLKit desktop bundle (exe) - repomono GAIT methodology.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/build_desktop.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> [1/3] Building frontend static export" -ForegroundColor Cyan
if (Test-Path "frontend") {
    pnpm --filter frontend build
    if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
} else {
    Write-Host "    no frontend dir - skipping"
}

Write-Host "==> [2/3] Ensuring python deps" -ForegroundColor Cyan
python -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { python -m pip install pyinstaller }
python -c "import qmlkit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    installing qmlkit (editable)"
    python -m pip install -e ".[dev]"
}

Write-Host "==> [3/3] PyInstaller bundle" -ForegroundColor Cyan
python -m pyinstaller packaging/qmlkit_desktop.spec --noconfirm --distpath dist --workpath build/pyinstaller
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

$out = "dist/QMLKitConsole/QMLKitConsole.exe"
Write-Host ""
Write-Host "[OK] Bundle ready: $out" -ForegroundColor Green

# Optional installer when Inno Setup is present.
$isccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc -and (Test-Path "packaging/installer/qmlkit.iss")) {
    Write-Host "==> Building installer with Inno Setup"
    & $iscc "packaging/installer/qmlkit.iss"
    Write-Host "[OK] Installer under dist/installer/" -ForegroundColor Green
}
