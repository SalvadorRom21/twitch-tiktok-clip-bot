# Twitch-to-TikTok Clip Bot — Windows setup
# Run in PowerShell:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
#   irm https://raw.githubusercontent.com/SalvadorRom21/twitch-tiktok-clip-bot/main/setup-windows.ps1 | iex

$ErrorActionPreference = "Stop"

$InstallDir = Join-Path $env:USERPROFILE "Documents\twitch-tiktok-clip-bot"
$RepoUrl = "https://github.com/SalvadorRom21/twitch-tiktok-clip-bot.git"

Write-Host ""
Write-Host "=== Twitch TikTok Clip Bot - Windows Setup ===" -ForegroundColor Cyan
Write-Host ""

# --- Python ---
$python = $null
foreach ($cmd in @("py", "python", "python3")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $python = $cmd
        break
    }
}
if (-not $python) {
    Write-Host "Python not found." -ForegroundColor Red
    Write-Host "Install from https://www.python.org/downloads/ and check 'Add python.exe to PATH'" -ForegroundColor Yellow
    Write-Host "Then disable Store aliases: Settings > Apps > Advanced app settings > App execution aliases" -ForegroundColor Yellow
    exit 1
}
Write-Host "Using Python: $python" -ForegroundColor Green
& $python --version

# --- Git ---
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git not found. Install from https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# --- Clone or update ---
if (Test-Path $InstallDir) {
    Write-Host "Project folder exists. Pulling latest..." -ForegroundColor Yellow
    Set-Location $InstallDir
    git pull origin main
} else {
    Write-Host "Cloning to: $InstallDir" -ForegroundColor Green
    $parent = Split-Path $InstallDir -Parent
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    git clone $RepoUrl $InstallDir
    Set-Location $InstallDir
}

# --- Virtual environment ---
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    & $python -m venv .venv
}
Write-Host "Activating virtual environment..." -ForegroundColor Green
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies (may take a few minutes)..." -ForegroundColor Green
python -m pip install --upgrade pip
pip install -r requirements.txt

# --- Config file ---
if (-not (Test-Path "config.local.yaml")) {
    Copy-Item "config.local.yaml.example" "config.local.yaml"
    Write-Host "Created config.local.yaml - add your Twitch credentials there." -ForegroundColor Yellow
}

# --- FFmpeg check ---
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host ""
    Write-Host "FFmpeg not found. Install with:" -ForegroundColor Yellow
    Write-Host "  winget install Gyan.FFmpeg" -ForegroundColor White
    Write-Host "Then restart PowerShell." -ForegroundColor Yellow
} else {
    Write-Host "FFmpeg: OK" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project path:" -ForegroundColor White
Write-Host "  $InstallDir" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Edit credentials:  notepad config.local.yaml" -ForegroundColor Gray
Write-Host "  2. Start web UI:      python main.py --web" -ForegroundColor Gray
Write-Host "  3. Or fetch clips:    python main.py --fetch-clips" -ForegroundColor Gray
Write-Host ""
Write-Host "Open in browser: http://127.0.0.1:8080" -ForegroundColor Cyan
Write-Host ""

# Leave shell in project dir with venv active
Set-Location $InstallDir
