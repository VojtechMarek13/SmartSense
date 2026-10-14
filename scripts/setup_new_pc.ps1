# SmartSense - one-shot setup for a fresh Windows PC.
#
# Installs Git, Python 3.11 and cloudflared (via winget), clones the repo,
# creates .venv311 with all dependencies and configures VS Code.
#
# Usage (PowerShell, no admin needed - winget may ask for UAC):
#   irm https://raw.githubusercontent.com/VojtechMarek13/SmartSense/main/scripts/setup_new_pc.ps1 | iex
# or, from an existing clone:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_new_pc.ps1

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/VojtechMarek13/SmartSense.git"
$Target  = if ($env:SMARTSENSE_DIR) { $env:SMARTSENSE_DIR } else { "C:\SW_projects\SmartSense" }

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

function Winget-Install($id) {
    # --source winget: skip the msstore source, which fails on some networks
    # (certificate error 0x8a15005e) and makes winget refuse to pick a package.
    winget install --id $id -e --source winget --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget failed to install $id (exit code $LASTEXITCODE)" }
    Refresh-Path
}

function Ensure-WingetPackage($id, $probe) {
    if (Get-Command $probe -ErrorAction SilentlyContinue) {
        Write-Host "    $id already installed"
        return
    }
    Winget-Install $id
}

# 1) System tools
Step "Installing Git, Python 3.11, cloudflared"
Ensure-WingetPackage "Git.Git" "git"
if (-not (Get-Command py -ErrorAction SilentlyContinue) -or -not (& py -3.11 --version 2>$null)) {
    Winget-Install "Python.Python.3.11"
}
Ensure-WingetPackage "Cloudflare.cloudflared" "cloudflared"

# 2) Source code
Step "Getting source code into $Target"
if (Test-Path (Join-Path $Target ".git")) {
    git -C $Target -c http.sslBackend=schannel pull --ff-only
} else {
    New-Item -ItemType Directory -Force (Split-Path $Target) | Out-Null
    git -c http.sslBackend=schannel clone $RepoUrl $Target
}
Set-Location $Target

# 3) Virtual environment + dependencies
Step "Creating .venv311 and installing dependencies (torch is large, takes a while)"
if (-not (Test-Path ".venv311\Scripts\python.exe")) { py -3.11 -m venv .venv311 }
$py = Join-Path $Target ".venv311\Scripts\python.exe"
# truststore = use the Windows certificate store (works behind TLS-inspecting proxies)
& $py -m pip install --use-feature=truststore --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $py -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
& $py -m pip install -r requirements-web.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 4) VS Code
Step "Configuring VS Code"
New-Item -ItemType Directory -Force ".vscode" | Out-Null
@"
{
  "python.defaultInterpreterPath": "`${workspaceFolder}\\.venv311\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true
}
"@ | Set-Content -Encoding utf8 ".vscode\settings.json"
if (Get-Command code -ErrorAction SilentlyContinue) {
    code --install-extension ms-python.python --force | Out-Null
}
# Allow venv activation scripts in the VS Code terminal
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force

# 5) Sanity check
Step "Verifying imports"
& $py -c "import numpy, torch, PyQt6, matplotlib, fastapi, asyncua; print('    all imports OK')"

if (-not (Test-Path "data")) {
    Write-Host "`n[!] Folder data\ is missing - copy the measurement data (~1.7 GB) from the old PC." -ForegroundColor Yellow
    Write-Host "    Target layout: data\Joint {1-5}\{DD.MM.YYYY}\Cobot Stand\Trajectory {1-5}\*.csv"
}

Write-Host "`nDone. Open the project with:  code `"$Target`"" -ForegroundColor Green
Write-Host "Web dashboard:  .venv311\Scripts\python -m App.web.server   -> http://localhost:8000"
Write-Host "Desktop GUI:    .venv311\Scripts\python -m App.main"
