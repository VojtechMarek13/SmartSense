# SmartSense - install the station event logger (Power BI efficiency dashboard).
#
# Puts a shortcut into the user's Startup folder that runs the logger at logon
# (no admin needed) and starts it right away. The logger writes STATION_ON /
# STATION_OFF into Desktop\Home\sf6\events; SmartSense adds CYCLE_START / CYCLE_DONE.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_station_logger.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_station_logger.ps1 -Uninstall

param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$Root     = Split-Path -Parent $PSScriptRoot
$Pythonw  = Join-Path $Root ".venv311\Scripts\pythonw.exe"
$Shortcut = Join-Path ([Environment]::GetFolderPath("Startup")) "SmartSense Station Logger.lnk"

if ($Uninstall) {
    if (Test-Path $Shortcut) { Remove-Item $Shortcut }
    Write-Host "Removed $Shortcut (a running logger keeps running until logoff)."
    return
}

if (-not (Test-Path $Pythonw)) { throw "Missing $Pythonw - run scripts\setup_new_pc.ps1 first." }

$lnk = (New-Object -ComObject WScript.Shell).CreateShortcut($Shortcut)
$lnk.TargetPath       = $Pythonw
$lnk.Arguments        = "-m src.events.station_logger"
$lnk.WorkingDirectory = $Root
$lnk.Description      = "SmartSense station event logger"
$lnk.Save()
Write-Host "Created $Shortcut"

# Start now (a second instance exits on its own, so this is safe to re-run)
Start-Process -FilePath $Pythonw -ArgumentList "-m src.events.station_logger" -WorkingDirectory $Root
Write-Host "Station logger started. Events: $([Environment]::GetFolderPath('Desktop'))\Home\sf6\events" -ForegroundColor Green
