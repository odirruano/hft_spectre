# ==============================
# HFT Spectre - NinjaTrader Sync
# ==============================

$source = "C:\HFT_SPECTRE\ninjatrader\Strategies\HFTSpectreStrategy.cs"
$destination = "C:\Users\oruano\Documents\NinjaTrader 8\bin\Custom\Strategies\HFTSpectreStrategy.cs"

if (!(Test-Path $source)) {
    Write-Host "ERROR: Source file not found." -ForegroundColor Red
    exit
}

Copy-Item -Path $source -Destination $destination -Force

Write-Host "Strategy successfully synced to NinjaTrader." -ForegroundColor Green