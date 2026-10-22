# Train every experiment, then calibrate the face detector and evaluate everything.
#   powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1 -ApacSource "D:\data\APAC asian images.zip"
#   powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1 -Experiments baseline,proposed -SkipEval
param(
    [string]$Python = "$PSScriptRoot\..\.venv\Scripts\python.exe",
    [string[]]$Experiments = @("baseline", "augmented", "proposed", "robust"),
    [string]$ApacSource = "",
    [switch]$SkipEval
)
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..")
New-Item -ItemType Directory -Force "results\logs" | Out-Null

& $Python -m src.dataset
if ($LASTEXITCODE -ne 0) { Write-Error "Dataset preparation failed"; exit 1 }

# src.train writes its own UTF-8 log to results\logs\train_<experiment>.log
foreach ($exp in $Experiments) {
    if ($exp -eq "robust" -and -not (Test-Path "data\cache\detector_boxes.npz")) {
        # The detector-robust model trains on MediaPipe-style boxes: learn the box mapping, then export boxes.
        if (-not (Test-Path "checkpoints\mediapipe_calibration.json")) { & $Python -m src.mediapipe_baseline }
        & $Python -m src.mediapipe_baseline --export-boxes
    }
    Write-Output "=== START $exp $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & $Python -m src.train --experiment $exp
    Write-Output "=== END $exp exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

function Invoke-Logged([string]$Log, [string[]]$PyArgs) {
    # Windows PowerShell 5.1's Tee-Object writes UTF-16; collect the lines and save them as UTF-8.
    $lines = & $Python @PyArgs | ForEach-Object { Write-Host $_; $_ }
    $lines | Out-File -FilePath $Log -Encoding utf8
}

if (-not $SkipEval) {
    Invoke-Logged "results\logs\calibrate.log" @("-m", "src.inference", "--calibrate-detector")
    Invoke-Logged "results\logs\evaluate.log" @("-m", "src.evaluate")
    Invoke-Logged "results\logs\mediapipe_baseline.log" @("-m", "src.mediapipe_baseline")
    if ($ApacSource) {
        Invoke-Logged "results\logs\apac_eval.log" @("-m", "src.apac_eval", "--source", $ApacSource)
    }
}
Write-Output "=== ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
