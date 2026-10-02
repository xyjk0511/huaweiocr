param(
    [string]$Model = "$PSScriptRoot/../runs/huawei_label/yolo26n_640_noaug/weights/best.pt",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:POLARS_SKIP_CPU_CHECK = "1"

$yolo = Join-Path (Get-Location) ".venv/Scripts/yolo.exe"

if (-not (Test-Path $yolo)) {
    throw "Ultralytics CLI not found: $yolo"
}

if (-not (Test-Path $model)) {
    throw "Model checkpoint not found: $model"
}

$exportArgs = @("export", "model=$Model", "format=onnx", "imgsz=640", "simplify=True")
if ($DryRun) {
    Write-Host "Command: $yolo $($exportArgs -join ' ')"
} else {
    & $yolo @exportArgs
}
