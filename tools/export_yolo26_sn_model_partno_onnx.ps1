param(
    [string]$Model = "F:/HuaweiOCR/.worktrees/yolo26-retrain/runs/sn_model_partno/yolo26s_960_clean/weights/best.pt",
    [ValidateSet("onnx", "openvino")]
    [string]$Format = "onnx",
    [int]$ImgSz = 960,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:POLARS_SKIP_CPU_CHECK = "1"

$yolo = "F:/HuaweiOCR/.venv/Scripts/yolo.exe"

if (-not (Test-Path $yolo)) {
    throw "Ultralytics CLI not found: $yolo"
}

if (-not (Test-Path $Model)) {
    throw "Model checkpoint not found: $Model"
}

$exportArgs = @(
    "export",
    "model=$Model",
    "format=$Format",
    "imgsz=$ImgSz"
)

if ($Format -eq "onnx") {
    $exportArgs += "dynamic=True"
    $exportArgs += "simplify=True"
}

Write-Host "Exporting checkpoint: $Model"
Write-Host "Command: $yolo $($exportArgs -join ' ')"

if (-not $DryRun) {
    & $yolo @exportArgs
}
