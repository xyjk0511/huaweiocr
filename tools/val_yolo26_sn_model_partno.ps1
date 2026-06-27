param(
    [string]$Model = "F:/HuaweiOCR/.worktrees/yolo26-retrain/runs/sn_model_partno/yolo26s_960_clean/weights/best.pt",
    [ValidateSet("train", "val", "test")]
    [string]$Split = "test",
    [int]$Device = 0,
    [int]$ImgSz = 960,
    [double]$Conf = 0.25,
    [double]$Iou = 0.60,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:POLARS_SKIP_CPU_CHECK = "1"

$repoRoot = (Get-Location).Path.Replace("\", "/")
$dataset = "$repoRoot/data.yaml"
$project = "$repoRoot/runs/sn_model_partno_val"
$yolo = "F:/HuaweiOCR/.venv/Scripts/yolo.exe"

if (-not (Test-Path $yolo)) {
    throw "Ultralytics CLI not found: $yolo"
}

if (-not (Test-Path $dataset)) {
    throw "Dataset YAML not found: $dataset"
}

if (-not (Test-Path $Model)) {
    throw "Model checkpoint not found: $Model"
}

$runName = "{0}_{1}_imgsz{2}_conf{3}" -f [System.IO.Path]::GetFileNameWithoutExtension($Model), $Split, $ImgSz, ($Conf.ToString("0.00").Replace(".", ""))
$valArgs = @(
    "detect", "val",
    "model=$Model",
    "data=$dataset",
    "imgsz=$ImgSz",
    "split=$Split",
    "conf=$Conf",
    "iou=$Iou",
    "device=$Device",
    "plots=True",
    "project=$project",
    "name=$runName"
)

Write-Host "Validating checkpoint: $Model"
Write-Host "Dataset: $dataset"
Write-Host "Command: $yolo $($valArgs -join ' ')"

if (-not $DryRun) {
    & $yolo @valArgs
}
