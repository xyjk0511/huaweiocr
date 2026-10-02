param(
    [string]$Dataset = "$PSScriptRoot/../datasets/all_label_yolo26_2class_701515/data.yaml",
    [string]$Project = "$PSScriptRoot/../runs/huawei_label",
    [string]$SmallWeights = "$PSScriptRoot/../yolo26s.pt",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:POLARS_SKIP_CPU_CHECK = "1"

$yolo = Join-Path (Get-Location) ".venv/Scripts/yolo.exe"

if (-not (Test-Path $yolo)) {
    throw "Ultralytics CLI not found: $yolo"
}

if (-not (Test-Path $dataset)) {
    throw "Dataset YAML not found: $dataset"
}

if (-not (Test-Path $smallWeights)) {
    throw "Missing YOLO26 small weights: $smallWeights"
}

$name = "yolo26s_960_2class_ignore_v1"
$trainArgs = @(
    "detect", "train",
    "model=$smallWeights",
    "data=$dataset",
    "imgsz=960",
    "epochs=300",
    "batch=-1",
    "patience=70",
    "device=0",
    "workers=4",
    "pretrained=True",
    "amp=True",
    "optimizer=auto",
    "seed=42",
    "deterministic=True",
    "project=$project",
    "name=$name",
    "mosaic=0",
    "close_mosaic=0",
    "mixup=0",
    "copy_paste=0",
    "fliplr=0",
    "flipud=0",
    "degrees=4",
    "translate=0.03",
    "scale=0.10",
    "shear=0",
    "perspective=0",
    "hsv_h=0",
    "hsv_s=0.08",
    "hsv_v=0.15"
)

Write-Host "Starting YOLO26 2-class training"
Write-Host "Dataset: $dataset"
Write-Host "Project: $project"

if ($DryRun) {
    Write-Host "Command: $yolo $($trainArgs -join ' ')"
} else {
    & $yolo @trainArgs
}
