$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:POLARS_SKIP_CPU_CHECK = "1"

$dataset = "F:/HuaweiOCR/.worktrees/yolo26-retrain/datasets/all_label_yolo26/data.yaml"
$project = "F:/HuaweiOCR/.worktrees/yolo26-retrain/runs/huawei_label"
$yolo = "F:/HuaweiOCR/.venv/Scripts/yolo.exe"
$nanoWeights = "F:/HuaweiOCR/yolo26n.pt"
$smallWeights = "yolo26s.pt"

if (-not (Test-Path $yolo)) {
    throw "Ultralytics CLI not found: $yolo"
}

if (-not (Test-Path $dataset)) {
    throw "Dataset YAML not found: $dataset"
}

if (-not (Test-Path $nanoWeights)) {
    throw "Missing YOLO26 nano weights: $nanoWeights"
}

$variant = if ($args.Count -gt 0) { $args[0].ToLowerInvariant() } else { "nano" }

switch ($variant) {
    "nano" {
        $name = "yolo26n_640_noaug"
        $trainArgs = @(
            "detect", "train",
            "model=$nanoWeights",
            "data=$dataset",
            "imgsz=640",
            "epochs=300",
            "batch=-1",
            "patience=60",
            "device=0",
            "amp=True",
            "pretrained=True",
            "workers=4",
            "optimizer=AdamW",
            "deterministic=True",
            "seed=0",
            "project=$project",
            "name=$name",
            "mosaic=0",
            "mixup=0",
            "copy_paste=0",
            "fliplr=0",
            "flipud=0",
            "degrees=0",
            "translate=0",
            "scale=0",
            "hsv_h=0",
            "hsv_s=0",
            "hsv_v=0"
        )
    }
    "small" {
        $name = "yolo26s_960_lightaug"
        $trainArgs = @(
            "detect", "train",
            "model=$smallWeights",
            "data=$dataset",
            "imgsz=960",
            "epochs=300",
            "batch=-1",
            "patience=70",
            "device=0",
            "amp=True",
            "pretrained=True",
            "workers=4",
            "optimizer=AdamW",
            "deterministic=True",
            "seed=0",
            "project=$project",
            "name=$name",
            "mosaic=0",
            "mixup=0",
            "copy_paste=0",
            "fliplr=0",
            "flipud=0",
            "degrees=3",
            "translate=0.03",
            "scale=0.10",
            "hsv_h=0",
            "hsv_s=0.15",
            "hsv_v=0.15"
        )
    }
    default {
        throw "Unknown variant '$variant'. Use: nano or small"
    }
}

Write-Host "Starting YOLO26 training variant: $variant"
Write-Host "Dataset: $dataset"
Write-Host "Project: $project"

& $yolo @trainArgs
