$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:POLARS_SKIP_CPU_CHECK = "1"

$yolo = "F:/HuaweiOCR/.venv/Scripts/yolo.exe"
$defaultModel = "F:/HuaweiOCR/.worktrees/yolo26-retrain/runs/huawei_label/yolo26n_640_noaug/weights/best.pt"

$model = if ($args.Count -gt 0) { $args[0] } else { $defaultModel }

if (-not (Test-Path $yolo)) {
    throw "Ultralytics CLI not found: $yolo"
}

if (-not (Test-Path $model)) {
    throw "Model checkpoint not found: $model"
}

& $yolo "export" "model=$model" "format=onnx" "imgsz=640" "simplify=True"
