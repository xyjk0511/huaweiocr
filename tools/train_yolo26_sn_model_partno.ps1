param(
    [ValidateSet("small", "nano")]
    [string]$Variant = "small",
    [int]$Device = 0,
    [int]$Workers = 4,
    [ValidateSet("ram", "disk", "False")]
    [string]$Cache = "ram",
    [Nullable[int]]$Batch = $null,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:POLARS_SKIP_CPU_CHECK = "1"

$repoRoot = (Get-Location).Path.Replace("\", "/")
$dataset = "$repoRoot/data.yaml"
$project = "$repoRoot/runs/sn_model_partno"
$yolo = "F:/HuaweiOCR/.venv/Scripts/yolo.exe"
$nanoWeights = "$repoRoot/yolo26n.pt"
$smallWeights = "$repoRoot/yolo26s.pt"

if (-not (Test-Path $yolo)) {
    throw "Ultralytics CLI not found: $yolo"
}

if (-not (Test-Path $dataset)) {
    throw "Dataset YAML not found: $dataset"
}

$baseArgs = @(
    "detect", "train",
    "data=$dataset",
    "imgsz=960",
    "batch=$(-1)",
    "device=$Device",
    "workers=$Workers",
    "pretrained=True",
    "amp=True",
    "optimizer=auto",
    "seed=42",
    "deterministic=True",
    "cache=$Cache",
    "project=$project",
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
    "hsv_s=0.05",
    "hsv_v=0.12"
)

switch ($Variant) {
    "small" {
        if (-not (Test-Path $smallWeights)) {
            throw "Missing YOLO26 small weights: $smallWeights"
        }
        $trainArgs = $baseArgs + @(
            "model=$smallWeights",
            "epochs=260",
            "patience=70",
            "name=yolo26s_960_clean"
        )
    }
    "nano" {
        if (-not (Test-Path $nanoWeights)) {
            throw "Missing YOLO26 nano weights: $nanoWeights"
        }
        $trainArgs = $baseArgs + @(
            "model=$nanoWeights",
            "epochs=220",
            "patience=60",
            "name=yolo26n_960_clean"
        )
    }
}

if ($Batch -ne $null) {
    $trainArgs = $trainArgs | Where-Object { $_ -notlike "batch=*" }
    $trainArgs += "batch=$Batch"
}

Write-Host "Starting YOLO26 training variant: $Variant"
Write-Host "Dataset: $dataset"
Write-Host "Project: $project"
Write-Host "Command: $yolo $($trainArgs -join ' ')"

if (-not $DryRun) {
    & $yolo @trainArgs
}
