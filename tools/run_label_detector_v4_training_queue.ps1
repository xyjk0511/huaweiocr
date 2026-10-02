$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$repoRoot = (Get-Location).Path
$data = Join-Path $repoRoot "datasets/huawei_yolov8_v4_boxonly/local_data.yaml"
$project = Join-Path $repoRoot "local_models/training"
New-Item -ItemType Directory -Path $project -Force | Out-Null
$queueLog = Join-Path $project "label_detector_v4_training_queue.log"

$commonArgs = @(
    "detect", "train", "data=$data",
    "epochs=300", "patience=60", "batch=-1", "device=0", "workers=8",
    "cache=disk", "optimizer=AdamW", "lr0=0.0015", "lrf=0.01",
    "weight_decay=0.0005", "warmup_epochs=5", "cos_lr=True", "amp=True",
    "degrees=7", "translate=0.04", "scale=0.25", "shear=1.5",
    "perspective=0.0006", "hsv_h=0.01", "hsv_s=0.25", "hsv_v=0.40",
    "flipud=0", "fliplr=0", "mosaic=0", "mixup=0", "copy_paste=0", "erasing=0",
    "project=$project"
)
$jobs = @(
    @{ Name = "label_detector_v4s_yolov8s_1280"; Model = "yolov8s.pt"; ImageSize = 1280 },
    @{ Name = "label_detector_v4n_yolov8n_960"; Model = "yolov8n.pt"; ImageSize = 960 },
    @{ Name = "label_detector_v4n_yolov8n_1280"; Model = "yolov8n.pt"; ImageSize = 1280 }
)

"$(Get-Date -Format s) queue start" | Tee-Object -FilePath $queueLog -Append
foreach ($job in $jobs) {
    $log = Join-Path $project ($job.Name + ".log")
    "$(Get-Date -Format s) START $($job.Name)" | Tee-Object -FilePath $queueLog -Append
    $trainArgs = $commonArgs + @("model=$($job.Model)", "imgsz=$($job.ImageSize)", "name=$($job.Name)")
    & yolo @trainArgs 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        "$(Get-Date -Format s) FAIL $($job.Name) exit=$LASTEXITCODE" | Tee-Object -FilePath $queueLog -Append
        exit $LASTEXITCODE
    }
    "$(Get-Date -Format s) DONE $($job.Name)" | Tee-Object -FilePath $queueLog -Append
}
"$(Get-Date -Format s) queue complete" | Tee-Object -FilePath $queueLog -Append
