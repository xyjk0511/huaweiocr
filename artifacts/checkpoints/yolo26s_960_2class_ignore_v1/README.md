Saved checkpoints for the current Huawei label retraining baseline.

Source run:
- `runs/huawei_label/yolo26s_960_2class_ignore_v1`

Files:
- `best.pt`: best validation checkpoint used for current image checks
- `last.pt`: final training checkpoint
- `results.csv`: epoch metrics exported by Ultralytics

Training intent:
- 2 classes
- `0: huawei_label`
- `1: shipping_ignore`
- `imgsz=960`
- small model baseline: `yolo26s.pt`

Notes:
- These files are copied here because `runs/` is gitignored.
- The repo `run_all.py` local path still expects ONNX detectors; these `.pt` checkpoints are for the current Ultralytics-based training and batch prediction workflow.
