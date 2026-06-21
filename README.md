# HuaweiOCR

[English](README_EN.md) | [中文](README_ZH.md)

Windows-first batch OCR pipeline for device labels.

## Quick Start

```bash
python -m pip install -r requirements.txt
```

The detector runs locally by default with ONNX weights under
`local_models/detectors`; no Roboflow API key is required for the normal path.

Put images in `new_images`, then run:

```bash
python run_all.py --input new_images --out runs --format jsonl --log-level info --device cpu
```

Or double-click `start.bat`. The script creates `new_images` if needed and stops with a clear prompt when the folder is empty.

Optional Roboflow mode is still available with `CROP_INFERENCE_BACKEND=roboflow`
and an `.env` containing `API_KEY=...`.
Local detection chooses conservative stage1/stage2 worker counts from the
machine and backend; override with `CROP_STAGE1_WORKERS`,
`CROP_STAGE2_WORKERS`, or `CROP_WORKERS`.
Recognition also runs in two scheduled pools: barcode work runs first with a
larger worker pool, and only misses fall back to the smaller OCR pool. Override
with `SCAN2_BARCODE_WORKERS`, `SCAN2_OCR_WORKERS`, or `SCAN2_WORKERS`.

## Outputs

Results are written under `runs/`. If output folders already exist, a per-run sibling folder is created to avoid overwrites.

- `stage1_labels/` or `stage1_labels_run_*`
- `stage2_fields/model/`
- `stage2_fields/sn/`
- `stage2_fields/manifest.jsonl`
- `stage2_fields/model_sn_ocr.jsonl`
- `stage2_fields/debug_ocr_barcode.log` only with `--log-level debug`

See [README_EN.md](README_EN.md) or [README_ZH.md](README_ZH.md) for details.
