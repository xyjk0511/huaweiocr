# HuaweiOCR

[English](README.md) | [中文](README_ZH.md)

## One-line summary
An automated device-label recognition tool: crop fields -> barcode decode -> OCR -> structured export (JSONL), with Windows one-click run.

## Overview
This project is a local batch-processing tool for device label / barcode images. It combines object detection (Roboflow inference), barcode decoding, and OCR to extract Model and SN. The core idea is "locate first, then recognize":

- Detect label areas in the full image using a Roboflow model.
- Crop label areas and then crop model/sn sub-fields.
- Run barcode decoding and OCR on the cropped fields.
- Output structured results (JSONL) and debug logs for tracing and tuning.


## Pipeline
```
Input -> crop -> barcode -> ocr -> postprocess -> output
```

## Highlights
- End-to-end batch processing: from raw images to results in one run.
- Multi-stage cropping and filtering to improve accuracy.
- Dual-channel decoding: barcode first, OCR as fallback.
- Debug artifacts saved for easier troubleshooting.
- Local/offline dependencies bundled to reduce external setup.

## Core structure
- `crop.py`: stage 1/2 cropping (Roboflow detect + crop)
- `scan2.py`: barcode/OCR + structured output
- `barcode.py`: barcode enhancement pipeline
- `run_all.py`: one-click pipeline entry
- `start.bat`: Windows one-click script

## Environment
- Windows
- Python 3.10+ (recommended)
- Valid Roboflow API key

## Quick start (Windows)
1) Create `.env` in the project root (do not commit):
```
API_KEY=your_api_key_here
```

2) Double-click `start.bat`.

## CLI
```
python run_all.py --input ./images --out ./out --format jsonl --log-level info --device cpu
```

Full options:
```
python run_all.py --help
```

## Pipeline overview
1) Read images from `new_images/`.
2) Stage 1: detect and crop label areas -> `stage1_labels/`.
3) Stage 2: crop model/sn fields -> `stage2_fields/`.
4) `scan2.py` runs barcode + OCR.
5) Outputs `model_sn_ocr.jsonl` and `debug_ocr_barcode.log`.

## Outputs
- `stage1_labels/`: label crops
- `stage2_fields/model/`: model crops
- `stage2_fields/sn/`: SN crops
- `model_sn_ocr.jsonl`: final results (one JSON per line)
- `debug_ocr_barcode.log`: debug log
Default output is repo root; use `--out` to change it.

## Quantitative metrics
CLI prints:
- total images, total time, average time per image
- SN extraction success rate
- regex pass rate
- error distribution (barcode_fail / ocr_fail / regex_fail)

## Output format example
Single JSONL line:
```
{"label_id":"img_001__label_1","model":"S380-S8P2T","sn":"4E25XXXXXXXX","model_src":"barcode","sn_src":"ocr","model_raw":"...","sn_raw":"..."}
```

## Robustness strategies
- Multi-scale upscaling for small barcodes
- ROI cropping to reduce noise
- Rotation attempts (0/90/180/270)
- Regex validation for SN/Model
- Failure samples saved for review

## Roadmap
- CSV export and configurable field mapping
- Finer error taxonomy and visual reports
- Incremental CLI processing with resume support
- Lighter releases (optional LFS or model download scripts)

## FAQ
- No results: check image clarity, angle, lighting; or tune crop/thresholds.
- API_KEY missing: ensure `.env` exists and is correct.
- Unstable results: tune parameters in `crop.py` / `scan2.py`.

## Security
- `.env` is excluded from the repo, so API keys are not exposed.
- Share `.env` privately when needed.
- Do not hard-code API keys.
- Key rotation: update `API_KEY` in `.env` without code changes.
