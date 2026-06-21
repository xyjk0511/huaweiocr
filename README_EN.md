# HuaweiOCR

Windows-first batch OCR pipeline for device labels.

## What It Does

The pipeline detects label regions, crops model/PartNo/SN fields, decodes label-local barcodes, falls back to OCR, and writes structured JSONL.

SN barcode recognition only uses crops bound to the current label, such as the SN crop, barcode candidates, and the label crop. The original full source photo is kept as provenance metadata only and is not scanned as an SN barcode fallback, because one photo can contain multiple valid labels.

Detection uses local ONNX models by default: `local_models/detectors/label_detector.onnx` for stage1 label crops and `local_models/detectors/field_detector.onnx` for model/PartNo/SN crops. The normal local path does not require a Roboflow API key.

## Requirements

- Windows
- Python 3.12 recommended
- Dependencies pinned in `requirements.txt`
- No `.env` is required for default local detection; an API key is only needed when explicitly using Roboflow

Install:

```bash
python -m pip install -r requirements.txt
```

To temporarily switch back to Roboflow, create `.env` and set the backend:

```text
API_KEY=your_api_key_here
CROP_INFERENCE_BACKEND=roboflow
```

Local detection runs in parallel by default and chooses conservative stage1/stage2 worker counts from the machine and backend. Override with `CROP_STAGE1_WORKERS`, `CROP_STAGE2_WORKERS`, or `CROP_WORKERS`. When `onnxruntime-gpu` is installed on an NVIDIA GPU machine, `LOCAL_YOLO_DEVICE=auto` prefers CUDA; otherwise it falls back to CPU.

Recognition is scheduled in two parallel pools: the larger barcode worker pool runs first, and only barcode misses enter the smaller OCR worker pool. Auto defaults are conservative and CPU-aware, with barcode workers capped at 8 and OCR defaulting to 1 because PaddleOCR multi-instance concurrent initialization is unstable. Override with `SCAN2_BARCODE_WORKERS`, `SCAN2_OCR_WORKERS`, or `SCAN2_WORKERS`; set `SCAN2_PARALLEL=0` to temporarily disable recognition parallelism.

## Windows Start

Double-click `start.bat`.

The script creates `new_images` if needed. If the folder has no supported image files, it stops and asks you to add images instead of running an empty pipeline.

Results are written under `runs/`. If an output folder already exists, the app creates a per-run sibling folder to avoid overwriting files.

## CLI

```bash
python run_all.py --input new_images --out runs --format jsonl --log-level info --device cpu
```

Useful options:

```bash
python run_all.py --help
```

## Outputs

Typical run outputs:

- `stage1_labels/` or `stage1_labels_run_*`
- `stage2_fields/model/`
- `stage2_fields/part_no/`
- `stage2_fields/sn/`
- `stage2_fields/manifest.jsonl`
- `stage2_fields/model_sn_ocr.jsonl`
- `stage2_fields/debug_ocr_barcode.log` only when `--log-level debug`

`model_raw` and `sn_raw` are masked by default in result JSONL. Set `SCAN2_UNSAFE_RAW=1` or `HUAWEIOCR_UNSAFE_RAW=1` only for trusted local debugging when complete raw values are required.
Model fields use barcode-first recognition by default. Set `SCAN2_MODEL_BARCODE=0` only when you need to temporarily disable barcode decoding on model crops.

Example JSONL line:

```json
{"label_id":"sample_label_001.png__label_1","model":"S380-S8P2T","sn":"2000000000AGQC000000","model_raw":"[masked-model-raw]","sn_raw":"2000********0000","model_src":"ocr_color","sn_src":"barcode"}
```

## GUI

Run:

```bash
python gui_app.py
```

or:

```bash
python gui_app_en.py
```

The GUI copies selected images into a unique per-run input folder, prevents repeated concurrent runs, and exports Excel from the original JSONL rows. The table, run log, and exported `model` / `sn` values keep the complete recognized values.

The default OCR profile uses `en_PP-OCRv5_mobile_rec`, which is more stable on the current label samples than the server recognizer. To compare the server recognizer, add `HUAWEIOCR_OCR_PROFILE=server` to `.env`.

## Tests

Run:

```bash
python -m unittest discover -v
```

The `tests` package includes regression tests for output directory isolation, manifest parsing, barcode CLI attempt budget, debug-log masking, GUI input staging, and model install lock recovery.

## Security Notes

- Do not commit `.env`.
- Do not hard-code API keys.
- Debug logs are disabled unless `--log-level debug`.
- GUI logs and self-check logs mask local paths.
- PyInstaller packaging includes only the barcode CLI runtime files, not vendor examples/configs.
