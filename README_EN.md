# HuaweiOCR

Windows-first batch OCR pipeline for device labels.

## What It Does

The pipeline detects label regions, crops model/SN fields, decodes SN barcodes, falls back to OCR, and writes structured JSONL.

Roboflow detection requires a valid `API_KEY`. Local PaddleOCR models and the barcode CLI can be bundled for packaging, but the detector step is not fully offline.

## Requirements

- Windows
- Python 3.12 recommended
- Dependencies pinned in `requirements.txt`
- Roboflow API key in `.env`

Install:

```bash
python -m pip install -r requirements.txt
```

Create `.env`:

```text
API_KEY=your_api_key_here
```

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
- `stage2_fields/sn/`
- `stage2_fields/manifest.jsonl`
- `stage2_fields/model_sn_ocr.jsonl`
- `stage2_fields/debug_ocr_barcode.log` only when `--log-level debug`

`model_raw` and `sn_raw` keep complete raw values by default for local review. Set `SCAN2_MASK_RAW=1` when you need a masked result file; if `SCAN2_UNSAFE_RAW=1` is also set, complete raw values are forced.
Model fields use barcode-first recognition by default. Set `SCAN2_MODEL_BARCODE=0` only when you need to temporarily disable barcode decoding on model crops.

Example JSONL line:

```json
{"label_id":"input_0001.png__label_1","model":"S380-S8P2T","sn":"4E25A0170000","model_raw":"S380-S8P2T","sn_raw":"4E25A0170000","model_src":"ocr_color","sn_src":"barcode"}
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
