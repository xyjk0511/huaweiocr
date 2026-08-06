# HuaweiOCR

[English](README_EN.md) | [中文](README_ZH.md)

Windows-first batch OCR pipeline for device labels.

## Quick Start

```bash
python -m pip install -r requirements.txt
```

Create `.env`:

```text
API_KEY=your_api_key_here
```

Put images in `new_images`, then run:

```bash
python run_all.py --input new_images --out runs --format jsonl --log-level info --device cpu
```

Or double-click `start.bat`. The script creates `new_images` if needed and stops with a clear prompt when the folder is empty.

## Outputs

Results are written under `runs/`. If output folders already exist, a per-run sibling folder is created to avoid overwrites.

- `stage1_labels/` or `stage1_labels_run_*`
- `stage2_fields/model/`
- `stage2_fields/sn/`
- `stage2_fields/manifest.jsonl`
- `stage2_fields/model_sn_ocr.jsonl`
- `stage2_fields/debug_ocr_barcode.log` only with `--log-level debug`

See [README_EN.md](README_EN.md) or [README_ZH.md](README_ZH.md) for details.
