# Debug Session: code-review-findings

## Symptoms

- Final JSONL can omit labels that were cropped but produced neither model nor SN crops.
- Filenames containing `__` can be grouped under the wrong label key.
- GUI-pasted images can be saved into the same directory that the run later clears.
- Empty or missing CLI input can continue into scan and look like a successful run.
- Crop output cleanup can delete previous run evidence by default.
- Roboflow temp images use deterministic names and are not cleaned up.
- Barcode fallback can invoke too many external CLI processes.
- GUI imports can fail before the window appears when `API_KEY` is missing.
- Debug logs include raw barcode/OCR text by default.
- PyInstaller spec contains user-specific absolute paths.
- Python dependencies are not declared.

## Root Cause Hypotheses

- Result identity is inferred from generated filenames rather than the crop manifest.
- Pipeline state is held in module globals and fixed directories.
- GUI and CLI wrappers lack preflight validation for input and runtime configuration.
- Packaging/dependency metadata was not made portable after local build fixes.

## Current Status

- Root causes confirmed and fixed in code.
- Regression coverage added for manifest-preserved labels, `__` filename keys, empty input exit, and unique Roboflow temp files.
- Verification passed:
  - `python -m unittest discover -s tests -v`
  - `python -m py_compile 1.py app_paths.py barcode.py crop.py debug.py gui_app.py gui_app_en.py ocr.py run_all.py scan.py scan2.py tests\test_locked_output_dirs.py`
  - `git diff --check`

## Fix Summary

- `scan2.py` now initializes records from `stage2_fields/manifest.jsonl`, keeps missing-field labels in JSONL, and strips only known filename suffixes instead of splitting on `__`.
- `crop.py` now imports without `API_KEY`, creates the Roboflow client lazily, uses unique temporary inference images, cleans temp files, records stable `label_id`, and uses non-destructive run directories unless `--clean` is explicit.
- `run_all.py` now fails fast with exit code 2 for missing, unreadable, or empty input image directories and passes `--clean` explicitly to crop.
- `gui_app.py` and `gui_app_en.py` now store pasted images outside the pipeline input directory, skip self-deleting selected sources, use module log sinks instead of replacing `builtins.print`, and schedule Tk updates through `after`.
- `barcode.py` now bounds CLI fallback scale/pixels and stops after the first CLI hit.
- `HuaweiOCR.spec`, `requirements.txt`, `README.md`, and `.gitignore` now provide portable packaging/dependency metadata and ignore pasted image cache.
