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

## Follow-up Review Findings

- GUI selected images with the same basename are now staged into a per-run input directory with unique target names and a `source_manifest.jsonl`.
- `barcode.py` now enforces `CLI_MAX_CALLS_PER_PATCH` so one no-hit patch cannot fan out into unbounded external CLI launches.
- `gui_app.py` and `gui_app_en.py` no longer import `crop`, `scan2`, OCR, barcode, Paddle, or pyzbar during window startup; pipeline modules are loaded only after the user clicks run, and load failures are shown in the GUI.
- `scan2.py` no longer writes debug logs in normal info mode. Debug mode uses the existing masking path for sensitive OCR/barcode text.
- Legacy local scripts were moved under `legacy/` and marked reference-only, with supported entrypoints documented in `legacy/README.md`.
- `app_paths.ensure_models_installed()` now uses a lock file, copies bundled models into a temporary directory, writes a completion marker, and replaces incomplete installs on the next run.

## Follow-up Verification

- `python -m unittest discover -s tests -v`
- `python -m py_compile app_paths.py barcode.py crop.py debug.py gui_app.py gui_app_en.py gui_pipeline.py legacy\roboflow_legacy.py legacy\barcode_debug_legacy.py ocr.py run_all.py scan2.py tests\test_locked_output_dirs.py`
- `git diff --check`

## Subagent Review Closure

- Subagent review found two remaining blockers after `b9dbb23`: stale model install locks were not recoverable, and GUI `source_manifest.jsonl` stored absolute source/input paths by default.
- `app_paths.py` now writes lock metadata, reclaims stale/malformed locks, and keeps the temporary-copy plus completion-marker install flow.
- `gui_pipeline.py` now stages files as `input_0001.ext` style names and writes only `source_index`, `input_name`, and content `sha256`; absolute paths are no longer persisted in the default GUI manifest.
- Regression coverage now includes stale lock recovery and asserts GUI staging manifest rows do not contain absolute paths.
