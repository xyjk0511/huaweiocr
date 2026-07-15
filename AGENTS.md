# HuaweiOCR Agent Notes

This repository is a Windows-first OCR pipeline for Huawei device labels.

## Project Rules

- Keep SN/model work narrow unless the user explicitly broadens scope.
- SN recognition is barcode-first, then OCR fallback.
- SN barcode hit-rate evidence must come from label-local sources only: SN crop, barcode-region candidates, and label crop.
- Do not add or restore whole-source-photo SN barcode fallback. Original image paths are provenance metadata only because one source photo can contain multiple labels.
- Keep barcode hits and OCR fallback metrics separate; OCR recovery must not inflate barcode hit rate.
- Do not commit `.env` or hard-code `API_KEY`.

## Verification

Use targeted tests first, then the full suite when changing recognition or packaging behavior:

```powershell
python -m unittest tests.test_sn_barcode_scanning
python -m unittest discover -s tests
```

For recognition-affecting changes (crop.py/scan2.py), additionally run the
115-image baseline regression check (opt-in, not part of CI since the source
photos are gitignored):

```powershell
$env:HUAWEIOCR_RUN_BASELINE_REGRESSION = "1"
python -m unittest tests.test_baseline_regression
# or directly:
python tools\check_baseline_regression.py
```

See `validation/baseline/README.md` for how to materialize `batch_runs/baseline_input`.

For package changes, build with:

```powershell
python -m PyInstaller --noconfirm HuaweiOCR.spec
```

Then verify the packaged app starts and that `_internal\Cython\Utility\CppSupport.cpp` exists (the build venv must have Cython installed, or the spec's collect silently packages nothing).

`.env` is intentionally NOT packaged since 2026-07: the API key must not ship in releases. The packaged app uses the local ONNX backend; Roboflow mode requires setting `API_KEY` via environment variable.

## Documented Knowledge

- `docs/solutions/` contains documented fixes and practices for past bugs, organized by category with YAML frontmatter.
- `CONCEPTS.md` defines shared OCR pipeline vocabulary such as Stage1 label crops, Stage2 field crops, PartNo fields, and label-local evidence.

## Commit Messages

Use concise decision-record commit messages with trailers when useful:

```text
<intent line>

Constraint: <external constraint>
Rejected: <alternative> | <reason>
Confidence: <low|medium|high>
Scope-risk: <narrow|moderate|broad>
Directive: <future warning>
Tested: <verification>
Not-tested: <known gap>
```
