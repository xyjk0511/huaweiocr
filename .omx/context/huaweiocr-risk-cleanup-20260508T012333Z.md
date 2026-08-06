# HuaweiOCR Risk Cleanup Autopilot Context

## Task statement
Finish the remaining likely-risk cleanup after the SN cross-label fix: align validation behavior with the GUI pipeline, remove stale original-image fallback guidance, repair unstable tests, and verify the repository.

## Desired outcome
- Main GUI/CLI SN recognition cannot use whole-source-photo barcode fallback.
- Validation tooling cannot count whole-source-photo SN scans as barcode hits.
- OpenSpec notes no longer instruct future agents to reintroduce original-image fallback.
- Full unittest discovery passes in a clean process.
- Changes are reviewed, committed, and pushed if clean.

## Known facts/evidence
- Current `master`/`origin/master` is `8e9ee50 Prevent SN cross-label barcode reuse`.
- `scan2.main()` passes `original_path=""` into `recognize_sn`, but `recognize_sn` still accepts and scans `original_path`.
- `validate_sn_barcodes._sources_for_row()` still appends `("original", original_image_path or image_path)`.
- Full `python -m unittest discover -s tests` currently fails: stale log-format assertion and fake `numpy` pollution before GUI import.
- Packaged release artifacts are ignored by git; source/spec changes are the commit surface.

## Constraints
- Preserve barcode-first behavior for current SN crop and label crop.
- Do not reintroduce whole-source-photo fallback for SN.
- Keep scope narrow; avoid broad OCR/barcode algorithm refactors.
- Use existing unittest test surface.

## Unknowns/open questions
- Whether `validate_sn_barcodes` should keep original-image fields only as metadata or remove them from template rows entirely. Default: keep metadata but do not scan them.
- Whether `recognize_sn` signature can be changed safely. Default: keep backward-compatible argument but ignore it with a debug note.

## Likely codebase touchpoints
- `scan2.py`
- `validate_sn_barcodes.py`
- `tests/test_sn_barcode_scanning.py`
- `tests/test_locked_output_dirs.py`
- `openspec/changes/raise-sn-barcode-scan-hit-rate/design.md`
- `openspec/changes/raise-sn-barcode-scan-hit-rate/tasks.md`
