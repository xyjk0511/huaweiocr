# PRD: HuaweiOCR Risk Cleanup

## Objective
Close the remaining risks after removing whole-source-photo SN fallback.

## Requirements
1. SN recognition must only use current SN crop and current label crop in production paths.
2. Validation hit-rate tooling must use the same source policy as production.
3. Documentation must not instruct future agents to use original-image SN fallback.
4. Full unittest discovery must pass in the current Python environment.
5. Existing barcode-first behavior for current label-local inputs must remain intact.

## Acceptance Criteria
- `recognize_sn` ignores or rejects whole-source original image input for barcode scanning.
- `validate_sn_barcodes._sources_for_row()` does not add an `original` source.
- OpenSpec text says original-image fallback was removed/forbidden for SN selection.
- `python -m unittest discover -s tests` exits 0.
- Focused SN tests still pass.
