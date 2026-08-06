## Why

SN extraction is a barcode-first requirement. The current pipeline can fall back to OCR, but OCR-derived SN values are not reliable enough for inventory or device traceability when the barcode is present.

This change raises the SN barcode path to a measurable production target: at least 90% exact SN barcode hit rate on a representative validation set, with OCR kept only as fallback evidence.

## What Changes

- Add a barcode-first SN recognition contract with an explicit hit-rate target and validation dataset.
- Expand SN barcode scanning beyond the current narrow SN crop by using the full label crop, detected SN regions, barcode-oriented candidate crops, rotations, scale passes, and decoder fallback.
- Record whether an SN came from barcode or OCR, and report barcode hit rate separately from OCR recovery rate.
- Add a repeatable evaluation command/report that fails if exact SN barcode hit rate is below 90%.
- Keep OCR as fallback only when barcode decoding fails or barcode text cannot be parsed as an SN.
- Do not treat OCR fallback as satisfying the barcode hit-rate target.

## Capabilities

### New Capabilities
- `sn-barcode-scanning`: Barcode-first SN detection, decoding, selection, and validation with at least 90% exact barcode hit rate.

### Modified Capabilities

## Impact

- Affected code: `barcode.py`, `scan2.py`, `crop.py`, `run_all.py`, GUI result display, and tests.
- Affected outputs: JSONL result rows need clearer `sn_src`, barcode metadata, and aggregate barcode metrics.
- Affected dependencies: barcode decoder strategy may need a stronger local decoder, CLI configuration changes, or an optional decoder abstraction.
- Affected validation: add a curated local validation set with ground-truth SN values and a deterministic evaluation report.
