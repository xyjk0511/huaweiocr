## 1. Validation Baseline

- [ ] 1.1 Define the SN barcode validation manifest schema with image path, label id, expected SN, barcode-present flag, accepted-quality flag, and notes.
- [ ] 1.2 Add a deterministic validation command that runs SN barcode extraction without counting OCR fallback as success.
- [ ] 1.3 Build or import a ground-truth validation set with enough accepted-quality SN barcode samples to justify the 90% gate.
- [ ] 1.4 Add quality diagnostics for too-small, clipped, blurred, or quiet-zone-missing SN barcodes.

## 2. Barcode Candidate Generation

- [ ] 2.1 Add barcode-region candidate detection on full label crops using 1D stripe morphology and contour filtering.
- [ ] 2.2 Generate candidate image variants with rotation, upscale, contrast enhancement, thresholding, and quiet-zone padding.
- [ ] 2.3 Add attempt budgets and early exit once a unique parseable SN barcode is found.
- [ ] 2.4 Add debug-only candidate dumps so missed barcode cases can be inspected without leaking sensitive raw values by default.

## 3. Decoder Integration

- [ ] 3.1 Introduce a local decoder adapter result schema with decoder name, raw text, source region, rotation, and confidence when available.
- [ ] 3.2 Update `pyzbar` and `BarcodeReaderCLI` calls to use the shared adapter schema and scan Code128/UCC128-oriented candidates.
- [ ] 3.3 Add a second local decoder adapter only if the validation report remains below 90% after candidate-generation tuning.
- [ ] 3.4 Keep decoder errors isolated so one decoder failure does not block the remaining barcode attempts.

## 4. SN Selection And Pipeline Output

- [ ] 4.1 Parse barcode payloads into SN candidates and reject non-SN payloads such as logistics codes, EAN values, and QR metadata.
- [ ] 4.2 Resolve duplicate barcode candidates and reject conflicting SN candidates as ambiguous instead of choosing silently.
- [ ] 4.3 Integrate SN crop, barcode-region crop, label crop, and original-image fallback sources before OCR fallback in `scan2.recognize_sn`.
- [ ] 4.4 Extend JSONL row metadata and aggregate run metrics for barcode attempts, barcode hits, barcode hit rate, OCR recoveries, parse failures, decoder misses, ambiguity, and quality rejects.
- [ ] 4.5 Update GUI/result display text so barcode-derived SN and OCR fallback are visibly different.

## 5. Tests And Release Gate

- [ ] 5.1 Add unit tests for source priority, OCR fallback accounting, non-SN barcode rejection, and ambiguous barcode rejection.
- [ ] 5.2 Add integration tests for manifest rows with missing SN crop but present label crop.
- [ ] 5.3 Run the validation command and make it fail when exact barcode-derived SN hit rate is below 90%.
- [ ] 5.4 Run `python -m unittest discover -v` and the SN barcode validation command before marking the change complete.
