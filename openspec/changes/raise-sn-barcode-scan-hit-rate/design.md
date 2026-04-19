## Context

The current SN path is barcode-first in intent, but practical decoding is limited by small SN crops, low-resolution multi-label photos, and decoder attempts that do not always cover the full label crop. Recent validation on three uploaded photos produced OCR fallback values while direct barcode decoding returned no barcode values.

The target is not simply "return an SN"; it is "return an SN from a decoded barcode" with at least 90% exact hit rate on a representative accepted-quality validation set. OCR can still help operations, but it must not hide barcode failures.

## Goals / Non-Goals

**Goals:**
- Achieve at least 90% exact SN barcode hit rate on accepted-quality validation images.
- Prefer barcode-derived SN values from every available visual source: SN crop, full label crop, and candidate barcode regions.
- Report barcode success, OCR fallback, parse failure, decoder failure, and image-quality rejection separately.
- Add deterministic validation so the threshold can be checked before release.
- Keep existing JSONL pipeline and GUI workflow compatible while adding more explicit metadata.

**Non-Goals:**
- Guarantee barcode decoding on photos where the SN barcode is too small, clipped, blurred, occluded, or lacks quiet zones.
- Count OCR-derived SN values as barcode hits.
- Replace Roboflow label detection in this change.
- Add a network dependency for barcode decoding.

## Decisions

1. Define a validation contract before tuning decoders.

   The 90% threshold will be measured on a ground-truth JSONL dataset with one row per expected SN barcode. Each row records image path, expected SN, source label id when available, and whether the image is accepted for barcode evaluation. Low-quality images are reported separately as quality rejects; they are not silently treated as successful OCR cases.

   Alternative considered: use ad hoc manual spot checks. Rejected because it cannot prove a 90% target.

2. Add a barcode candidate scanner before OCR.

   SN recognition will aggregate barcode candidates from:
   - the detected SN crop,
   - the full label crop from manifest,
   - barcode-band candidates found by 1D stripe morphology on the label crop,
   - rotated and upscaled versions with quiet-zone padding,
   - existing local decoders through a common adapter.

   Alternative considered: only increase padding around the SN crop. Rejected because current failures include missing SN crops and tiny crop geometry.

3. Keep decoder adapters local and swappable.

   The first implementation should use existing bundled decoders (`pyzbar` and `BarcodeReaderCLI`) with better candidate generation and CLI parameters. If the validation report remains below 90%, add a second local adapter behind the same interface rather than mixing decoder-specific logic into `scan2.py`.

   Alternative considered: make OCR the primary parser and call that "hit rate." Rejected because the requirement is barcode hit rate.

4. Separate barcode extraction from SN selection.

   Barcode decoding returns raw values with source metadata. SN selection parses raw barcode text with existing SN rules, de-duplicates candidates, rejects ambiguous multi-SN conflicts, and ranks sources in this order: SN crop, barcode-region crop, label crop, original image fallback.

   Alternative considered: return the first decoded string. Rejected because full label scans can include logistics or EAN codes that are not device SNs.

5. Make metrics first-class outputs.

   The pipeline will emit aggregate fields for barcode attempts, barcode hits, barcode parse failures, decoder misses, OCR fallbacks, and quality rejects. JSONL row metadata will preserve `sn_src=barcode` only when the final SN came from barcode parsing.

   Alternative considered: infer metrics from `sn_src` after the run. Rejected because failure causes need to be visible for tuning.

## Risks / Trade-offs

- Low-resolution photos may cap barcode hit rate below 90% -> Add quality-gate reporting and require a representative accepted-quality dataset for the threshold.
- More candidate crops increase runtime -> Add attempt budgets, early exit on exact SN hit, and debug-only candidate image dumps.
- Full-label scans may decode logistics barcodes instead of SN -> Parse and rank only SN-like barcode payloads; keep non-SN barcode values as diagnostics.
- Decoder-specific behavior can create fragile code -> Hide decoders behind a small adapter result schema.
- The validation set may be too small or biased -> Require enough accepted samples before claiming the 90% target and store the manifest with expected SNs.

## Migration Plan

1. Add the validation dataset manifest and evaluation command.
2. Implement barcode candidate generation and decoder adapters.
3. Integrate candidate scanning into `scan2.recognize_sn` before OCR fallback.
4. Update JSONL metrics and GUI display labels for barcode/OCR provenance.
5. Run unit tests and validation; do not claim the change complete until barcode hit rate is at least 90%.

Rollback is limited to reverting the change commit because output schema additions must remain backward-compatible.

## Open Questions

- How many real accepted-quality labeled SN barcode samples should be required for the release gate? Initial target: at least 50.
- Should low-quality captures block the run, or should they produce a separate "retake required" result in the GUI?
