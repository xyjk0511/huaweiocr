---
title: Recognize Lower Model Barcodes and AGQ Serials
date: 2026-06-21
category: logic-errors
module: HuaweiOCR Stage2 recognition
problem_type: logic_error
component: service_object
symptoms:
  - "Model barcode fallback could miss the lower model barcode after seeing a numeric barcode first."
  - "SN decoding could return a valid AGQ serial payload but still report barcode_parse_fail."
  - "A private outbound-label verification run needed label-local barcode fixes to reach 97/97 SN barcode hits."
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - scan2.py
  - sn_barcode.py
  - tests/test_sn_barcode_scanning.py
tags: [huaweiocr, stage2, barcode-first, model-barcode, sn-parsing, agq-serials]
---

# Recognize Lower Model Barcodes and AGQ Serials

## Problem

Stage2 barcode-first recognition had two narrow logic gaps. A model crop could expose multiple 1D barcodes and the first decoded payload might be a numeric PartNo-like code instead of the lower model barcode. Separately, SN barcode decoding could successfully read a synthetic AGQC-style payload such as `2000000000AGQC000000`, but the parser rejected that serial family because the SN20 rule only accepted `AGQA`.

Both fixes had to preserve the project boundary: SN evidence stays label-local, and OCR fallback must not inflate barcode hit metrics.

## Symptoms

- `sample_label_001.png__label_1` produced `sn_src="barcode_parse_fail"` even though `sn_raw` contained repeated synthetic AGQC-style payloads.
- The SN crop visually contained the `S/N:` text and the complete barcode, so the failure was not a crop-boundary miss.
- A model crop could decode a numeric payload first, then fail to continue to the lower model barcode.
- After the fix, `test_runs/<sanitized-run>/summary_counts.json` reported 97 rows, 97 SN barcode hits, and 0 SN failures.

## What Didn't Work

- Continuing to adjust Stage2 SN crops was not the main closure path once review images showed complete SN text and barcode. Earlier session review found several failures were parser or payload-selection issues, not strong geometry failures. (session history)
- Restoring whole-source-photo SN fallback was rejected. A source photo can contain multiple labels, so SN barcode evidence must remain limited to the SN crop, label crop, or label-local candidate regions. (session history)
- Adding OCR recovery to the barcode metric was rejected because it would hide barcode parser failures and corrupt hit-rate reporting. (session history)
- Relying only on `decode_cli_sharp_variants()` for model bands was insufficient because a sharp pass can miss while the lower or middle band is still decodable by the small-patch decoder.

## Solution

The model path now keeps scanning label-local barcode bands when the first decoded payload is not a plausible model. `_scan_model_barcode_band_entries()` collects detected and middle barcode bands, runs the sharp decoder when available, then runs `decode_small_patch()` on the same bands as a second pass.

```python
for source, band in bands:
    if len(band.shape) == 2:
        band_bgr = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
    else:
        band_bgr = band
    info = decode_small_patch(band_bgr)
    results = info.get("results", []) if isinstance(info, dict) else []
```

The SN parser now accepts the observed AGQ serial family without making the rule open-ended. `SN20_BODY_PATTERN` changed from fixed `AGQA` to `AGQ[A-Z]`, while keeping the surrounding Huawei serial shape strict.

```python
SN20_BODY_PATTERN = r"2[0-9]{9,10}(?:ER[A-Z]?|LDR[A-Z]?|LDS|SRA|AGQ[A-Z])[0-9]{6,7}"
```

The regression coverage is in `tests/test_sn_barcode_scanning.py`:

- `test_agqa_series_payload_is_accepted_by_sn_rules` covers both `AGQA` and `AGQC`.
- `test_model_barcode_band_second_scan_can_find_lower_model_code` simulates a direct scan returning a numeric payload first, a sharp-band miss, and a second-pass small-patch hit of `AP162E`.

The committed fix is `4a9e79e` (`Recognize lower model barcodes and AGQ serials`).

## Why This Works

The model issue was not that the crop lacked a model barcode. It was that the recognition path could stop too early after seeing a non-model barcode or after a band decoder miss. The second pass gives the lower model barcode another label-local decode opportunity without falling back to OCR or scanning the original photo.

The SN issue was not that the decoder failed. `select_sn_from_decoder_results()` had decoded payloads, but `extract_sn_from_payload()` produced no legal SN candidate because `AGQC` was outside the accepted pattern. Allowing `AGQ[A-Z]` turns the decoded payload into a valid candidate while still rejecting unrelated PartNo, EAN, MAC, and pure-digit payloads.

## Prevention

- Add parser tests whenever a new observed SN family appears in decoded payloads; do not treat a parse failure as a decoder miss.
- Keep true-image verification alongside unit tests. The closing evidence for this fix was `run_all.py` on a private outbound-label dataset, producing 97/97 SN barcode hits.
- Keep SN barcode metrics barcode-only. OCR fallback can recover values, but it must remain a separate source count.
- Keep SN source boundaries label-local: SN crop, label crop, and label-local candidate regions only.
- For model recognition, do not stop at a numeric barcode payload when the crop can contain a lower model barcode.

## Related Issues

- `openspec/changes/raise-sn-barcode-scan-hit-rate/design.md` documents the broader barcode-first and label-local SN boundary.
- `openspec/changes/raise-sn-barcode-scan-hit-rate/specs/sn-barcode-scanning/spec.md` documents the SN scanning contract.
- `AGENTS.md` records the project rule that SN evidence must not come from whole-source-photo fallback and that OCR fallback must not inflate barcode hit rate.
