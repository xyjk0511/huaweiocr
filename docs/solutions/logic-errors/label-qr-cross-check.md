---
title: A Pixel-Repaired SN Matched the PartNo but Was Still Wrong
date: 2026-11-13
category: logic-errors
module: HuaweiOCR recognition (scan2) SN/PartNo cross-check
problem_type: logic_error
component: service_object
symptoms:
  - "Zhangjiakou label_4 printed 21500871494ERB017591 but the pipeline output 21500871493ERB017592 (source: code128_pixel_repair)."
  - "The frozen baseline carried the same wrong value, so regression checks never flagged it."
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - huaweiocr/barcode/label_codes.py
  - huaweiocr/core/part_no_products.py
  - huaweiocr/recognition/pipeline.py
tags: [huaweiocr, sn, qr-code, iso-15434, ean, cross-check]
---

# A Pixel-Repaired SN Matched the PartNo but Was Still Wrong

## Problem

Code128 pixel repair can produce a structurally valid SN that embeds the
correct PartNo but has wrong serial digits. The PartNo cross-check
(`repaired_sn_conflict`) cannot see that.

## Evidence on the label

Huawei labels carry a QR code in ISO/IEC 15434 format 06:
`[)>` RS `06` GS `1P<PartNo>` GS `18V<vendor>` GS `S<SN>` RS EOT. It has
Reed-Solomon error correction. On the 115-image baseline (500 labels) the QR
was read on 419 label crops (363 plain, 56 more after 1.5–3x upscaling); 418
matched the pipeline SN and PartNo, and the one mismatch was the forged repair
above. EAN-13 (and UPC-A on APs) codes map one-to-one to PartNos.

## Fix

`_cross_check_with_label_codes` reads only the Stage1 label crop (label-local;
never the source photo, which also holds courier QR codes):

- Weak SN (pixel repair, or none yet): a QR SN whose PartNo matches the label
  replaces it (`sn_src = barcode_label_qr`, counted as a barcode hit and in
  `sn_label_qr_rescues`) and cancels the OCR fallback job.
- Genuine decode that disagrees with the QR: value kept,
  `sn_src = barcode_qr_conflict`, `review_reason` set, GUI lists it for review.
- With a known label PartNo only a QR carrying that same PartNo counts; a QR
  with another or no PartNo is ignored (crop caught a neighbour label).
- SNs are unique, so a QR SN that another label in the batch also reads, or
  genuinely decoded, never rescues or flags this label (same-SKU neighbour).
- EAN/UPC codes are learned per PartNo (`part_no_products.json`) only from
  PartNos read from the label's own barcodes. A code conflicts when one other
  PartNo has seen it ≥2 times and ≥3× as often as this PartNo, judged against
  persisted plus batch evidence minus the label itself
  (`model_src = part_no_product_conflict`); conflicting labels are not learned.
- Learning stores that exist but cannot be parsed are left untouched instead
  of being overwritten with the current batch only.

`SCAN2_LABEL_CODE_CHECK=0` disables the check.

## Pixel repairs without a QR read

An independent OCR pass over the 115-image baseline found four more wrong
pixel repairs on labels whose QR was unreadable (checked by eye):

- The default repair template fixed the 4E prefix to `4E26`, so 2025-dated
  serials (`4E25A0068800`, `4E25C0085603`, `4E25C0085582`) were forged into
  `4E26...` values. The template is now `4E2#*#######` (year digit, month
  1-9/A-C), which repairs all three correctly.
- `21500871494ES4051146` was repaired as `...145`; Code128 has only a mod-103
  check character, and the top two candidates were 0.007 apart.

Every repaired SN that the label QR did not confirm is now read again by OCR
(`SCAN2_REPAIR_OCR_CHECK=0` disables it): agreement marks it
`sn_repair_check = ocr_confirmed`; a disagreement is always sent to review
(`sn_repair_ocr_mismatch:<other>`). OCR misreads too (Zhangjiakou label_5
`4E25A` read as `4E26A`), so the two are fused per character: where every
repair candidate within 0.02 of the winning score agrees, the repair's
character stands; elsewhere OCR's character is used (`sn_src =
ocr_repair_conflict`, counted as an OCR recovery, not a barcode hit). A
difference made only of OCR-confusable characters (`ERB` read as `ER8`) or a
fused value contradicting the label PartNo keeps the repair. A wider window
(0.03) already flipped the `ERB` case. No OCR read marks it
`sn_repair_unverified`.

## QR first (speed)

With QR retries, 452 of the 500 baseline label crops yield a QR (394 plain,
58 after upscaling); every one carried the verified SN and PartNo, and no crop
held a neighbour's QR. The misses are crops that cut the QR off or QR codes
too small and blurred to decode (stronger upscales/thresholds rescued 6 of 48
at ~1.5 s per label, so they were not adopted). So the QR is now read first:

- Stage2 (`CROP_STAGE2_LABEL_QR=0` disables): a sole QR whose PartNo maps to a
  known model lets PartNo crops skip the slow decode chain (only the quick
  zxing read runs); when no crop decodes, the QR PartNo is used
  (`part_no_source = label_qr` in the manifest).
- Recognition (`SCAN2_LABEL_QR_FIRST=0` disables): a label with exactly one QR
  payload, whose PartNo agrees with the Stage2 PartNo and maps to a known
  model, and whose SN no other crop of the same photo reads, takes SN and
  PartNo from the QR and skips SN/PartNo barcode scans
  (`label_qr_check = qr_first`, counted in `sn_label_qr_first` and as a
  barcode hit). Codes are read once and reused by the cross-check.
- Stage1 preview PNGs are written in the background.

## Lesson

A check that only validates structure (embedded PartNo) cannot catch digit
errors inside the serial part. Use an independent, error-corrected encoding of
the same value when the label has one.
