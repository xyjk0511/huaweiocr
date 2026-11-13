---
title: Accept 980-Series Serials (S110-4P1T-BU-V2) and Bound Unreadable PartNo Decoding
date: 2026-11-13
category: logic-errors
module: HuaweiOCR Stage2 recognition
problem_type: logic_error
component: service_object
symptoms:
  - "Labels for S110-4P1T-BU-V2 (Part No. 98013170) were not recognized and the app appeared to freeze when they were in a batch."
  - "A clean SN barcode 21980131704ES6000762 was rejected, then code128 pixel repair forged a 4E26 serial and reported it as a barcode hit."
  - "Two low-resolution photos with 11 labels did not finish within 15 minutes."
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - huaweiocr/barcode/sn.py
  - huaweiocr/barcode/sn_rules.py
  - huaweiocr/core/extract.py
  - huaweiocr/core/profile.py
  - huaweiocr/detect/crop_pipeline.py
  - huaweiocr/recognition/pipeline.py
tags: [huaweiocr, stage2, sn-parsing, part-no, performance, pixel-repair]
---

# Accept 980-Series Serials and Bound Unreadable PartNo Decoding

## Problem

S110-4P1T-BU-V2 labels print `Part No. 98013170` and serials such as
`21980131704ES6000762`. Huawei 20-character serials are `"21" + PartNo + optional digit + marker + suffix`.
Every rule only accepted the `215` prefix (500-series Part Nos), so:

- The genuine SN barcode decode was discarded as a non-SN payload.
- SN scanning ran its whole fallback chain; code128 pixel repair, whose templates include `4E26########`,
  then produced a wrong serial that was counted as a barcode hit.
- OCR read the correct text, but the same regex rejected it, so every OCR pass ran.
- `98013170` was not in the PartNo model map.

Separately, when a PartNo barcode is unreadable (blur, chat-compressed photos), each decode runs the full
fallback chain (~8-10 s) and Stage2 tries up to ~8 crop variants per label, then recognition decodes the
same crop again. A handful of such labels stalled a batch for many minutes.

## Fix

- SN prefix is `21(?:5[0-9]{7,8}|980[0-9]{5,6})` in `SN20_PREFIX_PATTERN`, the OCR ES pattern, the
  `S21980` serial-field marker (stripped only where an SN actually starts), SN ranking, and `infer_standard_sn_family` (980 families get ids like
  `p11-21980-er-d7`; existing 215 ids are unchanged).
- `98013170 -> S110-4P1T-BU-V2` added to the built-in profile.
- `read_part_no_barcodes` skips duplicate scaled views (small crops cap at `PART_NO_MAX_SCALE` for every
  target width) and memoizes failed full decodes by pixel hash.
- Stage2 gives each label a PartNo decode budget (`CROP_STAGE2_PART_NO_BUDGET_SECONDS`, default 45 s, counting only time spent inside PartNo decoding);
  after it is spent, further PartNo crop variants are skipped. Readable labels decode in well under 1 s and
  never reach the budget.

## Remaining risk

Pixel repair can still forge a `4E26` serial from an SN barcode that is physically unreadable. A label-local
PartNo-to-SN-family consistency check would block this but needs the SN family of each product.
