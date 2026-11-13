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
  - huaweiocr/core/part_no_sn_kinds.py
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

## Follow-up: generic PartNo/SN rules with a label-local cross-check (2026-11-13)

Hard-coded prefixes meant every new product needed a code change, so the rules are now structural:

- SN prefix is `21[0-9]{8,9}` (any 8-digit PartNo). `SS` stays bound to its two known prefixes because a
  structurally valid false `SS` decode was seen before.
- PartNo: the 500/980 series are matched as before; any other 8-digit value is accepted only as a standalone
  token that is not a date (`YYYYMMDD`) or a repeated digit (`extract_part_numbers_from_text`). PartNo pixel
  repair and stripe rescue still require the 500/980 series, so they cannot invent a new PartNo.
- `huaweiocr/core/part_no_sn_kinds.py` + `_cross_check_sn_with_part_no` in the recognition pipeline:
  genuine (non-pixel-repair) SN decodes record which SN family ("21"+PartNo or "4E") each PartNo uses, in
  the batch and in `part_no_sn_kinds.json` in the user data dir (`SCAN2_PART_NO_SN_KINDS_PATH` overrides).
  A pixel-repaired SN is rejected and sent to OCR when its embedded PartNo differs from the label PartNo,
  or when it is a 4E serial and that PartNo has only been seen with "21" serials. Rejections are counted in
  `sn_barcode_part_no_conflicts`, not as barcode hits. A genuine "21" SN also supplies the PartNo when the
  PartNo crop is unreadable.

Evidence: in the frozen baseline all 377 genuine "21" SNs embed the label PartNo; the 6 conflicting values
were all pixel repairs (three `4E26...` on AP162E labels, three `215...ER...` with the wrong PartNo).

## Remaining risk

- A pixel-repaired 4E serial is only caught once a genuine decode of the same PartNo has been seen (in the
  batch or earlier). The first unreadable label of a new product can still get a forged 4E value.
- With no label PartNo, a structurally valid SN of any "21" prefix is accepted without a cross-check.
