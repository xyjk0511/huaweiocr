---
title: PartNo/Model Scans Silently Failed When a Scanner Locked Fresh Temp PNGs
date: 2026-11-13
category: logic-errors
module: HuaweiOCR Stage2 crop pipeline
problem_type: logic_error
component: service_object
symptoms:
  - "Stage2 PartNo crops dropped from 495 to 472 in one baseline run with unchanged Stage2 code; many S380 labels lost their PartNo."
  - "decode_raw_part_no_crop returned [] in 0.0 s; test_crop_field_padding failed intermittently."
  - "A pixel-repaired SN slipped past the PartNo cross-check because the label had no PartNo."
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - huaweiocr/detect/crop_pipeline.py
tags: [huaweiocr, stage2, part-no, windows, temp-files]
---

# PartNo/Model Scans Silently Failed When a Scanner Locked Fresh Temp PNGs

## Problem

`_decode_raw_part_no_crop_once` and `decode_model_crop` created an empty file with
`tempfile.mkstemp(suffix=".png")` in `%TEMP%`, then `save_png` wrote a second temp file and
`os.replace`d it onto the first. On this machine a background service (antivirus or indexer) started
holding freshly created `.png` files open, so every rename failed with `WinError 32`. The exception was
logged at debug level only and the decode returned `[]`, so whole batches silently lost PartNos.

## Fix

`_write_scan_candidate` writes the crop under a new name inside `tempfile.mkdtemp()` (nothing to replace
onto) and removes the directory afterwards. A write failure is now logged at `warn`. Regression test:
`test_decode_raw_part_no_crop_survives_locked_existing_temp_files` (fails on the old code).

## Lesson

Never rename onto a file that was just created empty on Windows; write to a fresh name instead. Do not
swallow I/O failures at debug level on a path that decides whether a field is recognized.
