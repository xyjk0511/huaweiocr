---
title: Make Model Recognition Local and Barcode-First
date: 2026-06-21
category: logic-errors
module: HuaweiOCR Stage2 model recognition
problem_type: logic_error
component: service_object
symptoms:
  - "Model crops and model recognition were unreliable enough that the selected Stage1 label crops needed local-only validation."
  - "The default detector path could depend on external Roboflow/API-key behavior instead of local detector artifacts."
  - "Model recognition could enter OCR too early instead of treating a legal model barcode as terminal."
  - "Barcode hits and OCR fallback recoveries needed separate accounting: the selected 35-label check passed with 26 barcode hits and 9 OCR recoveries."
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - local_yolo.py
  - scan2.py
  - barcode.py
  - tests/test_local_yolo_inference.py
  - tests/test_sn_barcode_scanning.py
  - tests/test_locked_output_dirs.py
tags: [huaweiocr, stage2, model-recognition, model-crop, local-yolo, barcode-first, ocr-fallback, label-local]
---

# Make Model Recognition Local and Barcode-First

## Problem

Stage2 model recognition was not stable enough on the supplied label set. The fix needed to avoid external training or cloud inference assumptions, use the provided/local detector data, and make model recognition follow the same evidence order every time: use a model barcode when it decodes, then fall back to label-local OCR or PartNo evidence only when barcode decoding misses.

This was not a request to tune one bad crop. The user explicitly wanted a generic rule for the current label shapes, with rotation handling left for a later pass.

## Symptoms

- Stage1 label crops were available, but Stage2 model crops or recognition results were unreliable.
- Several model crops visibly contained a readable model or barcode, yet the old path could still treat the field as missed.
- OCR could over-read unrelated text such as MAC values, "Made in China", or malformed `Model` text as a model candidate.
- Reporting needed to distinguish true model barcode hits from OCR recoveries; otherwise the barcode-first metric would be inflated.
- In the selected 35-label verification set, 9 images needed OCR fallback and were copied to `test_runs/stage1_afterfix_model_rule_selected35_20260620_v4/model_ocr_9` for review.

## What Didn't Work

- Keeping Roboflow/API-key inference as the default detector path was not acceptable for this local Windows workflow. The default crop path needs to run from local detector artifacts.
- Treating OCR as the main model path increased noise and made barcode coverage impossible to measure.
- Hard-coding around one image or one crop geometry would not satisfy the "generic enough" requirement.
- Counting OCR recovery as a barcode hit was rejected. It hides whether barcode scanning actually works.
- Whole-source-photo SN barcode fallback remains rejected for this project because one source photo can contain multiple physical labels.

## Solution

The model recognition path now has a local detector and an explicit barcode-first decision chain.

`local_yolo.py` adds a local ONNX Runtime client with Roboflow-compatible output:

```python
DEFAULT_MODEL_SPECS = {
    "huawei-2ha7t/7": ModelSpec(
        path=os.path.join(REPO_DIR, "local_models", "detectors", "label_detector.onnx"),
        names=("huawei_label",),
    ),
    "sn_model/9": ModelSpec(
        path=os.path.join(REPO_DIR, "local_models", "detectors", "field_detector.onnx"),
        names=("model", "partno", "sn"),
    ),
}
```

`scan2.py` separates model barcode recognition from OCR fallback:

```python
def recognize_model(model_path: str, label_id: str = "", use_barcode: bool = False):
    if use_barcode:
        model_code, raw, source = recognize_model_barcode(model_path, label_id=label_id)
        if model_code:
            return model_code, raw, source
    return recognize_model_ocr(
        model_path,
        label_id=label_id,
        verify_barcode_visual=use_barcode,
    )
```

The main Stage2 flow uses that split rather than a single blended recognition result:

- Scan PartNo and SN barcode evidence first when present.
- Try model barcode jobs before model OCR jobs.
- If model barcode misses, try label-local PartNo evidence, a delayed model crop from the Stage1 label crop, and finally OCR fallback when configured.
- Record the source separately as `barcode`, `part_no_*`, `ocr_*`, or `barcode_visual`.
- Keep `model_barcode_hits`, `model_part_no_hits`, and `model_ocr_recoveries` as separate counters.

The committed base fix is `a088b6d75c2deddca3a86cba70ba0e0dac38998a` (`Make model recognition local and barcode-first`). Later commits and current working-tree edits build on that same recognition boundary.

## Why This Works

The stable signal for model is the label-local model barcode. When it decodes to a plausible model, there is no reason to expose the result to OCR noise. When it does not decode, the fallback remains inside the same Stage1 label or Stage2 field crops, so recognition does not borrow evidence from another physical label.

The local ONNX detector removes network, API-key, and external service drift from the default crop path. The source counters make the result auditable: a passing model value is not enough; the pipeline records whether it came from barcode, PartNo mapping, or OCR.

The selected 35-label verification reflected the intended split: 35/35 model values passed, with 26 barcode hits and 9 OCR fallback recoveries.

## Prevention

- Keep model recognition barcode-first by default; disable `SCAN2_MODEL_BARCODE` only for targeted debugging.
- Preserve source-specific metrics whenever model recognition changes: `model_total`, `model_success`, `model_barcode_hits`, `model_part_no_hits`, `model_ocr_recoveries`, and `model_deferred_crops`.
- Add regression coverage when introducing new fallback order. The current coverage includes local detector defaults, model barcode-first behavior, PartNo-before-OCR behavior, delayed model crop fallback, PartNo model-map learning, and OCR garbage rejection.
- Save fallback review images, as with `model_ocr_9`, so OCR recoveries can be inspected without mixing them into barcode hit-rate evidence.
- Package builds need their own check: local ONNX files, PaddleOCR runtime files, and barcode decoder runtime files must all be bundled before a release is considered packaged.

## Related Issues

- `docs/solutions/logic-errors/recognize-lower-model-barcodes-agq-serials.md` covers a related but narrower Stage2 barcode parsing fix.
- `CONCEPTS.md` defines Stage1 label crops, Stage2 field crops, Model fields, PartNo fields, barcode-first recognition, and label-local evidence.
- `openspec/changes/raise-sn-barcode-scan-hit-rate/design.md` records the broader barcode-first and label-local boundary.
- `openspec/changes/raise-sn-barcode-scan-hit-rate/specs/sn-barcode-scanning/spec.md` records the SN scanning contract.

## Verification

Recorded verification for the fix:

- `python -m unittest tests.test_local_yolo_inference` -> 4 tests passed.
- `python -m unittest tests.test_sn_barcode_scanning` -> 45 tests passed.
- `python -m unittest tests.test_locked_output_dirs` -> 54 tests passed.
- `python -m unittest discover -s tests` -> 160 tests passed.
- Selected 35-label model check: 35/35 passed, with 26 barcode hits and 9 OCR fallback recoveries.

Not tested for this fix: PyInstaller package build.
