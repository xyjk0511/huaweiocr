---
title: OCR Fallback Paid a 75 s Cold Start on Every Run
date: 2026-11-13
category: performance
module: HuaweiOCR recognition (scan2) OCR engine
problem_type: performance
component: service_object
symptoms:
  - "A 15-image run spent 101 s in scan2 although barcode decoding finished in seconds."
  - "py-spy showed ~25% of busy time inside _load_ocr_module (import ocr)."
root_cause: config_error
resolution_type: code_fix
severity: medium
related_components:
  - huaweiocr/detect/ocr_engine.py
  - huaweiocr/cli.py
  - huaweiocr/ui/pipeline.py
tags: [huaweiocr, ocr, paddleocr, paddlex, startup, prewarm]
---

# OCR Fallback Paid a 75 s Cold Start on Every Run

## Problem

`import ocr` took 72–83 s in the dev environment. About 48 s came from
`paddlex.__init__` → `apply_langchain_shim()` → `langchain_text_splitters` →
`transformers`, which only PP-ChatOCR pipelines need. The rest is paddle itself.
The load happened on the first OCR fallback inside scan2, after cropping had
finished, so it added directly to wall time. The GUI prewarm thread that should
hide this was off by default: it read `SCAN2_OCR_FALLBACK` with default `False`,
while scan2 defaults the fallback to `True` (changed in 72b3784). The CLI had no
prewarm at all.

## Fix

- `ensure_langchain_import_compat()` (ocr_engine.py) fills the
  `langchain.docstore.document.Document` and
  `langchain.text_splitter.RecursiveCharacterTextSplitter` slots with lazy
  stand-ins before paddle/paddleocr import. They import the real class on first
  instantiation. It does nothing without LangChain and does not shadow a
  LangChain version that still ships those modules. `import ocr`: ~75 s → ~27–33 s.
- GUI prewarm default now matches scan2 (`True`).
- The CLI starts a daemon prewarm thread right after importing scan2, so OCR
  loads while Stage1/Stage2 cropping runs. scan2 waits on the same engine lock.
  `HUAWEIOCR_PREWARM_OCR=0` disables it.

15-image subset: 151 s → 80 s total (scan 101 s → 12 s), identical results.
The packaged exe excludes LangChain/transformers, so it only gains from prewarm.

## Lesson

Optional third-party integrations imported at package import time can dominate
startup. Measure with `python -X importtime -c "import ocr"` before guessing.
Keep prewarm defaults tied to the same flag default the pipeline uses.
