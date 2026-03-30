# HuaweiOCR / 设备标签 OCR 流水线

![Python](https://img.shields.io/badge/Python-OCR%20Pipeline-blue)
![Focus](https://img.shields.io/badge/Focus-Barcode%20%2B%20OCR-purple)
![Output](https://img.shields.io/badge/Output-JSONL-success)
![Platform](https://img.shields.io/badge/Platform-Windows-orange)

## At a glance / 项目速览

| Item | Summary |
|------|---------|
| Task | Device-label extraction / 设备标签字段提取 |
| Pipeline | Detect → crop → barcode → OCR → export |
| Output | Structured JSONL |
| Strength | Batch processing with debugging artifacts |
| Project value | OCR engineering, CV workflow design, structured extraction |

An end-to-end OCR pipeline for device-label extraction, combining detection, multi-stage cropping, barcode decoding, OCR, and structured JSONL export.

一个端到端的设备标签 OCR 流水线，整合了目标检测、多阶段裁剪、条码识别、OCR 和结构化 JSONL 导出。

---

## Overview / 项目概述

This project is designed for practical batch processing of device-label images.

这个项目面向真实设备标签图片的批量处理，而不是单张图像的演示级 OCR。

The core idea is:

> **Locate first, then recognize.**

核心思路可以概括为：

> **先定位，再识别。**

That means the workflow first detects or crops the relevant regions, then applies barcode decoding and OCR on much cleaner sub-images.

也就是说，这个流程不是直接对整张图 OCR，而是先提取相关区域，再对更干净的小图像做条码识别和 OCR。

---

## Pipeline / 技术流程

```text
Input image
  → label detection / crop
  → field crop (model / SN)
  → barcode decode
  → OCR fallback
  → regex / postprocess
  → JSONL export
```

---

## Main capabilities / 主要能力

### 1. Multi-stage cropping / 多阶段裁剪
The system narrows the problem step by step instead of relying on whole-image OCR.

系统采用逐步缩小问题空间的方式，而不是对整张图直接 OCR。

### 2. Barcode-first strategy / 条码优先策略
Barcode decoding is attempted first, with OCR used as a fallback when decoding is unreliable.

优先尝试条码识别，OCR 作为兜底，提高结构化字段提取成功率。

### 3. Structured export / 结构化输出
Results are exported in JSONL format, making the pipeline easier to integrate into downstream processing.

输出为 JSONL 格式，更适合后续自动化流程接入。

### 4. Debuggability / 可调试性
The project saves intermediate crops, logs, and failure artifacts to make iteration easier.

项目会保留中间裁剪结果、日志和失败样本，便于后续回溯与优化。

---

## Why this project matters / 为什么这个项目重要

This repository is one of the strongest engineering-oriented projects in the portfolio because it demonstrates:

- computer vision + OCR workflow design
- batch processing for real-world inputs
- structured data extraction rather than just text recognition
- practical robustness strategies for noisy images

这个仓库是你偏工程方向里很强的项目之一，因为它体现了：
- 计算机视觉 + OCR 的完整流程设计
- 面向真实输入的批处理能力
- 不只是“识别文本”，而是“提取结构化字段”
- 针对噪声图像的实用鲁棒性策略

---

## Repository Structure / 仓库结构

```text
crop.py         # stage-1 / stage-2 cropping logic
scan2.py        # barcode + OCR + structured output
barcode.py      # barcode enhancement pipeline
run_all.py      # main pipeline entry
start.bat       # Windows one-click launcher
ocr.py          # OCR helpers
gui_app.py      # GUI entrypoints
```

---

## Quick Start / 快速开始

### Windows one-click
1. Create `.env` in the project root:
```bash
API_KEY=your_api_key_here
```
2. Double-click `start.bat`

### CLI
```bash
python run_all.py --input ./images --out ./out --format jsonl --log-level info --device cpu
```

---

## Output / 输出结果

Typical outputs include:
- `stage1_labels/`
- `stage2_fields/model/`
- `stage2_fields/sn/`
- `model_sn_ocr.jsonl`
- `debug_ocr_barcode.log`

Example JSONL line:
```json
{"label_id":"img_001__label_1","model":"S380-S8P2T","sn":"4E25XXXXXXXX","model_src":"barcode","sn_src":"ocr"}
```

---

## Robustness strategies / 鲁棒性策略

- multi-scale upscaling for small barcodes
- ROI cropping to reduce noise
- rotation attempts (0 / 90 / 180 / 270)
- regex validation for structured fields
- failure sample logging for iterative improvement

- 多尺度放大
- ROI 截取降噪
- 多方向旋转尝试
- 正则约束结构化字段
- 失败样本记录与回看

---

## Security / 安全说明

- API keys are stored in `.env`, not in source code
- `.env` is excluded from the repository
- keys can be rotated without code changes

- API key 存在 `.env` 中，而不是硬编码在代码里
- `.env` 不提交到仓库
- 更换 key 不需要改代码

---

## Future improvements / 后续改进方向

- better visual demos and screenshots
- cleaner benchmark numbers on real device-label datasets
- stronger resumable batch-processing support
- configurable export schemas

如果继续完善，这个项目最值得补的是：
- 更直观的演示图和截图
- 在真实数据集上的量化指标
- 更好的断点续跑能力
- 更可配置的导出结构


---

## Overview / 项目概述

This project is designed for practical batch processing of device-label images.

这个项目面向真实设备标签图片的批量处理，而不是单张图像的演示级 OCR。

The core idea is:

> **Locate first, then recognize.**

核心思路可以概括为：

> **先定位，再识别。**

That means the workflow first detects or crops the relevant regions, then applies barcode decoding and OCR on much cleaner sub-images.

也就是说，这个流程不是直接对整张图 OCR，而是先提取相关区域，再对更干净的小图像做条码识别和 OCR。

---

## Pipeline / 技术流程

```text
Input image
  → label detection / crop
  → field crop (model / SN)
  → barcode decode
  → OCR fallback
  → regex / postprocess
  → JSONL export
```

---

## Main capabilities / 主要能力

### 1. Multi-stage cropping / 多阶段裁剪
The system narrows the problem step by step instead of relying on whole-image OCR.

系统采用逐步缩小问题空间的方式，而不是对整张图直接 OCR。

### 2. Barcode-first strategy / 条码优先策略
Barcode decoding is attempted first, with OCR used as a fallback when decoding is unreliable.

优先尝试条码识别，OCR 作为兜底，提高结构化字段提取成功率。

### 3. Structured export / 结构化输出
Results are exported in JSONL format, making the pipeline easier to integrate into downstream processing.

输出为 JSONL 格式，更适合后续自动化流程接入。

### 4. Debuggability / 可调试性
The project saves intermediate crops, logs, and failure artifacts to make iteration easier.

项目会保留中间裁剪结果、日志和失败样本，便于后续回溯与优化。

---

## Why this project matters / 为什么这个项目重要

This repository is one of the strongest engineering-oriented projects in the portfolio because it demonstrates:

- computer vision + OCR workflow design
- batch processing for real-world inputs
- structured data extraction rather than just text recognition
- practical robustness strategies for noisy images

这个仓库是你偏工程方向里很强的项目之一，因为它体现了：
- 计算机视觉 + OCR 的完整流程设计
- 面向真实输入的批处理能力
- 不只是“识别文本”，而是“提取结构化字段”
- 针对噪声图像的实用鲁棒性策略

---

## Repository Structure / 仓库结构

```text
crop.py         # stage-1 / stage-2 cropping logic
scan2.py        # barcode + OCR + structured output
barcode.py      # barcode enhancement pipeline
run_all.py      # main pipeline entry
start.bat       # Windows one-click launcher
ocr.py          # OCR helpers
gui_app.py      # GUI entrypoints
```

---

## Quick Start / 快速开始

### Windows one-click
1. Create `.env` in the project root:
```bash
API_KEY=your_api_key_here
```
2. Double-click `start.bat`

### CLI
```bash
python run_all.py --input ./images --out ./out --format jsonl --log-level info --device cpu
```

---

## Output / 输出结果

Typical outputs include:
- `stage1_labels/`
- `stage2_fields/model/`
- `stage2_fields/sn/`
- `model_sn_ocr.jsonl`
- `debug_ocr_barcode.log`

Example JSONL line:
```json
{"label_id":"img_001__label_1","model":"S380-S8P2T","sn":"4E25XXXXXXXX","model_src":"barcode","sn_src":"ocr"}
```

---

## Robustness strategies / 鲁棒性策略

- multi-scale upscaling for small barcodes
- ROI cropping to reduce noise
- rotation attempts (0 / 90 / 180 / 270)
- regex validation for structured fields
- failure sample logging for iterative improvement

- 多尺度放大
- ROI 截取降噪
- 多方向旋转尝试
- 正则约束结构化字段
- 失败样本记录与回看

---

## Security / 安全说明

- API keys are stored in `.env`, not in source code
- `.env` is excluded from the repository
- keys can be rotated without code changes

- API key 存在 `.env` 中，而不是硬编码在代码里
- `.env` 不提交到仓库
- 更换 key 不需要改代码

---

## Future improvements / 后续改进方向

- better visual demos and screenshots
- cleaner benchmark numbers on real device-label datasets
- stronger resumable batch-processing support
- configurable export schemas

如果继续完善，这个项目最值得补的是：
- 更直观的演示图和截图
- 在真实数据集上的量化指标
- 更好的断点续跑能力
- 更可配置的导出结构
