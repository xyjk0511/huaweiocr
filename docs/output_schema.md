# 输出字段协议（Output Schema）

> 两个 JSONL 是 crop → scan2 → 消费端（GUI 表格 / Excel 导出 / 验证脚本）之间的契约。
> 字段清单提取自 2026-07-06 基线（500 行全量核对）。加字段向后兼容；改名/删字段
> 属于破坏性变更，必须同步所有消费端并更新本文档。

## 关联键

`label_id` = `<源图文件名>__label_<N>`，同一源照片的多个物理标签各占一行。
贯穿 manifest.jsonl → model_sn_ocr.jsonl → GUI/Excel。

## stage2_fields/manifest.jsonl（生产者：crop.py；消费者：scan2.py、审计工具）

| 字段 | 类型 | 含义 |
|---|---|---|
| label_id | str | 关联键 |
| label_crop | str | Stage1 标签裁剪图绝对路径 |
| original_image_path | str | 源照片路径（**仅溯源元数据**——一张源图可含多标签，禁止用于整图条码回扫，见 AGENTS.md） |
| model_path | str\|null | Model 字段裁剪图路径（null=未裁出，可能走延迟裁剪） |
| sn_path | str\|null | SN 字段裁剪图路径 |
| part_no_path | str\|null | PartNo 字段裁剪图路径 |
| model_conf / sn_conf / part_no_conf | float\|null | 各字段检测置信度 |
| part_no | str | 裁剪阶段已解出的 PartNo（可为空串） |
| part_no_codes | list[str] | PartNo 候选码 |
| stage2_rotation | int | Stage2 采用的旋转角 |
| model_crop_source / part_no_crop_source | str | 字段框来源（检测/条码区推导等） |

## stage2_fields/model_sn_ocr.jsonl（生产者：scan2.py；消费者：GUI、Excel、validate 工具）

| 字段 | 类型 | 含义 |
|---|---|---|
| label_id | str | 关联键 |
| model | str | 最终型号值（空串=未识别） |
| sn | str | 最终序列号值（空串=未识别） |
| model_src / sn_src | str | 值的来源；**指标分账依据**。已见枚举：model: `barcode` / `part_no_barcode`（纯条码，计入条码命中）/ `part_no_hint` / `part_no_ocr` / `ocr_label` / `ocr` / `barcode_visual`（型号经 OCR 读出、条码条纹仅视觉校验）/ `barcode_ocr_consensus`（条码原文经 OCR 共识确认）；sn: `barcode` / `ocr` / `barcode_ocr_consensus`。**仅纯条码来源**（`barcode` / `part_no_barcode`）计入条码命中率；含 OCR 的来源（`ocr*` / `barcode_visual` / `barcode_ocr_consensus`）计入 OCR 恢复，不得混淆 |
| model_raw / sn_raw | str | 原始证据摘要，默认脱敏（`SCAN2_UNSAFE_RAW` 控制） |
| part_no | str | 该标签的 PartNo 值 |
| part_no_src | str | PartNo 来源 |
| part_no_model_map_updated | bool | 本行是否新学到 PartNo→Model 映射 |
| sn_barcode_status | str | SN 条码结论（如 `hit` / miss 类状态） |
| sn_barcode_attempts / sn_barcode_decoded_count | int | 解码尝试/成功次数 |
| sn_barcode_sources / sn_barcode_source_regions / sn_barcode_decoder_names | list | 命中的候选来源、区域、解码器（label-local 证据审计用） |
| sn_barcode_ambiguous_sns | list[str] | 多候选歧义时的备选 SN |
| sn_barcode_failed_payloads | list[str] | 解析失败或被抑制（含已知外来码）的条码原文，默认脱敏（`mask_raw`），仅供诊断 |
| sn_problem / sn_problem_reason | str | SN 问题分类与原因说明 |

## run_summary.json（生产者：run_all.py）

`schema_version`（当前 1）、`input_dir`、`output_paths`（各阶段目录与结果文件）、
`timing_sec`（total/crop/scan）、`image_count`、`crop_stats`（含 label_count、
manifest_rows；record-keeping 增强后含 stage1_rejects）、`scan2_stats`（model_/sn_
成功与条码命中/OCR 恢复各项计数）、`exit_status`、`status`。

## Excel 导出（run_all.py --excel-out）

列：`label_id, model, sn, model_src, sn_src`（model_sn_ocr.jsonl 的子集）。
