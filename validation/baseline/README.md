# 指标基线（Metrics Baseline）

重构 crop.py / scan2.py 前后必须用这套基线做行为一致性验证。

## 样本与参照

- **输入样本**：`batch_runs/baseline_input/`（115 张源照片，复制自
  `dist/HuaweiOCR/new_images/gui_run_20260706_141456_260961/`；dist 会被打包重建覆盖，
  以该副本为准）。
- **参照结果**：`old_dist_run/`（旧版打包应用 2026-07-06 GUI 运行产出）：
  - `model_sn_ocr.jsonl` — 500 行识别结果（逐 label 对比的金标准）
  - `manifest.jsonl` — 500 行 Stage2 字段清单
  - `source_manifest.jsonl` — 源图 SHA256 清单
- `run_summary.20260706_source_rerun.json` — 当前源码同日重跑的 run_summary，
  与参照结果对比：**500/500 label 完全对齐，model/sn 值 0 差异**。

## 自动化命令（推荐）

`tools/check_baseline_regression.py` 把下面的验证步骤 + 判定标准都自动化了：

```powershell
.venv\Scripts\python.exe tools\check_baseline_regression.py
```

默认输入 `batch_runs/baseline_input`、输出 `batch_runs/baseline_check_auto`、基线
`validation/baseline/old_dist_run/model_sn_ocr.jsonl`，跑完打印通过/失败报告并以
退出码 0/1 收尾（可接入脚本判断）。已自动限流 `CROP_WORKERS=2` / `SCAN2_WORKERS=2`
（可用同名环境变量覆盖）。

已有一次跑批结果、只想重新比对时用 `--skip-run --out <已有输出目录>`，不再重跑管线：

```powershell
.venv\Scripts\python.exe tools\check_baseline_regression.py --skip-run --out batch_runs\baseline_check_final
```

对应的 `tests/test_baseline_regression.py` 里有：
- 常规单测（比较逻辑本身，无需跑图，纳入 `python -m unittest discover -s tests` / CI）；
- 一个 opt-in 的全量端到端测试，默认跳过，需要设置
  `HUAWEIOCR_RUN_BASELINE_REGRESSION=1` 才会真的跑 115 张图（同时要求
  `batch_runs/baseline_input` 已就位，否则清晰跳过）。

## 手动验证步骤（了解自动化命令具体做了什么，或需要调试时用）

```powershell
.venv\Scripts\python.exe run_all.py --input batch_runs/baseline_input --out batch_runs/baseline_check_<日期> --log-level info
```

然后逐 `label_id` 对比 `stage2_fields/model_sn_ocr.jsonl` 与 `old_dist_run/model_sn_ocr.jsonl`。

## 判定标准

1. **必须逐条相等（硬门槛）**：label_id 集合（500 个）、每条的 `model` 与 `sn` 值。
2. **允许极小漂移（软门槛）**：`sn_src` / `model_src` 的 barcode↔ocr 来源分布。
   条码 CLI 子进程有超时（`BARCODE_CLI_TIMEOUT_SECONDS`），机器负载高时个别条码
   尝试会超时并由 OCR 兜底得到相同值。2026-07-06 验证中出现 2 例（496/4 → 494/6），
   值完全一致。若来源漂移超过 ±5 条，或伴随任何值差异，视为回归，须在空闲机器
   上以正常优先级复跑确认。
3. 跑批时建议限流（`CROP_WORKERS=2` / `SCAN2_WORKERS=2`）或将进程优先级设为 Idle，
   避免占满整机。

## 基线运行时统计（参考，非门槛）

- images: 115 → labels: 500
- model_success: 500/500；sn_success: 500/500
- sn_barcode_hit_rate ≈ 0.988；model 主要来源为 part_no_barcode（492/500）
