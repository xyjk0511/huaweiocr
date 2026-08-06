# HuaweiOCR

面向 Windows 的设备标签批量 OCR 流水线。

## 做什么

流程会检测标签区域，裁剪 model/PartNo/SN 字段，优先识别标签内条码，再用 OCR 兜底，最后输出结构化 JSONL。

SN 条码识别只使用当前标签绑定的裁剪图，例如 SN 小图、条码候选区和标签小图；整张原始照片只保留为来源元数据，不作为 SN 条码兜底扫描源，避免一张照片里多个标签互相串号。

检测步骤默认使用本地 ONNX 模型：`local_models/detectors/label_detector.onnx` 负责 stage1 标签裁剪，`local_models/detectors/field_detector.onnx` 负责 model/PartNo/SN 字段裁剪。正常本地运行不需要 Roboflow API key。

## 环境

- Windows
- 推荐 Python 3.12
- `requirements.txt` 已锁定依赖版本
- 默认本地检测不需要 `.env`；只有显式使用 Roboflow 时才需要 API key

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如需临时切回 Roboflow，创建 `.env` 并设置后端：

```text
API_KEY=your_api_key_here
CROP_INFERENCE_BACKEND=roboflow
```

本地检测默认并发运行：stage1 和 stage2 会按机器与后端保守自动分配 worker。可用 `CROP_STAGE1_WORKERS`、`CROP_STAGE2_WORKERS` 或 `CROP_WORKERS` 调整；有 NVIDIA GPU 且安装了 `onnxruntime-gpu` 时，`LOCAL_YOLO_DEVICE=auto` 会优先使用 CUDA，否则自动走 CPU。

识别阶段也分池并行：先用较大的 barcode worker 池批量快扫，只有条码没命中的字段才进入较小的 OCR worker 池。默认 worker 数按 CPU 核数保守计算，barcode 最多自动到 8，OCR 默认自动为 1（PaddleOCR 多实例并发初始化不稳定）；可用 `SCAN2_BARCODE_WORKERS`、`SCAN2_OCR_WORKERS` 或 `SCAN2_WORKERS` 调整，`SCAN2_PARALLEL=0` 可临时关闭识别并行。

## Windows 一键启动

双击 `start.bat`。

脚本会自动创建 `new_images`。如果目录里没有支持的图片，会直接提示放入图片，不会继续跑空流水线。

结果写到 `runs/` 下。如果输出目录已存在，程序会自动创建本次运行专用目录，避免覆盖旧结果。

## 命令行

```bash
python run_all.py --input new_images --out runs --format jsonl --log-level info --device cpu
```

查看完整参数：

```bash
python run_all.py --help
```

## 输出

常见输出：

- `stage1_labels/` 或 `stage1_labels_run_*`
- `stage2_fields/model/`
- `stage2_fields/part_no/`
- `stage2_fields/sn/`
- `stage2_fields/manifest.jsonl`
- `stage2_fields/model_sn_ocr.jsonl`
- `stage2_fields/debug_ocr_barcode.log`，只在 `--log-level debug` 时写入

结果 JSONL 中的 `model_raw` 和 `sn_raw` 默认脱敏。只有在可信本地调试且确实需要完整 raw 值时，才设置 `SCAN2_UNSAFE_RAW=1` 或 `HUAWEIOCR_UNSAFE_RAW=1`。
model 字段默认按 barcode-first 处理；如需临时禁用 model 裁剪图条码识别，可设置 `SCAN2_MODEL_BARCODE=0`。

JSONL 示例：

```json
{"label_id":"sample_label_001.png__label_1","model":"S380-S8P2T","sn":"2000000000AGQC000000","model_raw":"[masked-model-raw]","sn_raw":"2000********0000","model_src":"ocr_color","sn_src":"barcode"}
```

## GUI

运行：

```bash
python gui_app.py
```

或英文界面：

```bash
python gui_app_en.py
```

GUI 会把选择的图片复制到本次运行专用输入目录，阻止重复并发运行，并从原始 JSONL 行导出 Excel。界面表格、运行日志和导出的 `model` / `sn` 都保留完整识别值。

默认 OCR 配置使用 `en_PP-OCRv5_mobile_rec`，它在当前标签样本上比 server 识别更稳。需要对比 server 识别器时，可在 `.env` 中加入 `HUAWEIOCR_OCR_PROFILE=server`。

## 测试

```bash
python -m unittest discover -v
```

测试覆盖输出目录隔离、manifest 解析、条码 CLI 调用预算、debug 日志脱敏、GUI 输入暂存、模型安装锁恢复等回归场景。

## 安全说明

- 不要提交 `.env`。
- 不要把 API key 写死在代码里。
- debug 日志默认关闭，只在 `--log-level debug` 时写入。
- GUI 日志和 self-check 日志会脱敏本机路径。
- PyInstaller 打包只包含条码 CLI 运行所需文件，不包含 vendor examples/configs。
