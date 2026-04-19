# HuaweiOCR

面向 Windows 的设备标签批量 OCR 流水线。

## 做什么

流程会检测标签区域，裁剪 model/SN 字段，优先识别 SN 条码，再用 OCR 兜底，最后输出结构化 JSONL。

Roboflow 检测步骤需要有效 `API_KEY`。PaddleOCR 模型和条码 CLI 可以随包携带，但检测步骤不是完全离线。

## 环境

- Windows
- 推荐 Python 3.12
- `requirements.txt` 已锁定依赖版本
- `.env` 中配置 Roboflow API key

安装依赖：

```bash
python -m pip install -r requirements.txt
```

创建 `.env`：

```text
API_KEY=your_api_key_here
```

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
- `stage2_fields/sn/`
- `stage2_fields/manifest.jsonl`
- `stage2_fields/model_sn_ocr.jsonl`
- `stage2_fields/debug_ocr_barcode.log`，只在 `--log-level debug` 时写入

默认会对 `model_raw` 和 `sn_raw` 脱敏，降低结果文件中的原始 OCR/条码文本泄露风险。只有受控本地调试才建议在代码层显式使用 `unsafe_raw=True`。

JSONL 示例：

```json
{"label_id":"input_0001.png__label_1","model":"S380-S8P2T","sn":"4E25A0170000","model_raw":"********","sn_raw":"4E25********0000","model_src":"ocr_color","sn_src":"barcode"}
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

GUI 会把选择的图片复制到本次运行专用输入目录，阻止重复并发运行，并从原始 JSONL 行导出 Excel。英文界面表格展示可以脱敏，但导出的 `model` 和 `sn` 保留识别值。

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
