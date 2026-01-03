# HuaweiOCR

## 一句话定义
面向设备标签图片的自动化识别工具：裁剪字段 -> 条码识别 -> OCR -> 结构化导出（JSONL），提供 Windows 一键运行。

## 项目简介
这是一个面向设备标签/条码图片的本地批处理工具，结合目标检测（Roboflow 推理）、条码识别与 OCR，自动提取设备型号（Model）与序列号（SN）。项目核心思路是“先定位，再识别”：

- 使用 Roboflow 模型检测整张图片中的标签区域。
- 对标签区域进行二次裁剪，分别提取 model/sn 的小块图像。
- 对小块图像执行条码识别与 OCR，提取最终的型号/序列号。
- 输出结构化结果（JSONL）与调试日志，便于回溯与优化。

## Demo
建议放一张 GIF 或截图，让人不看代码也能理解。

- `docs/demo.gif`
- `docs/screenshot.png`

## Pipeline
```
Input -> crop -> barcode -> ocr -> postprocess -> output
```

## 功能亮点
- 端到端批量处理：从原图到结果一键完成。
- 多阶段裁剪与过滤：提升定位精度与识别成功率。
- 条码+OCR 双通道：条码优先，OCR 兜底。
- 调试产物留存：失败样本与日志方便复盘。
- 本地离线依赖：包含模型与辅助工具，尽量减少外部依赖。

## 目录结构（核心）
- `crop.py`：阶段 1/2 裁剪逻辑（Roboflow 检测 + 裁剪）
- `scan2.py`：OCR/条码识别与结构化输出
- `barcode.py`：条码识别增强逻辑
- `run_all.py`：一键流水线入口
- `start.bat`：Windows 一键运行脚本

## 运行环境
- Windows
- Python 3.10+（建议）
- 需要有效的 Roboflow API Key

## 快速开始（Windows）
1) 在项目根目录创建 `.env` 文件（不要提交到 Git）：
```
API_KEY=your_api_key_here
```

2) 双击 `start.bat` 启动流程。

## CLI（命令行接口）
```
python run_all.py --input ./images --out ./out --format jsonl --log-level info --device cpu
```

查看完整参数：
```
python run_all.py --help
```

## 处理流程概览
1) 读取 `new_images/` 中的原始图片。
2) 阶段 1：检测并裁剪标签区域，保存到 `stage1_labels/`。
3) 阶段 2：从标签区域裁剪 model/sn，小块保存到 `stage2_fields/`。
4) `scan2.py` 对裁剪结果执行条码识别和 OCR。
5) 生成最终结果文件 `model_sn_ocr.jsonl` 和调试日志 `debug_ocr_barcode.log`。

## 输出说明
- `stage1_labels/`：标签裁剪结果
- `stage2_fields/model/`：型号裁剪结果
- `stage2_fields/sn/`：序列号裁剪结果
- `model_sn_ocr.jsonl`：最终识别结果（每行一个 JSON）
- `debug_ocr_barcode.log`：识别过程日志
默认输出在项目根目录；使用 `--out` 可指定输出根目录。

## 量化统计
CLI 会打印统计信息（示例字段）：
- 处理总量 N、总耗时、平均单张耗时
- SN 提取成功率
- 正则校验通过率
- 错误类型分布（barcode_fail / ocr_fail / regex_fail）

## 输出格式示例
下面是一个简化的 JSONL 示例（单行）：
```
{"label_id":"img_001__label_1","model":"S380-S8P2T","sn":"4E25XXXXXXXX","model_src":"barcode","sn_src":"ocr","model_raw":"...","sn_raw":"..."}
```

## 鲁棒性策略
- 多尺度放大：对小条码做放大后再识别。
- ROI 截取：只处理标签关键区域，减少干扰。
- 旋转尝试：0/90/180/270 方向轮询。
- 正则校验：对 SN/Model 做规则过滤。
- 失败样本归档：失败图片会落盘，方便复盘。

## 常见问题
- 没有识别结果：请检查图片清晰度、角度、光照；或调大裁剪尺寸/阈值。
- 报 API_KEY 缺失：确认 `.env` 存在且格式正确。
- 识别不稳定：可调整 `crop.py` / `scan2.py` 中阈值参数。

## 安全说明
- GitHub 仓库不包含 `.env`，API Key 不会暴露。
- 需要共享给别人使用时，私下发送 `.env` 文件即可。
- 不建议将 API Key 写进代码或公开仓库。
- 轮换 Key：替换 `.env` 中的 `API_KEY` 即可，无需改代码。
