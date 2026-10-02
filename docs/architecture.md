# 架构说明（Architecture）

> 面向维护者：读完本文应能回答"数据怎么流、哪个文件负责什么、改动会波及谁"。
> 领域词汇（Stage1/Stage2/barcode-first 等）见 [CONCEPTS.md](../CONCEPTS.md)。

## 一句话

Windows 优先的华为设备标签批量识别管线：YOLO 把源照片切成标签和字段小图，
"条码优先、OCR 兜底"识别 Model/SN/PartNo，输出 JSONL；CLI 和 Tkinter GUI 两个入口，
PyInstaller 打包发布。

## 数据流

```
new_images/（源照片，一张照片可能含多个标签）
  │
  ▼ crop.py — Stage1：YOLO 检测标签 → 方向矫正/多道拒绝门过滤
stage1_labels/*.png（每个物理标签一张裁剪图）
  │
  ▼ crop.py — Stage2：YOLO 检测字段（model / sn / part_no）
stage2_fields/{model,sn,part_no}/*.png
stage2_fields/manifest.jsonl        ←— 阶段间契约，以 label_id 为键
  │
  ▼ scan2.py — 识别：PartNo条码 → Model条码 → SN条码，各自 OCR 兜底
stage2_fields/model_sn_ocr.jsonl    ←— 最终结果（字段协议见 output_schema.md）
  │
  ▼ run_all.py — 汇总
run_summary.json（+ 可选 --excel-out 导出 Excel）
```

阶段交接以文件系统为主（manifest + 图片）。Stage2 裁剪会调用识别模块验证
PartNo/Model 条码；识别阶段的延迟 Model 补裁会调用裁剪函数。这些调用保留为
运行时导入，像素分析模块不反向依赖管线或 GUI。
识别策略与指标纪律：条码命中与 OCR 恢复分开统计，OCR 恢复不得计入条码命中率
（项目规则，见 AGENTS.md）。

## 模块地图

| 文件 | 职责 | 依赖 |
|---|---|---|
| huaweiocr/cli.py | CLI 编排：裁剪 → 识别，写 run_summary | crop, scan2, io.results |
| huaweiocr/detect/crop_pipeline.py | Stage1/Stage2 裁剪、输出路径与 manifest | local_yolo, (可选)inference_sdk, detect 下的像素分析模块 |
| huaweiocr/detect/barcode_regions.py | 条码带定位、SN 分段与条码图像归一化 | OpenCV, NumPy, core.geometry |
| huaweiocr/detect/text_regions.py | 连通域文字行定位 | OpenCV |
| huaweiocr/detect/field_quality.py | 字段裁剪完整性、邻行过滤、留白与修剪 | barcode_regions, text_regions |
| huaweiocr/recognition/pipeline.py | 条码优先识别、OCR 兜底、PartNo→Model 学习映射 | barcode, sn_barcode, ocr, core.extract |
| huaweiocr/recognition/metrics.py | 型号/SN 统计、条码与 OCR 恢复分开计数 | — |
| huaweiocr/core/parallel.py | 保序并发执行与进度回调 | Python 标准库 |
| huaweiocr/core/sources.py | 共用中英文来源名称与 SN 辅助判断后缀 | — |
| huaweiocr/io/images.py | 图片格式清单、稳定递归枚举与 Unicode 路径写图 | Python 标准库；写图时加载 OpenCV |
| huaweiocr/io/results.py | JSONL 读取与 CLI/GUI 共用 Excel 导出 | openpyxl（导出时加载） |
| local_yolo.py | 本地 YOLO 推理（ONNX/PT，letterbox+NMS），无内部依赖 | — |
| ocr.py | PaddleOCR 引擎初始化与兼容处理 | app_paths |
| barcode.py | 通用一维码解码库（pyzbar / BarcodeReaderCLI），无业务逻辑 | win_subprocess |
| sn_barcode.py | SN 专用条码扫描（候选生成/多解码器/评分选择），dataclass 结构 | barcode |
| part_no_barcode_rescue.py | PartNo 的 Code128 救援解码 | — |
| linear_barcode_repair.py | Code128 像素级修复框架（scan2 的修复路径调用） | — |
| huaweiocr/barcode/code128.py | 通用解码、PartNo 救援和像素修复共用的 Code128 码表与位展开 | Python 标准库 |
| huaweiocr/ui/app.py | 中/英文 Tkinter GUI | ui.pipeline, ui.i18n, io.results |
| huaweiocr/ui/pipeline.py | GUI↔管线桥接：动态 import、OCR 预热线程、输入复制 | crop, scan2（动态） |
| huaweiocr/ui/i18n.py | 桌面文案 | core.sources |
| app_paths.py | frozen/dev 路径解析、模型安装同步锁 | — |
| envutil.py | 统一的环境变量解析工具 | — |
| validate_sn_barcodes.py | SN 条码结果对 manifest 的验证工具 | sn_barcode |
| tools/ | 训练/数据集/审计脚本，**非运行时依赖** | — |

根目录的 `crop.py`、`scan2.py`、`run_all.py`、`gui_app.py`、`gui_pipeline.py`
和 `gui_i18n.py` 都是兼容入口，沿用已有模块别名模式，旧 import 与包内实现共享
同一个模块对象。`gui_app_en.py` 继续委托同一个 GUI；PyInstaller 仍从
`gui_app.py` 开始静态收集。也可以使用 `python -m huaweiocr` 运行 CLI。

## 配置

三层：`.env`（由 crop.load_dotenv 加载，主要是 API_KEY）→ 环境变量（约 70 个，
全量清单见 [configuration.md](configuration.md)）→ CLI 参数（run_all.py）。
阈值类常量位于 `huaweiocr/detect/crop_pipeline.py` 头部（MIN_CONF/PADDING/NMS 等）。

## 输出目录

由裁剪管线的模块级路径常量 + `configure_paths()` 决定；输出目录已存在时自动建
`*_run_N` 兄弟目录防覆盖（tests/test_locked_output_dirs.py 守护此行为）。

## 验证体系

- 单元测试：`python -m unittest discover -s tests`（core 纯函数已有回归网）
- 指标基线：115 图样本集，见 [validation/baseline/README.md](../validation/baseline/README.md)。
  **任何 crop/scan2 改动后必须重跑并保证 model/sn 值逐条相等。**
  `python tools/check_baseline_regression.py` 自动跑批 + 对比 + 判定（opt-in，
  需要本地 `batch_runs/baseline_input`，CI 上不跑）。
- 打包验证：`python -m PyInstaller --noconfirm HuaweiOCR.spec`，
  检查 dist 产物可启动且 `_internal\Cython\Utility\CppSupport.cpp` 存在
  （构建 venv 必须装有 Cython）。`.env` 自 2026-07 起有意不打包：API 密钥
  不得随发布包扩散，打包版默认本地 ONNX 后端。
