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

crop 与 scan2 **只通过文件系统交互**（manifest + 图片），互不 import。
识别策略与指标纪律：条码命中与 OCR 恢复分开统计，OCR 恢复不得计入条码命中率
（项目规则，见 AGENTS.md）。

## 模块地图

| 文件 | 职责 | 依赖 |
|---|---|---|
| run_all.py | CLI 编排入口：调 crop.main → scan2.main，写 run_summary | crop, scan2 |
| crop.py | Stage1 标签裁剪 + Stage2 字段裁剪 + 启发式过滤 + manifest 写入 | local_yolo, (可选)inference_sdk, envutil |
| scan2.py | 条码优先识别、OCR 兜底、PartNo→Model 学习映射、统计 | barcode, sn_barcode, ocr, app_paths, envutil |
| local_yolo.py | 本地 YOLO 推理（ONNX/PT，letterbox+NMS），无内部依赖 | — |
| ocr.py | PaddleOCR 引擎初始化与兼容处理 | app_paths |
| barcode.py | 通用一维码解码库（pyzbar / BarcodeReaderCLI），无业务逻辑 | win_subprocess |
| sn_barcode.py | SN 专用条码扫描（候选生成/多解码器/评分选择），dataclass 结构 | barcode |
| part_no_barcode_rescue.py | PartNo 的 Code128 救援解码 | — |
| linear_barcode_repair.py | Code128 像素级修复框架（scan2 的修复路径调用） | — |
| gui_app.py / gui_app_en.py | 中/英文 Tkinter GUI（打包入口是 gui_app.py） | gui_pipeline |
| gui_pipeline.py | GUI↔管线桥接：动态 import、OCR 预热线程、输入复制 | crop, scan2（动态） |
| app_paths.py | frozen/dev 路径解析、模型安装同步锁 | — |
| envutil.py | 统一的环境变量解析工具 | — |
| validate_sn_barcodes.py | SN 条码结果对 manifest 的验证工具 | sn_barcode |
| tools/ | 训练/数据集/审计脚本，**非运行时依赖** | — |
| legacy/ | 已退役代码，勿引用 | — |

## 配置

三层：`.env`（由 crop.load_dotenv 加载，主要是 API_KEY）→ 环境变量（约 70 个，
全量清单见 [configuration.md](configuration.md)）→ CLI 参数（run_all.py）。
阈值类常量目前硬编码在 crop.py 头部（MIN_CONF/PADDING/NMS 等）。

## 输出目录

由 crop.py 的模块级路径常量 + `configure_paths()` 决定；输出目录已存在时自动建
`*_run_N` 兄弟目录防覆盖（tests/test_locked_output_dirs.py 守护此行为）。

## 验证体系

- 单元测试：`python -m unittest discover -s tests`（core 纯函数已有回归网）
- 指标基线：115 图样本集，见 [validation/baseline/README.md](../validation/baseline/README.md)。
  **任何 crop/scan2 改动后必须重跑并保证 model/sn 值逐条相等。**
- 打包验证：`python -m PyInstaller --noconfirm HuaweiOCR.spec`，
  检查 dist 产物可启动且 `_internal\.env` 等存在。
