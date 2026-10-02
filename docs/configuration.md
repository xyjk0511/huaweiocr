# 配置项清单（Configuration Reference）

> 以环境变量名为准；运行实现位于 `huaweiocr/`，根目录同名脚本保留兼容入口。
> 布尔开关统一接受 `1/true/yes/on` 与 `0/false/no/off`。
> 开发环境的 `.env` 由 `crop.load_dotenv` 加载，可放 `API_KEY` 及其他运行时环境变量。
> 发布包不包含 `.env`，默认走本地 ONNX；Roboflow 模式需要用户提供 `API_KEY` 环境变量。

## 入口与日志

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| LOG_LEVEL | info | 全局日志级别（run_all --log-level 会写入） | huaweiocr/detect/crop_pipeline.py |
| CROP_PROGRESS_LOG | 有 GUI sink 时开 | Stage 进度日志 | huaweiocr/detect/crop_pipeline.py |
| SCAN2_PROGRESS_LOG | 有 GUI sink 时开 | 识别进度日志 | huaweiocr/recognition/pipeline.py |

## 推理后端与检测

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| CROP_INFERENCE_BACKEND | local | local=本地 ONNX；roboflow=云端 | huaweiocr/detect/crop_pipeline.py |
| API_KEY | 空 | Roboflow 密钥（仅 roboflow 后端需要，放 .env） | huaweiocr/detect/crop_pipeline.py |
| LOCAL_YOLO_DEVICE | auto | 本地推理设备 | huaweiocr/detect/crop_pipeline.py |
| LOCAL_YOLO_LABEL_MODEL | 内置路径 | Stage1 检测器权重覆盖 | local_yolo.py:45 |
| LOCAL_YOLO_LABEL_MODEL_PREFER_HARDCASE | True | 优先加载难例回训版检测器 | local_yolo.py:48 |
| LOCAL_YOLO_CONF / LOCAL_YOLO_LABEL_CONF / LOCAL_YOLO_FIELD_CONF | -/派生/0.25 | 置信度阈值 | local_yolo.py:498-508 |
| LOCAL_YOLO_NMS | 0.45 | NMS 阈值 | local_yolo.py:495 |

## Stage1 裁剪门（诊断"裁剪失败"先看这里）

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| CROP_STAGE1_USE_RAW_LABEL_DETECTIONS | True | 原始检测框模式 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_KEEP_ALL_CROPS | 跟随 raw 模式 | 保留全部候选（关过滤） | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_SAVE_PREVIEWS | True | 保存带框预览图 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_LOCAL_DIRECT_SOURCE | True | 本地后端直接用源图 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_ORIENTATION_NORMALIZE | 1 | 180° 方向矫正 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_ROTATION_RETRY | 开 | 旋转重试 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_REQUIRE_FIELD_STRUCTURE | True | 要求条码状字段结构 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_FILTER_BACKGROUND_LABELS | True | 过滤模糊小背景标签 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_BACKGROUND_LABEL_MAX_CONF / _MAX_ABS_AREA_RATIO / _MAX_REL_MEDIAN_RATIO / _FOCUS_MIN | 常量 | 背景过滤阈值微调 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE1_HARDCASE_MODEL_SUPPLEMENT | True | 难例模型补充检测 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE2_ROTATION_RETRY | 1 | Stage2 旋转重试 | huaweiocr/detect/crop_pipeline.py |
| CROP_STAGE2_SAVE_MODEL | False | 保存 model 字段裁剪图 | huaweiocr/detect/crop_pipeline.py |

## 并发（机器卡顿时调小这些）

| 变量 | 默认 | 作用 |
|---|---|---|
| CROP_WORKERS / CROP_STAGE1_WORKERS / CROP_STAGE2_WORKERS | 按机器自动 | 裁剪并发数（huaweiocr/detect/crop_pipeline.py） |
| SCAN2_WORKERS / SCAN2_BARCODE_WORKERS / SCAN2_OCR_WORKERS | 按机器自动 | 识别并发数（huaweiocr/recognition/pipeline.py） |
| SCAN2_PARALLEL | True | 识别并发总开关（huaweiocr/recognition/pipeline.py） |

## 识别策略（scan2）

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| SCAN2_OCR_FALLBACK | True | 条码失败后 OCR 兜底 | huaweiocr/recognition/pipeline.py |
| SCAN2_PART_NO_FIRST | True | PartNo 条码优先策略 | huaweiocr/recognition/pipeline.py |
| SCAN2_PART_NO_OCR_FALLBACK | 跟随 OCR_FALLBACK | PartNo 的 OCR 兜底 | huaweiocr/recognition/pipeline.py |
| SCAN2_MODEL_BARCODE | True | Model 条码扫描 | huaweiocr/recognition/pipeline.py |
| SCAN2_DELAYED_MODEL_CROP | True | 延迟 Model 裁剪 | huaweiocr/recognition/pipeline.py |
| SCAN2_SCAN_LABEL_WITH_SN / _WITHOUT_SN | True | 标签级扫描范围 | huaweiocr/recognition/pipeline.py |
| SCAN2_ALLOW_UNKNOWN_MODELS | False | 放开型号白名单（新产品导入期可临时开） | huaweiocr/recognition/pipeline.py |
| SCAN2_PART_NO_MODEL_MAP_PATH | 空 | 外部 PartNo→Model 映射 JSON（新产品适配通道） | huaweiocr/recognition/pipeline.py |
| SCAN2_LEARNED_MODEL_CODES_PATH | 空 | 学习型号持久化路径覆盖 | huaweiocr/recognition/pipeline.py |
| HUAWEIOCR_PRODUCT_PROFILE | 空 | 外部产品知识 profile（product_profile.json，覆盖 PartNo→Model 等知识表；新产品适配通道） | huaweiocr/core/profile.py:40 |
| SCAN2_PART_NO_MAP_OVERWRITE | False | 允许覆盖已学映射 | huaweiocr/recognition/pipeline.py |
| SCAN2_PART_NO_PIXEL_REPAIR / _MAX_SCORE / _MIN_MARGIN | 1 / 0.17 / 0.010 | PartNo 像素修复 | huaweiocr/recognition/pipeline.py |
| SCAN2_PART_NO_STRIPE_RESCUE | 1 | PartNo 条纹救援 | huaweiocr/recognition/pipeline.py |
| SCAN2_UNSAFE_RAW / HUAWEIOCR_UNSAFE_RAW | 关 | 结果中的 raw 字段不脱敏（默认脱敏） | huaweiocr/recognition/pipeline.py |

## SN 条码解码（sn_barcode）

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| SN_BARCODE_DECODERS | 全部 | 限定解码器集合（pyzbar/cli/zxingcpp） | huaweiocr/barcode/sn.py |
| SN_BARCODE_MAX_CANDIDATES / MAX_DECODER_ATTEMPTS | 常量 | 候选/尝试预算 | huaweiocr/barcode/sn.py |
| SN_BARCODE_PIXEL_REPAIR / _SOURCES | True / sn | 像素修复及来源限定 | huaweiocr/barcode/sn.py |
| SN_BARCODE_REPAIR_CHARSET / _LENGTHS / _TEMPLATE / _REGEX / _MAX_PROFILES | alnum/常量/常量/空/6 | 修复参数 | huaweiocr/barcode/sn.py |
| SN_BARCODE_DEBUG_CANDIDATES / _DEBUG_DIR | 关/空 | 候选图调试落盘 | huaweiocr/recognition/pipeline.py |

## 条码 CLI（barcode.py）

| 变量 | 默认 | 作用 |
|---|---|---|
| BARCODE_CLI_TIMEOUT_SECONDS | 2 | 单次 CLI 调用超时（负载高时超时会转 OCR 兜底，见基线 README 软门槛） |
| BARCODE_CLI_MAX_CALLS_PER_PATCH | 4 | 每 patch 最大调用数 |
| CODE128_VISUAL_MIN_SCORE / _MIN_CORR / _MIN_SEP / _MIN_SYMBOL_CORR / _MIN_SYMBOL_SEP | 0.75/0.53/0.20/0/0 | 视觉校验阈值 |

## 路径与 OCR 引擎

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| HUAWEIOCR_DATA_DIR | LOCALAPPDATA 派生 | 用户数据目录覆盖 | huaweiocr/io/paths_runtime.py |
| HUAWEIOCR_MODEL_DIR | 派生 | 模型目录覆盖 | huaweiocr/io/paths_runtime.py |
| HUAWEIOCR_OCR_PROFILE | mobile | OCR 引擎档位（mobile/server） | huaweiocr/detect/ocr_engine.py |
| HUAWEIOCR_PREWARM_OCR | 跟随 OCR_FALLBACK | GUI 启动预热 OCR | huaweiocr/ui/pipeline.py |
| OCR_IMG_DIR / OCR_OUT_JSONL | stage2_fields/sn 派生 | ocr.py 独立运行时的输入输出 | huaweiocr/detect/ocr_engine.py |

（LOCALAPPDATA / XDG_DATA_HOME / PATH / PROCESSOR_ARCHITECTURE 为系统变量，仅读取。）
