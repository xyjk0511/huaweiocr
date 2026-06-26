# Stage1 Label Detector Retrain Plan (2026-06-22)

## 目标

这次不再继续堆 Stage1 规则，改成补 `label_detector` 数据，让模型自己学会两件事：

1. 不把快递单识别成 `huawei_label`
2. 被快递单部分遮挡时，仍然尽量框出真正的华为产品标签

当前线上 Stage1 类别只有一个：

- `0 = huawei_label`

对应数据集与模型：

- 当前较新的 box-only 数据集：`datasets/huawei_yolov8_v2_boxonly`
- 当前本地 Stage1 ONNX：`local_models/detectors/label_detector.onnx`
- 历史训练目录：`local_models/training/label_detector_v2s_yolov8s`

## 这批图的真实问题

这 24 张图不是单一问题，而是两类问题叠在一起：

### A. 快递单干扰导致误检/误过滤

典型图：

- `常州.jpg`
- `武汉.jpg`
- `上海.jpg`

现象：

- detector 已经在右侧区域给了框
- 但框里混入快递单或只截到一部分，后面的 Stage1 过滤把它丢了

### B. 多箱堆叠场景下的真实召回不足

典型图：

- `佳木斯.jpg`
- `上海嘉定.jpg`
- `成都.jpg`
- `珠海.jpg`

现象：

- 右侧或底部的产品标签被快递单挡住后，detector 自身就没稳定框出来
- 这种问题继续靠规则补会越来越脆，最好直接补训练样本

## 已整理的硬样本

已准备一批待标注图：

- 脚本：`tools/prepare_stage1_hardcases.py`
- 输出目录：`datasets/huawei_stage1_hardcases_20260622`

运行：

```powershell
.\\.venv\\Scripts\\python.exe .\\tools\\prepare_stage1_hardcases.py
```

输出内容：

- `datasets/huawei_stage1_hardcases_20260622/raw_images/`
- `datasets/huawei_stage1_hardcases_20260622/labels_pending/`
- `datasets/huawei_stage1_hardcases_20260622/manifest.csv`

`manifest.csv` 里已经写了：

- 当前 Stage1 裁出数
- 目标可见产品标签数
- 问题类型
- 标注备注

## 标注规则

### 只标什么

只标华为设备箱体上的产品标签矩形区域。

也就是这种白底黑字、包含 `Part No / Model / SN / MAC / EAN / UPC` 的产品信息标签。

### 明确不要标什么

不要标：

- 顺丰快递单
- 抖音商城收货单
- 手写红圈红字
- 箱体大字品牌文案
- 纯条码贴纸但不是产品信息标签的内容

### 遮挡标签怎么标

如果产品标签被快递单挡住，但仍能看出它是标准华为产品标签，并且还能稳定画出标签矩形，就继续标。

建议经验线：

- 可见部分大于约 `35%`：标
- 虽然被挡住，但四边边界还能基本判断：标
- 只剩很小一条边、已经无法稳定定义矩形：不标

### 框怎么画

框住“产品标签矩形本体”，不要：

- 把整个纸箱框进去
- 把快递单一起框进去
- 故意缩得只剩中间文字区

目标是让模型学到“产品标签整体外观”，不是学内部某一行字。

## 推荐的数据组织方式

不要直接改旧数据集。先克隆一份 v3：

```text
datasets/
  huawei_yolov8_v3_boxonly/
    train/
      images/
      labels/
    valid/
      images/
      labels/
    test/
      images/
      labels/
    local_data.yaml
```

建议做法：

1. 复制 `datasets/huawei_yolov8_v2_boxonly` 为 `datasets/huawei_yolov8_v3_boxonly`
2. 把这 24 张图标完后并入 `train/` 和 `valid/`
3. 这批 hard cases 里保留 `5-6` 张到 `valid/`，不要全丢 `train/`

建议优先放到 `valid/` 的图：

- `佳木斯.jpg`
- `常州.jpg`
- `上海.jpg`
- `武汉.jpg`
- `珠海.jpg`
- `上海嘉定.jpg`

这几张最能反映“快递单干扰 + 遮挡召回”。

## 训练建议

直接沿用当前较稳的 `yolov8s` 路线。

参考现有参数：

- 历史训练：`local_models/training/label_detector_v2s_yolov8s/args.yaml`
- 旧配置：`imgsz=640 batch=8 epochs=160`

建议首轮：

```powershell
yolo detect train ^
  model=yolov8s.pt ^
  data=datasets/huawei_yolov8_v3_boxonly/local_data.yaml ^
  imgsz=640 ^
  epochs=120 ^
  batch=8 ^
  patience=30 ^
  project=local_models/training ^
  name=label_detector_v3s_hardcases
```

如果显存紧张：

- 先把 `batch` 改到 `4`

如果首轮效果仍然对快递单敏感，再考虑：

- 增加更多“快递单很大但不该标”的负样本图
- 保持单类检测，不新增 `shipping_note` 类

这里不建议先上多类。你的运行链路现在只需要稳定的 `huawei_label`，多加一个 `shipping_note` 类会增加推理和后处理复杂度，但不一定带来更高收益。

## 导出 ONNX

训练后导出：

```powershell
yolo export ^
  model=local_models/training/label_detector_v3s_hardcases/weights/best.pt ^
  format=onnx ^
  imgsz=640 ^
  simplify=True
```

然后替换：

```text
local_models/detectors/label_detector.onnx
```

替换前先备份旧模型，例如：

```text
local_models/detectors/label_detector.prev_before_hardcases_20260622.onnx
```

## 验收标准

不要只看训练集指标，要看真实图回归。

至少做两组验证：

### 1. 这 24 张 hard cases

目标：

- 快递单不被识别成产品标签
- 目标产品标签数尽量贴近 `manifest.csv` 中的 `target_label_count`

### 2. 旧的稳定样本集

目标：

- 之前已经能正确裁出的普通图不能回退
- Stage2/扫码成功率不能因为 Stage1 框形状漂移而下降

## 实际判断标准

这次模型更新是否合格，看这三条：

1. `常州 / 武汉 / 上海 / 佳木斯` 这些图的 Stage1 数量上来
2. 快递单本身没有新增误检
3. 旧数据集上没有明显回退

只满足第 1 条不够；如果靠放宽模型把快递单也框进来，那还是失败。
