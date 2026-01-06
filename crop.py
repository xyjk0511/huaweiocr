import os
import sys
import json
import cv2
import shutil
import numpy as np
from inference_sdk import InferenceHTTPClient

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()

def set_log_level(level: str) -> None:
    global LOG_LEVEL
    LOG_LEVEL = (level or "info").lower()

def _log(msg: str, level: str = "info") -> None:
    levels = {"debug": 10, "info": 20, "warn": 30, "error": 40}
    cur = levels.get(LOG_LEVEL, 20)
    val = levels.get(level, 20)
    if val >= cur:
        print(msg)

def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass

# ==================== 基础配置 ====================
load_dotenv()
API_KEY = os.environ.get("API_KEY", "")
if not API_KEY:
    raise RuntimeError("API_KEY is missing. Set it in .env or environment variables.")

# 模型1：裁剪大标签（你原来类似 "huawei-2ha7t/7"）
MODEL1_ID = "huawei-2ha7t/7"
MODEL1_LABEL_CLASS = "huawei_label"   # 如果你的大标签类名不是这个，就改成你Roboflow里的class名

# 模型2：裁剪字段（你说的新模型 sn_model / sn_model 2）
# 常见写法就是 "sn_model/2"；如果Roboflow显示的是 "sn-model-xxxx/2"，就写那个
MODEL2_ID = ("sn_model/9")
MODEL2_MODEL_CLASS = "model"          # 你的字段类名
MODEL2_SN_CLASS = "sn"                # 你的字段类名

INPUT_DIR = "new_images"
STAGE1_DIR = "stage1_labels"          # 所有小图都放这里（扁平文件夹）
STAGE2_DIR = "stage2_fields"
OUT_MODEL_DIR = os.path.join(STAGE2_DIR, "model")
OUT_SN_DIR = os.path.join(STAGE2_DIR, "sn")
MANIFEST_PATH = os.path.join(STAGE2_DIR, "manifest.jsonl")
FAILED_DIR = os.path.join(STAGE2_DIR, "failed")

def configure_paths(input_dir=None, out_dir=None):
    global INPUT_DIR, STAGE1_DIR, STAGE2_DIR
    global OUT_MODEL_DIR, OUT_SN_DIR, MANIFEST_PATH, FAILED_DIR, TMP_DIR
    if input_dir:
        INPUT_DIR = input_dir
    if out_dir:
        STAGE1_DIR = os.path.join(out_dir, "stage1_labels")
        STAGE2_DIR = os.path.join(out_dir, "stage2_fields")
    OUT_MODEL_DIR = os.path.join(STAGE2_DIR, "model")
    OUT_SN_DIR = os.path.join(STAGE2_DIR, "sn")
    MANIFEST_PATH = os.path.join(STAGE2_DIR, "manifest.jsonl")
    FAILED_DIR = os.path.join(STAGE2_DIR, "failed")
    TMP_DIR = os.path.join(STAGE1_DIR, "_tmp_infer")

# 裁剪/过滤参数（按需微调）
MIN_CONF_1 = 0.50
MIN_SIZE_1 = 300
PADDING_1 = 0.15
NMS_1 = 0.30

# Stage2 分类阈值（推荐）
MIN_CONF_MODEL = 0.20
MIN_CONF_SN    = 0.15   # 长期稳定阈值

# Model 尺寸过滤 (宽>=120, 高>=18)
MIN_W_MODEL = 120
MIN_H_MODEL = 18

MIN_SIZE_SN    = 60     # 60 能覆盖大部分窄条 sn

PADDING_2_MODEL = 0.10
PADDING_2_SN    = 0.15  # sn 带条码，边缘留多一点更稳
NMS_2 = 0.30

# ==================== 环境细节：Windows中文路径 ====================
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def ensure_dirs():
    # 每次启动时清空目录
    if os.path.exists(STAGE1_DIR):
        shutil.rmtree(STAGE1_DIR)
    if os.path.exists(STAGE2_DIR):
        shutil.rmtree(STAGE2_DIR)

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(STAGE1_DIR, exist_ok=True)
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_SN_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def save_png(path, bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if ok:
        buf.tofile(path)
        return True
    return False

def pred_class(p):
    return p.get("class") or p.get("class_name") or ""

def to_xywh_topleft(p):
    x = float(p["x"]); y = float(p["y"])
    w = float(p["width"]); h = float(p["height"])
    x1 = int(x - w/2); y1 = int(y - h/2)
    return x1, y1, int(w), int(h)

def crop_from_pred(img, p, pad_ratio):
    H, W = img.shape[:2]
    x = float(p["x"]); y = float(p["y"])
    w = float(p["width"]); h = float(p["height"])
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio

    x1 = int(x - w/2 - pad_w)
    y1 = int(y - h/2 - pad_h)
    x2 = int(x + w/2 + pad_w)
    y2 = int(y + h/2 + pad_h)

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size else None

def nms(preds, min_conf, nms_thresh):
    boxes, confs, kept = [], [], []
    for p in preds:
        if not isinstance(p, dict):
            continue
        if "x" not in p:
            continue
        conf = float(p.get("confidence", 1.0))
        if conf < min_conf:
            continue
        x1, y1, w, h = to_xywh_topleft(p)
        boxes.append([x1, y1, w, h])
        confs.append(conf)
        kept.append(p)

    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, confs, min_conf, nms_thresh)
    if idxs is None or len(idxs) == 0:
        return []
    idxs = idxs.flatten().tolist() if hasattr(idxs, "flatten") else list(idxs)
    return [kept[i] for i in idxs]

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts]

# ==================== Roboflow Client ====================
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

# ⚠️ 修正：TMP_DIR 必须在 ensure_dirs 之后创建，或者在 infer_with_resize 里确保存在
# 因为 ensure_dirs 会删除 STAGE1_DIR，导致 TMP_DIR 也不存在了
TMP_DIR = os.path.join(STAGE1_DIR, "_tmp_infer")

def _write_tmp_jpg(bgr, out_path, quality=85):
    # 确保父目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok:
        buf.tofile(out_path)
        return True
    return False

def infer_with_resize(original_img_bgr, original_img_path, model_id, max_side=1600):
    """
    目的：避免 413，把图缩放压缩后再 infer。
    返回：predictions（坐标已映射回 original_img_bgr 的像素坐标系）
    """
    H, W = original_img_bgr.shape[:2]
    longest = max(H, W)

    # 计算缩放比例（只缩小不放大）
    scale = 1.0
    if longest > max_side:
        scale = max_side / float(longest)

    # 生成临时图
    if scale < 1.0:
        resized = cv2.resize(original_img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        resized = original_img_bgr

    base = os.path.splitext(os.path.basename(original_img_path))[0]
    tmp_path = os.path.join(TMP_DIR, f"{base}__infer_tmp.jpg")

    # 两次尝试：第一次质量85，仍413就再降质量+再缩
    attempts = [
        {"quality": 85, "max_side": max_side},
        {"quality": 70, "max_side": min(max_side, 1200)},
    ]

    last_err = None
    for a in attempts:
        ms = a["max_side"]
        q = a["quality"]

        # 重新按本轮 max_side 计算 scale
        scale = 1.0
        if longest > ms:
            scale = ms / float(longest)
        if scale < 1.0:
            resized = cv2.resize(original_img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resized = original_img_bgr

        _write_tmp_jpg(resized, tmp_path, quality=q)

        try:
            res = client.infer(tmp_path, model_id=model_id)
            preds = (res.get("predictions", []) if isinstance(res, dict) else res) or []

            # 坐标从 resized 映射回 original：除以 scale
            if scale != 1.0:
                mapped = []
                inv = 1.0 / scale
                for p in preds:
                    if not isinstance(p, dict) or "x" not in p:
                        continue
                    p2 = dict(p)
                    p2["x"] = float(p2["x"]) * inv
                    p2["y"] = float(p2["y"]) * inv
                    p2["width"] = float(p2["width"]) * inv
                    p2["height"] = float(p2["height"]) * inv
                    mapped.append(p2)
                preds = mapped

            return preds

        except Exception as e:
            last_err = e
            # 继续下一轮降质量/降分辨率
            continue

    # 两次都失败，抛出最后一次错误
    raise last_err

# ==================== Stage 1：大图 -> 标签小图 ====================
def stage1_crop_labels(img_path):
    img = read_image(img_path)
    if img is None:
        _log(f"❌ 读图失败: {img_path}", "error")
        return []

    # 直接推理，不做额外过滤
    preds = infer_with_resize(img, img_path, model_id=MODEL1_ID)

    # 只保留大标签类别（这个保留，否则别的乱七八糟的框也会被裁出来）
    preds = [p for p in preds if pred_class(p) == MODEL1_LABEL_CLASS]

    # 不再做尺寸过滤 / 置信度过滤 / NMS
    final_preds = preds

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_paths = []

    for i, p in enumerate(final_preds, 1):
        crop = crop_from_pred(img, p, PADDING_1)
        if crop is None:
            continue
        out_name = f"{base}__label_{i}.png"
        out_path = os.path.join(STAGE1_DIR, out_name)
        save_png(out_path, crop)
        out_paths.append(out_path)

    _log(f"Stage1: {os.path.basename(img_path)} -> {len(out_paths)} label crops", "info")
    return out_paths

# ==================== Stage 2：标签小图 -> model/sn 小块 ====================
def stage2_crop_fields(label_img_path):
    img = read_image(label_img_path)
    if img is None:
        return None

    # 内部函数：只按类别分组，不做任何过滤
    def parse_preds(raw_preds):
        by_cls = {MODEL2_MODEL_CLASS: [], MODEL2_SN_CLASS: []}
        for p in raw_preds:
            if not isinstance(p, dict):
                continue
            if "x" not in p:
                continue
            c = pred_class(p)
            if c in by_cls:
                by_cls[c].append(p)
        # 不做尺寸/置信度过滤，也不做 NMS，全部保留
        return by_cls[MODEL2_MODEL_CLASS], by_cls[MODEL2_SN_CLASS]

    # Round 1: max_side=1600
    preds1 = infer_with_resize(img, label_img_path, MODEL2_ID, max_side=1600)

    # Debug（保留你原本的调试逻辑）
    bn = os.path.basename(label_img_path)
    if bn in {"1__label_6.png", "2__label_6.png"} or bn.startswith("2dc0ee5e") or bn.startswith("358d508d"):
        _log(f"DEBUG {bn} RAW PREDS (R1): {json.dumps(preds1, ensure_ascii=False)[:2000]}", "debug")

    model_preds, sn_preds = parse_preds(preds1)

    # 如果某一类为空，再跑一轮更大 max_side（只为了可能多抓一点框，不做“合规判断”）
    if not model_preds or not sn_preds:
        preds2 = infer_with_resize(img, label_img_path, MODEL2_ID, max_side=2048)
        combined = (preds1 or []) + (preds2 or [])
        model_preds, sn_preds = parse_preds(combined)

    # 只取每一类置信度最高的那个框
    def best_pred(pred_list):
        if not pred_list:
            return None
        return max(pred_list, key=lambda x: float(x.get("confidence", 1.0)))

    best_model_pred = best_pred(model_preds)
    best_sn_pred = best_pred(sn_preds)

    base = os.path.splitext(os.path.basename(label_img_path))[0]
    out = {
        "label_crop": label_img_path,
        "model_path": None,
        "sn_path": None,
        "model_conf": None,
        "sn_conf": None,
    }

    # 处理 Model：只裁剪最高置信的一个
    if best_model_pred is not None:
        crop_m = crop_from_pred(img, best_model_pred, PADDING_2_MODEL)
        if crop_m is not None:
            mp = os.path.join(OUT_MODEL_DIR, f"{base}__model.png")
            save_png(mp, crop_m)
            out["model_path"] = mp
            out["model_conf"] = float(best_model_pred.get("confidence", 0))

    # 处理 SN：只裁剪最高置信的一个
    if best_sn_pred is not None:
        crop_s = crop_from_pred(img, best_sn_pred, PADDING_2_SN)
        if crop_s is not None:
            sp = os.path.join(OUT_SN_DIR, f"{base}__sn.png")
            save_png(sp, crop_s)
            out["sn_path"] = sp
            out["sn_conf"] = float(best_sn_pred.get("confidence", 0))

    # 如果 model/sn 都没出来，把原小图复制到 failed 方便回看
    if not out["model_path"] and not out["sn_path"]:
        fail_path = os.path.join(FAILED_DIR, f"{base}__FAILED.png")
        save_png(fail_path, img)

    return out

def main(input_dir=None, out_dir=None, log_level="info"):
    set_log_level(log_level)
    configure_paths(input_dir=input_dir, out_dir=out_dir)
    ensure_dirs()
    
    # 额外创建分类文件夹
    MISS_SN_DIR = os.path.join(STAGE2_DIR, "miss_sn")
    MISS_MODEL_DIR = os.path.join(STAGE2_DIR, "miss_model")
    MISS_BOTH_DIR = os.path.join(STAGE2_DIR, "miss_both")
    os.makedirs(MISS_SN_DIR, exist_ok=True)
    os.makedirs(MISS_MODEL_DIR, exist_ok=True)
    os.makedirs(MISS_BOTH_DIR, exist_ok=True)

    imgs = list_images(INPUT_DIR)
    if not imgs:
        _log(f"⚠️ {INPUT_DIR} 没有图片，把原始大图放进去再跑", "warn")
        return

    # 清空 manifest（每次跑重新生成）
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        pass

    # Stage1：全部大图 -> 小图
    all_label_crops = []
    for p in imgs:
        all_label_crops.extend(stage1_crop_labels(p))

    # Stage2：全部小图 -> model/sn
    ok_any = 0
    ok_both = 0
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        for lp in all_label_crops:
            r = stage2_crop_fields(lp)
            if not r:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
            has_model = bool(r.get("model_path"))
            has_sn = bool(r.get("sn_path"))
            
            if has_model or has_sn:
                ok_any += 1
            if has_model and has_sn:
                ok_both += 1
                
            # 分类拷贝失败样本
            if has_model and not has_sn:
                shutil.copy2(lp, os.path.join(MISS_SN_DIR, os.path.basename(lp)))
            elif has_sn and not has_model:
                shutil.copy2(lp, os.path.join(MISS_MODEL_DIR, os.path.basename(lp)))
            elif (not has_sn) and (not has_model):
                shutil.copy2(lp, os.path.join(MISS_BOTH_DIR, os.path.basename(lp)))

    _log(f"\nStage1 complete: {len(all_label_crops)} label crops generated", "info")
    _log(f"Stats: at least one field {ok_any}; both fields {ok_both}", "info")
    _log(f"Manifest: {MANIFEST_PATH}", "info")
    _log(f"Failed categories: {MISS_SN_DIR} / {MISS_MODEL_DIR}", "info")

if __name__ == "__main__":
    main()
