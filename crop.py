import concurrent.futures
import hashlib
import os
import sys
import json
import cv2
import re
import shutil
import tempfile
import threading
import time
from collections import Counter
import numpy as np
from huaweiocr.core.geometry import (
    SLANT_GUARD_ANGLE_DEG,
    box_iou,
    box_overlap_ratio,
    crop_from_box,
    expand_box_pixels,
    pred_class,
    slant_guard_px,
    to_xywh_topleft,
    union_boxes,
)
from envutil import env_flag_default as _env_flag_default
from envutil import env_float_default as _env_float_default

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
LOG_SINK = None
STAGE1_REJECT_COUNTS = Counter()
STAGE1_REJECT_COUNTS_LOCK = threading.Lock()

def set_log_level(level: str) -> None:
    global LOG_LEVEL
    LOG_LEVEL = (level or "info").lower()

def set_log_sink(sink) -> None:
    global LOG_SINK
    LOG_SINK = sink

def _log(msg: str, level: str = "info") -> None:
    levels = {"debug": 10, "info": 20, "warn": 30, "error": 40}
    cur = levels.get(LOG_LEVEL, 20)
    val = levels.get(level, 20)
    if val >= cur:
        if LOG_SINK:
            LOG_SINK(msg)
        else:
            print(msg)


def crop_progress_log_enabled() -> bool:
    raw = os.environ.get("CROP_PROGRESS_LOG")
    if raw is None:
        return bool(LOG_SINK)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _crop_label_name(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path or "")))[0]


def reset_stage1_reject_counts():
    with STAGE1_REJECT_COUNTS_LOCK:
        STAGE1_REJECT_COUNTS.clear()


def record_stage1_reject(reason: str, crop_img=None, img_path: str = "", index: int = 0):
    safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(reason or "unknown")).strip("._")
    if not safe_reason:
        safe_reason = "unknown"
    with STAGE1_REJECT_COUNTS_LOCK:
        STAGE1_REJECT_COUNTS[safe_reason] += 1
        reason_count = STAGE1_REJECT_COUNTS[safe_reason]

    if LOG_LEVEL != "debug" or crop_img is None:
        return

    try:
        source_base = _crop_label_name(img_path) or "unknown"
        safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_base).strip("._") or "unknown"
        reject_dir = os.path.join(FAILED_DIR, "stage1_rejects", safe_reason)
        filename = f"{safe_base}__{int(index):04d}__{reason_count:06d}.png"
        save_png(os.path.join(reject_dir, filename), crop_img)
    except Exception:
        pass


def _stage2_result_summary(result: dict | None) -> str:
    if not result:
        return "未保留"
    hits = []
    if result.get("part_no_path"):
        hits.append("PartNo")
    if result.get("model_path"):
        hits.append("Model")
    if result.get("sn_path"):
        hits.append("SN")
    return "命中 " + "/".join(hits) if hits else "未保留"

def load_dotenv(path=".env"):
    paths = []
    if os.path.isabs(path):
        paths.append(path)
    else:
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            paths.append(os.path.join(exe_dir, path))
            paths.append(os.path.join(exe_dir, "_internal", path))
        paths.append(os.path.join(os.getcwd(), path))
        paths.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))

    seen = set()
    for dotenv_path in paths:
        dotenv_path = os.path.abspath(dotenv_path)
        if dotenv_path in seen or not os.path.exists(dotenv_path):
            continue
        seen.add(dotenv_path)
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip().lstrip("\ufeff")
                    val = val.strip().strip('"').strip("'")
                    if key and val and not os.environ.get(key):
                        os.environ[key] = val
        except Exception:
            pass

# ==================== 基础配置 ====================
load_dotenv()
API_KEY = os.environ.get("API_KEY", "")

# 模型1：裁剪大标签（你原来类似 "huawei-2ha7t/7"）
MODEL1_ID = "huawei-2ha7t/7"
MODEL1_HARDCASE_SUPPLEMENT_ID = "huawei-2ha7t-hardcase/1"
MODEL1_LABEL_CLASS = "huawei_label"   # 如果你的大标签类名不是这个，就改成你Roboflow里的class名

# 模型2：裁剪字段（你说的新模型 sn_model / sn_model 2）
# 常见写法就是 "sn_model/2"；如果Roboflow显示的是 "sn-model-xxxx/2"，就写那个
MODEL2_ID = ("sn_model/9")
MODEL2_MODEL_CLASS = "model"          # 你的字段类名
MODEL2_PART_NO_CLASS = "partno"       # Part No: text + its barcode
MODEL2_SN_CLASS = "sn"                # 你的字段类名

INPUT_DIR = "new_images"
STAGE1_DIR = "stage1_labels"          # 所有小图都放这里（扁平文件夹）
STAGE1_PREVIEW_DIR = "stage1_previews"
STAGE2_DIR = "stage2_fields"
OUT_MODEL_DIR = os.path.join(STAGE2_DIR, "model")
OUT_SN_DIR = os.path.join(STAGE2_DIR, "sn")
OUT_PART_NO_DIR = os.path.join(STAGE2_DIR, "part_no")
MANIFEST_PATH = os.path.join(STAGE2_DIR, "manifest.jsonl")
FAILED_DIR = os.path.join(STAGE2_DIR, "failed")
DEFAULT_INPUT_DIR = INPUT_DIR
DEFAULT_STAGE1_DIR = STAGE1_DIR
DEFAULT_STAGE1_PREVIEW_DIR = STAGE1_PREVIEW_DIR
DEFAULT_STAGE2_DIR = STAGE2_DIR
LABEL_ID_ORIGINAL_PATHS = {}

def _refresh_output_paths():
    global OUT_MODEL_DIR, OUT_SN_DIR, OUT_PART_NO_DIR, MANIFEST_PATH, FAILED_DIR, TMP_DIR
    OUT_MODEL_DIR = os.path.join(STAGE2_DIR, "model")
    OUT_SN_DIR = os.path.join(STAGE2_DIR, "sn")
    OUT_PART_NO_DIR = os.path.join(STAGE2_DIR, "part_no")
    MANIFEST_PATH = os.path.join(STAGE2_DIR, "manifest.jsonl")
    FAILED_DIR = os.path.join(STAGE2_DIR, "failed")
    TMP_DIR = os.path.join(STAGE1_DIR, "_tmp_infer")

def configure_paths(input_dir=None, out_dir=None):
    global INPUT_DIR, STAGE1_DIR, STAGE1_PREVIEW_DIR, STAGE2_DIR
    INPUT_DIR = input_dir or DEFAULT_INPUT_DIR
    if out_dir:
        STAGE1_DIR = os.path.join(out_dir, "stage1_labels")
        STAGE1_PREVIEW_DIR = os.path.join(out_dir, "stage1_previews")
        STAGE2_DIR = os.path.join(out_dir, "stage2_fields")
    else:
        STAGE1_DIR = DEFAULT_STAGE1_DIR
        STAGE1_PREVIEW_DIR = DEFAULT_STAGE1_PREVIEW_DIR
        STAGE2_DIR = DEFAULT_STAGE2_DIR
    _refresh_output_paths()

def stage2_save_model_crops_enabled() -> bool:
    return _env_flag_default("CROP_STAGE2_SAVE_MODEL", False)


def stage1_raw_label_mode_enabled() -> bool:
    if not _env_flag_default("CROP_STAGE1_USE_RAW_LABEL_DETECTIONS", True):
        return False
    backend = inference_backend()
    return _is_local_inference_backend(backend) or _is_auto_inference_backend(backend)


def stage1_keep_all_crops_enabled() -> bool:
    return _env_flag_default("CROP_STAGE1_KEEP_ALL_CROPS", stage1_raw_label_mode_enabled())


def stage1_save_preview_enabled() -> bool:
    return _env_flag_default("CROP_STAGE1_SAVE_PREVIEWS", True)


def stage1_local_direct_source_enabled() -> bool:
    return _env_flag_default("CROP_STAGE1_LOCAL_DIRECT_SOURCE", True)

# 裁剪/过滤参数（按需微调）
MIN_CONF_1 = 0.50
MIN_CONF_1_FALLBACK = 0.25
MIN_SIZE_1 = 300
PADDING_1 = 0.07
NMS_1 = 0.30
STAGE1_MIN_ASPECT = 1.35
STAGE1_MAX_ASPECT = 3.10
STAGE1_MAX_RED_RATIO = 0.02
STAGE1_PARTIAL_MIN_ASPECT = 0.65
STAGE1_PARTIAL_MAX_RED_RATIO = 0.04
STAGE1_RAW_EDGE_FALSEPOS_MAX_ASPECT = 1.15
STAGE1_MIN_BOX_AREA_RATIO = 0.02
STAGE1_RELAXED_MIN_BOX_AREA_RATIO = 0.018
STAGE1_BACKGROUND_LABEL_FOCUS_MIN = 800.0
STAGE1_BACKGROUND_LABEL_MAX_ABS_AREA_RATIO = 0.028
STAGE1_BACKGROUND_LABEL_MAX_REL_MEDIAN_RATIO = 0.45
STAGE1_BACKGROUND_LABEL_MAX_CONF = 0.55
STAGE1_TIGHTEN_MIN_AREA_RATIO = 0.10
STAGE1_TIGHTEN_MAX_AREA_RATIO = 0.92
STAGE1_TIGHTEN_PAD_X_RATIO = 0.015
STAGE1_TIGHTEN_PAD_TOP_RATIO = 0.025
STAGE1_TIGHTEN_PAD_BOTTOM_RATIO = 0.12
STAGE1_TIGHTEN_PAD_BOTTOM_IMAGE_RATIO = 0.08
STAGE1_TIGHTEN_MIN_WIDTH_RETAIN_RATIO = 0.65
STAGE1_FALLBACK_MAX_CROPS = 1
STAGE1_SUPPLEMENT_MIN_CONF = 0.15
STAGE1_SUPPLEMENT_BOX_IOU = 0.35
STAGE1_HARDCASE_SUPPLEMENT_MAX_CROPS = 2
DEFAULT_LOCAL_MAX_WORKERS = 4
DEFAULT_CLOUD_MAX_WORKERS = 8

# Stage2 分类阈值（推荐）
MIN_CONF_MODEL = 0.20
MIN_CONF_SN    = 0.15   # 长期稳定阈值

# Model 尺寸过滤 (宽>=120, 高>=18)
MIN_W_MODEL = 120
MIN_H_MODEL = 18

MIN_SIZE_SN    = 60     # 60 能覆盖大部分窄条 sn

PADDING_2_MODEL = 0.10
PADDING_2_SN    = 0.15  # sn 带条码，边缘留多一点更稳

# 字段模型通常会框住文字行本身；真实标签上的 model/sn 条码在字段文字下方。
# Model 不一定有条码：没检测到紧邻下方的一维条码时，只保存 model 文本行。
PADDING_2_MODEL_TEXT_X = 0.20
PADDING_2_MODEL_TEXT_TOP = 0.00
PADDING_2_MODEL_TEXT_BOTTOM = 0.00
PADDING_2_MODEL_BARCODE_X_LEFT = 0.80
PADDING_2_MODEL_BARCODE_X_RIGHT = 1.40
PADDING_2_MODEL_BARCODE_TOP = 0.05
PADDING_2_MODEL_BARCODE_BOTTOM = 1.60
PADDING_2_MODEL_INSIDE_BARCODE_X_LEFT = 0.45
PADDING_2_MODEL_INSIDE_BARCODE_X_RIGHT = 0.65
PADDING_2_MODEL_INSIDE_BARCODE_TOP = 0.00
PADDING_2_MODEL_INSIDE_BARCODE_BOTTOM = 0.47
PADDING_2_SN_X = 0.06
PADDING_2_SN_TOP = 0.05
PADDING_2_SN_BOTTOM = -0.04
STAGE1_SLANT_GUARD_MAX_PX = 36
STAGE2_SLANT_GUARD_MAX_PX = 18
STAGE2_TEXT_SLANT_GUARD_MAX_PX = 8
NMS_2 = 0.30

# ==================== 环境细节：Windows中文路径 ====================
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def _is_file_busy_error(exc):
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32}

def _run_sibling_dir(path):
    parent = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(os.path.normpath(path))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(100):
        suffix = f"{stamp}_{os.getpid()}"
        if index:
            suffix = f"{suffix}_{index}"
        candidate = os.path.join(parent, f"{base}_run_{suffix}")
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"Could not allocate a clean output directory near {path}")

def _prepare_output_dir(path, label, clean=False):
    if not os.path.exists(path):
        return path

    if not clean:
        fallback = _run_sibling_dir(path)
        _log(
            f"WARN: {label} output folder already exists: {path}. "
            f"Using {fallback} for this run.",
            "warn",
        )
        return fallback

    last_error = None
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return path
        except OSError as exc:
            if not _is_file_busy_error(exc):
                raise
            last_error = exc
            time.sleep(0.2)

    fallback = _run_sibling_dir(path)
    _log(
        f"WARN: {label} output folder is busy and cannot be cleared: {path}. "
        f"Using {fallback} for this run. Original error: {last_error}",
        "warn",
    )
    return fallback

def ensure_dirs(clean=False):
    # 每次启动时清空目录
    global STAGE1_DIR, STAGE1_PREVIEW_DIR, STAGE2_DIR
    STAGE1_DIR = _prepare_output_dir(STAGE1_DIR, "stage1", clean=clean)
    STAGE1_PREVIEW_DIR = _prepare_output_dir(STAGE1_PREVIEW_DIR, "stage1_preview", clean=clean)
    STAGE2_DIR = _prepare_output_dir(STAGE2_DIR, "stage2", clean=clean)
    _refresh_output_paths()

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(STAGE1_DIR, exist_ok=True)
    os.makedirs(STAGE1_PREVIEW_DIR, exist_ok=True)
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_SN_DIR, exist_ok=True)
    os.makedirs(OUT_PART_NO_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def save_png(path, bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return False
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".png", dir=os.path.dirname(path))
    os.close(fd)
    try:
        buf.tofile(tmp_path)
        os.replace(tmp_path, path)
        return os.path.isfile(path)
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def save_png_required(path, bgr, context):
    if not save_png(path, bgr):
        raise RuntimeError(f"Failed to write {context}: {path}")
    return path


def stage1_preview_path_for_image(img_path):
    base = input_label_base(img_path)
    return os.path.join(STAGE1_PREVIEW_DIR, f"{base}__preview.png")


def _stage1_draw_preview_image(img, entries, rotation_deg=0):
    if img is None:
        return None
    annotated = img.copy()
    for idx, entry in enumerate(entries or [], start=1):
        box = entry.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        conf = float(entry.get("confidence", 0.0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 4)
        text = f"huawei_label {conf:.2f} #{idx}"
        cv2.putText(
            annotated,
            text,
            (x1, max(36, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    if rotation_deg:
        cv2.putText(
            annotated,
            f"stage1 rotation retry: {rotation_deg} deg",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 128, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _stage1_save_preview_image(img, img_path, entries, rotation_deg=0):
    if not stage1_save_preview_enabled():
        return None
    preview = _stage1_draw_preview_image(img, entries, rotation_deg=rotation_deg)
    if preview is None:
        return None
    return save_png_required(stage1_preview_path_for_image(img_path), preview, "stage1 preview")

def add_slant_guard_to_box(img, box, max_px, top=True, bottom=True, angle_deg=SLANT_GUARD_ANGLE_DEG):
    if box is None or max_px <= 0:
        return box
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    guard = slant_guard_px(x2 - x1, max_px, angle_deg=angle_deg)
    if guard <= 0:
        return (
            max(0, int(x1)),
            max(0, int(y1)),
            min(W, int(x2)),
            min(H, int(y2)),
        )
    if top:
        y1 -= guard
    if bottom:
        y2 += guard
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(W, int(x2)),
        min(H, int(y2)),
    )


def crop_from_pred(img, p, pad_ratio, slant_guard_max_px=0):
    box = box_from_pred(img, p, pad_ratio, slant_guard_max_px=slant_guard_max_px)
    return crop_from_box(img, box)


def box_from_pred(img, p, pad_ratio, slant_guard_max_px=0):
    H, W = img.shape[:2]
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    slant_guard = slant_guard_px(w, slant_guard_max_px)

    x1 = int(x - w/2 - pad_w)
    y1 = int(y - h/2 - pad_h - slant_guard)
    x2 = int(x + w/2 + pad_w)
    y2 = int(y + h/2 + pad_h + slant_guard)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def crop_from_pred_asym(
    img,
    p,
    pad_x_ratio,
    pad_top_ratio,
    pad_bottom_ratio,
    slant_guard_max_px=0,
):
    box = box_from_pred_asym(
        img,
        p,
        pad_x_ratio,
        pad_top_ratio,
        pad_bottom_ratio,
        slant_guard_max_px=slant_guard_max_px,
    )
    return crop_from_box(img, box)


def box_from_pred_asym(img, p, pad_x_ratio, pad_top_ratio, pad_bottom_ratio, slant_guard_max_px=0):
    H, W = img.shape[:2]
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    slant_guard = slant_guard_px(w, slant_guard_max_px)

    x1 = int(x - w/2 - w * pad_x_ratio)
    y1 = int(y - h/2 - h * pad_top_ratio - slant_guard)
    x2 = int(x + w/2 + w * pad_x_ratio)
    y2 = int(y + h/2 + h * pad_bottom_ratio + slant_guard)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    return (x1, y1, x2, y2)


def box_from_pred_xy_asym(
    img,
    p,
    pad_left_ratio,
    pad_right_ratio,
    pad_top_ratio,
    pad_bottom_ratio,
    slant_guard_max_px=0,
):
    H, W = img.shape[:2]
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    slant_guard = slant_guard_px(w, slant_guard_max_px)

    x1 = int(x - w / 2 - w * pad_left_ratio)
    y1 = int(y - h / 2 - h * pad_top_ratio - slant_guard)
    x2 = int(x + w / 2 + w * pad_right_ratio)
    y2 = int(y + h / 2 + h * pad_bottom_ratio + slant_guard)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def stage1_red_pixel_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 70, 60])
    upper_red_1 = np.array([12, 255, 255])
    lower_red_2 = np.array([170, 70, 60])
    upper_red_2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)
    return float(np.count_nonzero(mask)) / float(mask.size)


def stage1_has_product_corner_mark(img):
    h, w = img.shape[:2]
    x1 = int(w * 0.48)
    x2 = int(w * 0.99)
    y1 = int(h * 0.02)
    y2 = int(h * 0.68)
    if x2 <= x1 or y2 <= y1:
        return False

    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    components = cv2.connectedComponentsWithStats(dark, 8)
    if len(components) < 4:
        return False
    _, _, stats, _ = components
    min_w = max(9, int(w * 0.008))
    min_h = max(10, int(h * 0.018))
    max_w = max(min_w + 1, int(w * 0.16))
    max_h = max(min_h + 1, int(h * 0.20))

    for i in range(1, len(stats)):
        _, _, cw, ch, area = stats[i]
        if cw < min_w or ch < min_h or cw > max_w or ch > max_h:
            continue
        fill = float(area) / float(max(1, cw * ch))
        aspect = float(cw) / float(max(1, ch))
        if fill > 0.40 and 0.35 <= aspect <= 2.80:
            return True
    return False


def _stage1_corner_mark_score(img, x1_ratio, y1_ratio, x2_ratio, y2_ratio):
    h, w = img.shape[:2]
    x1 = max(0, min(w, int(w * x1_ratio)))
    x2 = max(0, min(w, int(w * x2_ratio)))
    y1 = max(0, min(h, int(h * y1_ratio)))
    y2 = max(0, min(h, int(h * y2_ratio)))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = float(np.count_nonzero(dark)) / float(max(1, dark.size))
    components = cv2.connectedComponentsWithStats(dark, 8)
    score = dark_ratio * 0.2
    if len(components) < 4:
        return score

    _, _, stats, _ = components
    for i in range(1, len(stats)):
        _, _, cw, ch, area = stats[i]
        if cw < 8 or ch < 8:
            continue
        if cw > roi.shape[1] * 0.75 or ch > roi.shape[0] * 0.75:
            continue
        fill = float(area) / float(max(1, cw * ch))
        aspect = float(cw) / float(max(1, ch))
        if fill >= 0.28 and 0.25 <= aspect <= 2.8:
            score = max(score, min(0.45, fill * min(float(area), 4000.0) / 4000.0))
    return score


def _stage1_qr_centers(img):
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return []
    if not hasattr(cv2, "QRCodeDetector"):
        return []
    try:
        detector = cv2.QRCodeDetector()
        centers = []

        if hasattr(detector, "detectMulti"):
            result = detector.detectMulti(img)
            ok = bool(result[0]) if isinstance(result, tuple) and result else bool(result)
            points = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            if ok and points is not None:
                pts = np.asarray(points, dtype=np.float32).reshape(-1, 4, 2)
                for quad in pts:
                    centers.append(tuple(np.mean(quad, axis=0)))

        if centers:
            return centers

        result = detector.detect(img)
        ok = bool(result[0]) if isinstance(result, tuple) and result else bool(result)
        points = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if ok and points is not None:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 4, 2)
            for quad in pts:
                centers.append(tuple(np.mean(quad, axis=0)))
        return centers
    except Exception:
        return []


def _stage1_qr_orientation_decision(img):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return ""
    centers = _stage1_qr_centers(img)
    for cx, cy in centers:
        rx = float(cx) / float(w)
        ry = float(cy) / float(h)
        if rx < 0.25 and ry > 0.55:
            return "rotate_180"
    for cx, cy in centers:
        rx = float(cx) / float(w)
        ry = float(cy) / float(h)
        if rx > 0.70 and 0.15 < ry < 0.70:
            return "upright"
    return ""


def _stage1_region_product_mark_score(img, side):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    components = cv2.connectedComponentsWithStats(dark, 8)
    if len(components) < 4:
        return 0.0

    _, _, stats, _ = components
    best = 0.0
    min_w = max(24, int(w * 0.018))
    min_h = max(24, int(h * 0.035))
    max_w = max(min_w + 1, int(w * 0.24))
    max_h = max(min_h + 1, int(h * 0.32))
    for i in range(1, len(stats)):
        x, y, cw, ch, area = stats[i]
        cx = (float(x) + float(cw) * 0.5) / float(w)
        cy = (float(y) + float(ch) * 0.5) / float(h)
        if side == "left":
            if cx > 0.26 or cy < 0.42:
                continue
        else:
            if cx < 0.68 or cy > 0.70:
                continue
        if cw < min_w or ch < min_h or cw > max_w or ch > max_h:
            continue
        fill = float(area) / float(max(1, cw * ch))
        aspect = float(cw) / float(max(1, ch))
        if fill < 0.16 or aspect < 0.25 or aspect > 1.75:
            continue
        score = fill * (float(cw) / float(w)) * (float(ch) / float(h)) * 100.0
        best = max(best, score)
    return best


def _stage1_left_bottom_product_mark_score(img):
    return _stage1_region_product_mark_score(img, "left")


def _stage1_right_product_mark_score(img):
    return _stage1_region_product_mark_score(img, "right")


def stage1_should_rotate_180_label(img):
    if os.environ.get("CROP_STAGE1_ORIENTATION_NORMALIZE", "1").strip().lower() in {"0", "false", "no"}:
        return False
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return False
    h, w = img.shape[:2]
    if h < 80 or w < 120:
        return False

    qr_decision = _stage1_qr_orientation_decision(img)
    if qr_decision == "rotate_180":
        return True
    if qr_decision == "upright":
        return False

    # If the top-right M/L + QC/QR corner is clearly visible, the label is already upright.
    # This prevents AP labels with dense left-side barcodes from being flipped by noise.
    upright = (
        _stage1_corner_mark_score(img, 0.58, 0.00, 1.00, 0.42)
        + 0.45 * _stage1_corner_mark_score(img, 0.70, 0.24, 1.00, 0.78)
    )
    inverted = (
        _stage1_corner_mark_score(img, 0.00, 0.58, 0.42, 1.00)
        + 0.45 * _stage1_corner_mark_score(img, 0.00, 0.22, 0.30, 0.76)
    )
    if upright >= 0.28 and upright > inverted * 1.75 and (upright - inverted) >= 0.12:
        return False

    left_mark = _stage1_left_bottom_product_mark_score(img)
    right_mark = _stage1_right_product_mark_score(img)
    if left_mark >= 0.55 and left_mark > right_mark * 2.0:
        return True
    if right_mark >= 0.45 and right_mark > left_mark * 1.5:
        return False

    # Fallback for crops where QR detection fails but the M/L + QR/QC cluster is still visible.
    return (
        left_mark >= 0.25
        and left_mark > right_mark * 1.25
        and inverted >= 0.13
        and inverted > upright * 1.25
        and (inverted - upright) >= 0.07
    )


def rotate_image(img, rotation):
    if rotation not in {90, 180, 270}:
        return img
    if hasattr(cv2, "rotate"):
        if rotation == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if not hasattr(np, "rot90"):
        return img
    k_by_rotation = {90: 3, 180: 2, 270: 1}
    rotated = np.rot90(img, k=k_by_rotation[rotation])
    if hasattr(np, "ascontiguousarray"):
        return np.ascontiguousarray(rotated)
    return rotated


def stage1_normalize_label_orientation(img):
    if stage1_should_rotate_180_label(img):
        return rotate_image(img, 180)
    return img


def stage1_rotation_retry_angles():
    raw = os.environ.get("CROP_STAGE1_ROTATION_RETRY")
    if raw is None:
        return (180,)
    value = raw.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return ()
    if value in {"1", "true", "yes", "on", "180", "rot180"}:
        return (180,)
    if value in {"90", "270", "sideways"}:
        return (90, 270)
    if value in {"all", "full", "any"}:
        return (90, 270, 180)
    return (180,)


def stage1_rotation_retry_enabled():
    return bool(stage1_rotation_retry_angles())


def stage1_rotated_image(img, rotation):
    return rotate_image(img, rotation)


def _stage1_dark_edge_ratios(img):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return {"top": 0.0, "right": 0.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = gray < 110
    top_h = max(2, int(h * 0.04))
    right_w = max(2, int(w * 0.04))
    return {
        "top": float(np.count_nonzero(dark[:top_h, :])) / float(max(1, top_h * w)),
        "right": float(np.count_nonzero(dark[:, w - right_w:])) / float(max(1, h * right_w)),
    }


def _stage1_label_crop_edge_ink_ratios(img):
    if img is None:
        return {"top": 0.0, "left": 0.0, "right": 0.0}
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return {"top": 0.0, "left": 0.0, "right": 0.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = gray < 120
    top_h = max(2, int(h * 0.045))
    side_w = max(3, int(w * 0.035))
    edge_guard = min(4, max(1, side_w // 4))
    upper_h = max(top_h + 1, int(h * 0.76))
    left_slice = dark[:upper_h, edge_guard : min(w, edge_guard + side_w)]
    right_x2 = max(0, w - edge_guard)
    right_x1 = max(0, right_x2 - side_w)
    right_slice = dark[:upper_h, right_x1:right_x2]
    return {
        "top": float(np.count_nonzero(dark[edge_guard : min(h, edge_guard + top_h), :])) / float(max(1, top_h * w)),
        "left": float(np.count_nonzero(left_slice)) / float(max(1, left_slice.size)),
        "right": float(np.count_nonzero(right_slice)) / float(max(1, right_slice.size)),
    }


def stage1_has_invalid_edge_contamination(img):
    ratios = _stage1_dark_edge_ratios(img)
    return ratios["right"] > 0.65 and ratios["top"] > 0.35


def _stage1_box_touches_image_edge(box, image_shape):
    if box is None or image_shape is None or len(image_shape) < 2:
        return False
    img_h, img_w = image_shape[:2]
    if img_h <= 0 or img_w <= 0:
        return False
    margin_px = max(8, int(min(img_h, img_w) * 0.01))
    x1, y1, x2, y2 = box
    return (
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= (img_w - margin_px)
        or y2 >= (img_h - margin_px)
    )


def _stage1_pred_ratios(pred, crop_shape):
    h, w = crop_shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(pred.get("x", 0.0)) / float(w),
        float(pred.get("y", 0.0)) / float(h),
        float(pred.get("width", 0.0)) / float(w),
        float(pred.get("height", 0.0)) / float(h),
    )


def _stage1_field_layout_is_plausible(crop_img, model_preds, part_no_preds, sn_preds):
    if crop_img is None or not hasattr(crop_img, "shape") or len(crop_img.shape) < 2:
        return False
    if len(part_no_preds) != 1 or not (1 <= len(sn_preds) <= 2) or len(model_preds) > 1:
        return False

    part_x, part_y, part_w, part_h = _stage1_pred_ratios(part_no_preds[0], crop_img.shape)
    if not (
        0.02 <= part_x <= 0.45
        and 0.05 <= part_y <= 0.38
        and 0.08 <= part_w <= 0.42
        and 0.04 <= part_h <= 0.22
    ):
        return False

    if model_preds:
        model_x, model_y, model_w, model_h = _stage1_pred_ratios(model_preds[0], crop_img.shape)
        if not (
            0.02 <= model_x <= 0.42
            and 0.10 <= model_y <= 0.55
            and 0.08 <= model_w <= 0.42
            and 0.04 <= model_h <= 0.22
        ):
            return False

    valid_sn = False
    for pred in sn_preds:
        sn_x, sn_y, sn_w, sn_h = _stage1_pred_ratios(pred, crop_img.shape)
        if (
            0.10 <= sn_x <= 0.65
            and 0.42 <= sn_y <= 0.82
            and 0.18 <= sn_w <= 0.78
            and 0.05 <= sn_h <= 0.24
        ):
            valid_sn = True
            break
    return valid_sn


def _stage1_field_layout_is_relaxed_plausible(crop_img, model_preds, part_no_preds, sn_preds):
    if crop_img is None or not hasattr(crop_img, "shape") or len(crop_img.shape) < 2:
        return False
    if len(part_no_preds) != 1 or not (1 <= len(sn_preds) <= 2) or len(model_preds) > 1:
        return False

    part_x, part_y, part_w, part_h = _stage1_pred_ratios(part_no_preds[0], crop_img.shape)
    if not (
        0.01 <= part_x <= 0.50
        and 0.00 <= part_y <= 0.42
        and 0.08 <= part_w <= 0.45
        and 0.04 <= part_h <= 0.24
    ):
        return False

    valid_sn_preds = []
    for pred in sn_preds:
        sn_x, sn_y, sn_w, sn_h = _stage1_pred_ratios(pred, crop_img.shape)
        if (
            0.08 <= sn_x <= 0.72
            and 0.34 <= sn_y <= 0.86
            and 0.18 <= sn_w <= 0.82
            and 0.05 <= sn_h <= 0.26
        ):
            valid_sn_preds.append((sn_x, sn_y, sn_w, sn_h))

    if not valid_sn_preds:
        return False

    if not any(sn_y >= (part_y + 0.18) for _sn_x, sn_y, _sn_w, _sn_h in valid_sn_preds):
        return False

    if model_preds:
        model_x, model_y, model_w, model_h = _stage1_pred_ratios(model_preds[0], crop_img.shape)
        if not (
            0.01 <= model_x <= 0.48
            and max(0.04, part_y - 0.02) <= model_y <= 0.60
            and 0.08 <= model_w <= 0.45
            and 0.04 <= model_h <= 0.24
        ):
            return False
        if not any(model_y <= (sn_y + 0.04) for _sn_x, sn_y, _sn_w, _sn_h in valid_sn_preds):
            return False

    return True


def stage1_has_product_field_structure(img):
    if not _env_flag_default("CROP_STAGE1_REQUIRE_FIELD_STRUCTURE", True):
        return True
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return False
    h, w = img.shape[:2]
    if h < 80 or w < 140:
        return False
    boxes = _barcode_like_boxes_in_region(
        img,
        0,
        int(h * 0.08),
        w,
        int(h * 0.95),
        min_run_h=max(14, int(h * 0.035)),
        min_span_w=max(110, int(w * 0.18)),
        row_trans_threshold=0.055,
        active_threshold=0.25,
    )
    return bool(boxes)


def _stage1_focus_score(img):
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return 0.0
    h, w = img.shape[:2]
    if h < 12 or w < 12:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _stage1_should_reject_blurry_small_background_label(
    box,
    crop_img,
    image_shape,
    confidence=None,
    reference_box_areas=None,
):
    if not _env_flag_default("CROP_STAGE1_FILTER_BACKGROUND_LABELS", True):
        return False
    if box is None or crop_img is None or image_shape is None or len(image_shape) < 2:
        return False
    if confidence is None:
        return False

    max_confidence = _env_float_default(
        "CROP_STAGE1_BACKGROUND_LABEL_MAX_CONF",
        STAGE1_BACKGROUND_LABEL_MAX_CONF,
    )
    if float(confidence) > max_confidence:
        return False

    img_h, img_w = image_shape[:2]
    img_area = max(1, img_h * img_w)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    if box_area <= 0:
        return False

    box_area_ratio = box_area / float(img_area)
    max_abs_area_ratio = _env_float_default(
        "CROP_STAGE1_BACKGROUND_LABEL_MAX_ABS_AREA_RATIO",
        STAGE1_BACKGROUND_LABEL_MAX_ABS_AREA_RATIO,
    )
    max_rel_median_ratio = _env_float_default(
        "CROP_STAGE1_BACKGROUND_LABEL_MAX_REL_MEDIAN_RATIO",
        STAGE1_BACKGROUND_LABEL_MAX_REL_MEDIAN_RATIO,
    )
    is_small_abs = box_area_ratio < max_abs_area_ratio
    is_small_rel = False
    if reference_box_areas:
        median_area = float(np.median(reference_box_areas))
        if median_area > 0:
            is_small_rel = box_area < (median_area * max_rel_median_ratio)
    if not (is_small_abs or is_small_rel):
        return False

    focus_score = _stage1_focus_score(crop_img)
    min_focus_score = _env_float_default(
        "CROP_STAGE1_BACKGROUND_LABEL_FOCUS_MIN",
        STAGE1_BACKGROUND_LABEL_FOCUS_MIN,
    )
    return focus_score < min_focus_score


def stage1_is_product_label_candidate_crop(img):
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return False
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return False
    aspect = float(w) / float(h)
    if aspect < STAGE1_MIN_ASPECT or aspect > STAGE1_MAX_ASPECT:
        return False
    if stage1_red_pixel_ratio(img) > STAGE1_MAX_RED_RATIO:
        return False
    if stage1_has_invalid_edge_contamination(img):
        return False
    return stage1_has_product_corner_mark(img)


def stage1_is_partial_product_label_crop(img):
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return False
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return False
    aspect = float(w) / float(h)
    if aspect < STAGE1_PARTIAL_MIN_ASPECT or aspect > STAGE1_MAX_ASPECT:
        return False
    if stage1_red_pixel_ratio(img) > STAGE1_PARTIAL_MAX_RED_RATIO:
        return False
    if stage1_has_invalid_edge_contamination(img):
        return False
    if not stage1_has_product_corner_mark(img):
        return False
    return stage1_has_product_field_structure(img)


def stage1_is_product_label_crop(img):
    if stage1_is_product_label_candidate_crop(img) and stage1_has_product_field_structure(img):
        return True
    return stage1_is_partial_product_label_crop(img)


def stage1_tighten_label_crop(img):
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return img
    required_cv2 = ("cvtColor", "threshold", "morphologyEx", "dilate", "findContours", "boundingRect")
    if any(not hasattr(cv2, name) for name in required_cv2):
        return img
    H, W = img.shape[:2]
    if H <= 0 or W <= 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        iterations=1,
    )
    dark = cv2.dilate(
        dark,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    image_area = float(W * H)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        area_ratio = (w * h) / image_area
        if area_ratio < STAGE1_TIGHTEN_MIN_AREA_RATIO or area_ratio > STAGE1_TIGHTEN_MAX_AREA_RATIO:
            continue
        aspect = w / float(h)
        if aspect < STAGE1_MIN_ASPECT or aspect > STAGE1_MAX_ASPECT:
            continue
        if w < W * 0.35 or h < H * 0.20:
            continue
        score = w * h
        if score > best_score:
            best_score = score
            best = (x, y, x + w, y + h)

    if best is None:
        return img

    x1, y1, x2, y2 = best
    pad_x = max(6, int((x2 - x1) * STAGE1_TIGHTEN_PAD_X_RATIO))
    pad_top = max(6, int((y2 - y1) * STAGE1_TIGHTEN_PAD_TOP_RATIO))
    pad_bottom = max(
        18,
        int((y2 - y1) * STAGE1_TIGHTEN_PAD_BOTTOM_RATIO),
        int(H * STAGE1_TIGHTEN_PAD_BOTTOM_IMAGE_RATIO),
    )
    box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_top),
        min(W, x2 + pad_x),
        min(H, y2 + pad_bottom),
    )
    if (
        (box[2] - box[0]) < W * STAGE1_TIGHTEN_MIN_WIDTH_RETAIN_RATIO
        and stage1_is_product_label_crop(img)
    ):
        return img
    if (box[2] - box[0]) > W * 0.96 and (box[3] - box[1]) > H * 0.96:
        return img
    tightened = crop_from_box(img, box)
    return tightened if tightened is not None else img


def expand_box_from_pred_asym(
    img,
    p,
    box,
    pad_left,
    pad_right,
    pad_top,
    pad_bottom,
    slant_guard_max_px=0,
):
    if box is None:
        return None
    H, W = img.shape[:2]
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    top = int(y - h / 2 - h * pad_top)
    if pad_top <= 0:
        top = box[1]
    pred_box = (
        max(0, int(x - w / 2 - w * pad_left)),
        max(0, top),
        min(W, int(np.ceil(x + w / 2 + w * pad_right))),
        min(H, int(np.ceil(y + h / 2 + h * pad_bottom))),
    )
    merged = union_boxes(box, pred_box)
    return add_slant_guard_to_box(img, merged, slant_guard_max_px)


def barcode_like_box_below(img, p):
    H, W = img.shape[:2]
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])

    x1 = int(x - w/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    sx1 = max(0, int(x1 - w * PADDING_2_MODEL_BARCODE_X_LEFT))
    sx2 = min(W, int(x2 + w * PADDING_2_MODEL_BARCODE_X_RIGHT))
    sy1 = max(0, int(y2 + h * PADDING_2_MODEL_BARCODE_TOP))
    sy2 = min(H, int(y2 + h * PADDING_2_MODEL_BARCODE_BOTTOM))
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    return barcode_like_box_in_region(
        img,
        sx1,
        sy1,
        sx2,
        sy2,
        min_run_h=max(22, int(h * 0.35)),
        min_span_w=max(100, int(w * 0.70)),
        row_trans_threshold=0.07,
        active_threshold=0.35,
    )


def barcode_like_box_in_region(
    img,
    sx1,
    sy1,
    sx2,
    sy2,
    min_run_h,
    min_span_w,
    row_trans_threshold=0.07,
    active_threshold=0.35,
):
    H, W = img.shape[:2]
    sx1 = max(0, min(W, int(sx1)))
    sx2 = max(0, min(W, int(sx2)))
    sy1 = max(0, min(H, int(sy1)))
    sy2 = max(0, min(H, int(sy2)))
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    roi = img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    if dark.shape[0] < 8 or dark.shape[1] < 80:
        return None

    row_dark = dark.mean(axis=1)
    row_trans = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1) / max(1, dark.shape[1] - 1)
    candidate_rows = (row_dark > 0.08) & (row_dark < 0.75) & (row_trans > row_trans_threshold)

    best = None
    best_score = 0
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            run_h = end - start
            if run_h >= min_run_h:
                band = dark[start:end]
                col_dark = band.mean(axis=0)
                active = col_dark > active_threshold
                active_idx = np.where(active)[0]
                if active_idx.size:
                    span_w = int(active_idx[-1] - active_idx[0] + 1)
                    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
                    aspect = span_w / float(max(run_h, 1))
                    if span_w >= min_span_w and transitions >= 18 and aspect >= 3.0:
                        score = span_w * run_h
                        if score > best_score:
                            pad_x = max(8, int(span_w * 0.04))
                            pad_y = max(4, int(run_h * 0.25))
                            best = (
                                max(0, sx1 + int(active_idx[0]) - pad_x),
                                max(0, sy1 + start - pad_y),
                                min(W, sx1 + int(active_idx[-1]) + 1 + pad_x),
                                min(H, sy1 + end + pad_y),
                            )
                            best_score = score
            start = None

    return best


def _barcode_like_boxes_in_region(
    img,
    sx1,
    sy1,
    sx2,
    sy2,
    min_run_h,
    min_span_w,
    row_trans_threshold=0.07,
    active_threshold=0.35,
):
    H, W = img.shape[:2]
    sx1 = max(0, min(W, int(sx1)))
    sx2 = max(0, min(W, int(sx2)))
    sy1 = max(0, min(H, int(sy1)))
    sy2 = max(0, min(H, int(sy2)))
    if sx2 <= sx1 or sy2 <= sy1:
        return []

    roi = img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    if dark.shape[0] < 8 or dark.shape[1] < 80:
        return []

    return _barcode_like_boxes_from_mask(
        dark,
        sx1,
        sy1,
        W,
        H,
        min_run_h=min_run_h,
        min_span_w=min_span_w,
        row_trans_threshold=row_trans_threshold,
        active_threshold=active_threshold,
    )


def _barcode_like_boxes_from_mask(
    dark,
    sx1,
    sy1,
    image_w,
    image_h,
    min_run_h,
    min_span_w,
    row_trans_threshold=0.07,
    active_threshold=0.35,
):
    row_dark = dark.mean(axis=1)
    row_trans = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1) / max(1, dark.shape[1] - 1)
    candidate_rows = (row_dark > 0.08) & (row_dark < 0.75) & (row_trans > row_trans_threshold)

    candidates = []
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            run_h = end - start
            if run_h >= min_run_h:
                band = dark[start:end]
                col_dark = band.mean(axis=0)
                active = col_dark > active_threshold
                active_idx = np.where(active)[0]
                if active_idx.size:
                    span_w = int(active_idx[-1] - active_idx[0] + 1)
                    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
                    aspect = span_w / float(max(run_h, 1))
                    if span_w >= min_span_w and transitions >= 18 and aspect >= 3.0:
                        score = span_w * run_h
                        pad_x = max(8, int(span_w * 0.04))
                        pad_y = max(4, int(run_h * 0.25))
                        candidates.append(
                            (
                                (
                                    max(0, sx1 + int(active_idx[0]) - pad_x),
                                    max(0, sy1 + start - pad_y),
                                    min(image_w, sx1 + int(active_idx[-1]) + 1 + pad_x),
                                    min(image_h, sy1 + end + pad_y),
                                ),
                                score,
                            )
                        )
            start = None

    return candidates


def model_pred_has_lower_barcode(img, p):
    return model_barcode_box_near_pred(img, p) is not None


def model_barcode_box_inside_pred(img, p):
    H, W = img.shape[:2]
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    x1 = max(0, int(x - w/2))
    y1 = max(0, int(y - h/2))
    x2 = min(W, int(x + w/2))
    y2 = min(H, int(y + h/2))
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    if dark.shape[0] < 12 or dark.shape[1] < 80:
        return None

    row_dark = dark.mean(axis=1)
    row_trans = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1) / max(1, dark.shape[1] - 1)
    candidate_rows = (row_dark > 0.08) & (row_dark < 0.75) & (row_trans > 0.09)
    min_run_h = max(18, int(float(h) * 0.28))
    best = None
    best_score = 0
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            run_h = end - start
            lower_band = start >= dark.shape[0] * 0.30 and end >= dark.shape[0] * 0.60
            if lower_band and run_h >= min_run_h:
                band = dark[start:end]
                col_dark = band.mean(axis=0)
                active = col_dark > 0.35
                active_idx = np.where(active)[0]
                if active_idx.size:
                    span_w = int(active_idx[-1] - active_idx[0] + 1)
                    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
                    aspect = span_w / float(max(run_h, 1))
                    if span_w >= max(90, int(float(w) * 0.55)) and transitions >= 18 and aspect >= 3.0:
                        score = span_w * run_h
                        if score > best_score:
                            pad_x = max(6, int(span_w * 0.03))
                            pad_y = max(2, int(run_h * 0.08))
                            best = (
                                max(0, x1 + int(active_idx[0]) - pad_x),
                                max(0, y1 + start - pad_y),
                                min(W, x1 + int(active_idx[-1]) + 1 + pad_x),
                                min(H, y1 + end + pad_y),
                            )
                            best_score = score
            start = None
    return best


def model_barcode_box_below_pred(img, p, text_box=None):
    barcode_box = barcode_like_box_below(img, p)
    if barcode_box is None:
        return None

    if text_box is None:
        text_box = model_text_line_box(img, p)
    if text_box is not None:
        max_gap = max(8, int(float(p["height"]) * 0.45))
        if barcode_box[1] - text_box[3] > max_gap:
            return None
    if not model_box_looks_like_barcode(img, barcode_box):
        return None
    return barcode_box


def model_box_decodes_as_barcode(img, box):
    candidate = crop_from_box(img, box)
    if candidate is None:
        return False
    return bool(decode_model_crop(candidate))


def model_box_looks_like_barcode(img, box):
    candidate = crop_from_box(img, box)
    if candidate is None:
        return False
    return _crop_has_sn_barcode_stripes(candidate) and model_box_decodes_as_barcode(img, box)


def model_barcode_box_near_pred(img, p, text_box=None):
    inside_barcode_box = model_barcode_box_inside_pred(img, p)
    if inside_barcode_box is not None and model_box_looks_like_barcode(img, inside_barcode_box):
        return inside_barcode_box
    if text_box is None:
        text_box = model_text_line_box(img, p)
    below_barcode_box = model_barcode_box_below_pred(img, p, text_box=text_box)
    if below_barcode_box is not None:
        return below_barcode_box
    if model_box_looks_like_barcode(img, text_box):
        return text_box
    return None


def model_text_box_above_barcode(img, p, barcode_box):
    H, W = img.shape[:2]
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    x1 = max(0, int(x - w/2 - w * PADDING_2_MODEL_TEXT_X))
    x2 = min(W, int(x + w/2 + w * PADDING_2_MODEL_TEXT_X))
    sy1 = max(0, int(y - h/2))
    if int(barcode_box[1]) - sy1 < max(12, int(h * 0.35)):
        sy1 = max(0, int(barcode_box[1] - max(45, h * 1.35)))
    sy2 = max(sy1, min(H, int(barcode_box[1])))
    if x2 <= x1 or sy2 <= sy1:
        return model_text_line_box(img, p)

    component_box = text_component_line_box_near_y(
        img,
        x1,
        x2,
        sy1,
        sy2,
        target_y=float(barcode_box[1]) - max(8, float(h) * 0.25),
        min_span_w=max(70, int(w * 0.35)),
        pad_top_min=1,
        pad_bottom_min=1,
        pad_y_ratio=0.0,
    )
    if component_box is not None and (component_box[3] - component_box[1]) <= max(44, int(h * 0.55)):
        return add_slant_guard_to_box(img, component_box, 0)

    roi = img[sy1:sy2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    if dark.shape[0] < 6 or dark.shape[1] < 20:
        return model_text_line_box(img, p)

    row_dark = dark.mean(axis=1)
    candidate_rows = row_dark > 0.035
    best = None
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            if end - start >= 4:
                best = (start, end)
            start = None

    if best is None:
        return model_text_line_box(img, p)
    start, end = best
    pad_y = max(2, int((end - start) * 0.15))
    text_box = (
        x1,
        max(0, sy1 + start - pad_y),
        x2,
        min(H, sy1 + end + pad_y),
    )
    return add_slant_guard_to_box(img, text_box, STAGE2_TEXT_SLANT_GUARD_MAX_PX)


def text_component_line_box_near_y(
    img,
    x1,
    x2,
    y1,
    y2,
    target_y,
    min_span_w,
    pad_top_min=5,
    pad_bottom_min=5,
    pad_y_ratio=0.22,
):
    H, W = img.shape[:2]
    x1 = max(0, min(W, int(x1)))
    x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H, int(y1)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    components = cv2.connectedComponentsWithStats(bw, 8)
    if len(components) < 4:
        return None
    _, _, stats, _ = components

    comps = []
    roi_h, roi_w = bw.shape[:2]
    for i in range(1, len(stats)):
        cx, cy, cw, ch, area = stats[i]
        if area < 6 or cw < 2 or ch < 4:
            continue
        if ch > max(48, int(roi_h * 0.75)):
            continue
        if cw > int(roi_w * 0.80):
            continue
        fill = float(area) / float(max(1, cw * ch))
        if fill < 0.08:
            continue
        comps.append((cx, cy, cw, ch, area))
    if not comps:
        return None

    comps.sort(key=lambda c: c[1] + c[3] / 2.0)
    clusters = []
    for comp in comps:
        cy = comp[1] + comp[3] / 2.0
        if not clusters:
            clusters.append({"items": [comp], "sum_cy": cy, "count": 1})
            continue
        cluster = clusters[-1]
        mean_cy = cluster["sum_cy"] / cluster["count"]
        if abs(cy - mean_cy) <= max(9, comp[3] * 0.85):
            cluster["items"].append(comp)
            cluster["sum_cy"] += cy
            cluster["count"] += 1
        else:
            clusters.append({"items": [comp], "sum_cy": cy, "count": 1})

    best = None
    best_score = None
    for cluster in clusters:
        items = cluster["items"]
        gx1 = min(c[0] for c in items)
        gy1 = min(c[1] for c in items)
        gx2 = max(c[0] + c[2] for c in items)
        gy2 = max(c[1] + c[3] for c in items)
        span_w = gx2 - gx1
        line_h = gy2 - gy1
        avg_w = sum(c[2] for c in items) / float(len(items))
        if span_w < min_span_w or line_h < 8:
            continue
        if len(items) < 3 and avg_w < 8:
            continue
        if len(items) >= 5 and avg_w <= 3.5:
            continue
        center_y = y1 + (gy1 + gy2) / 2.0
        barcode_like_penalty = 14 if len(items) >= 8 and avg_w < 7.0 else 0
        score = abs(center_y - target_y) + barcode_like_penalty
        if best_score is None or score < best_score:
            best_score = score
            best = (x1, y1 + gy1, x2, y1 + gy2)
    if best is None:
        return None

    line_h = best[3] - best[1]
    pad_top = max(pad_top_min, int(line_h * pad_y_ratio))
    pad_bottom = max(pad_bottom_min, int(line_h * pad_y_ratio))
    return (
        best[0],
        max(0, best[1] - pad_top),
        best[2],
        min(H, best[3] + pad_bottom),
    )


def model_text_line_box(img, p):
    H, W = img.shape[:2]
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    x1 = int(x - w/2 - w * PADDING_2_MODEL_TEXT_X)
    x2 = int(x + w/2 + w * PADDING_2_MODEL_TEXT_X)
    target_y = y + h * 0.10
    search_y1 = int(y - h * 0.75)
    search_y2 = int(y + h * 0.75)
    text_box = text_component_line_box_near_y(
        img,
        x1,
        x2,
        search_y1,
        search_y2,
        target_y,
        min_span_w=max(80, int(w * 0.45)),
    )
    if text_box is not None and (text_box[3] - text_box[1]) <= max(44, int(h * 0.65)):
        return add_slant_guard_to_box(img, text_box, STAGE2_TEXT_SLANT_GUARD_MAX_PX)

    line_h = max(32, int(h * 0.50))
    y = y + h * 0.14
    y1 = int(y - line_h / 2)
    y2 = int(y + line_h / 2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    return add_slant_guard_to_box(img, (x1, y1, x2, y2), STAGE2_TEXT_SLANT_GUARD_MAX_PX)


def trim_model_top_fragment(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 40 or w < 80:
        return crop_img

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_dark = (bw > 0).mean(axis=1)
    candidate_rows = row_dark > 0.03
    runs = []
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            runs.append((start, end))
            start = None

    if len(runs) < 2:
        return crop_img
    first_start, first_end = runs[0]
    second_start, second_end = runs[1]
    first_h = first_end - first_start
    second_h = second_end - second_start
    if first_start == 0 and first_h <= max(14, int(h * 0.10)) and second_h >= 20 and second_start - first_end <= 4:
        trim_y = max(0, second_start - 3)
        if trim_y > 0:
            return crop_img[trim_y:h, :]
    return crop_img


def trim_model_text_only_crop(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 36 or w < 80:
        return crop_img

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_dark = (bw > 0).mean(axis=1)
    candidate_rows = row_dark > 0.03
    runs = []
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            runs.append((start, end))
            start = None

    if len(runs) < 2:
        return crop_img
    first_start, first_end = runs[0]
    second_start, second_end = runs[1]
    first_h = first_end - first_start
    second_h = second_end - second_start
    gap = second_start - first_end
    if first_h >= 20 and gap >= 3 and second_h <= max(14, int(h * 0.25)):
        bottom = min(h, first_end + max(3, int(first_h * 0.12)))
        return crop_img[:bottom, :]
    return crop_img


def crop_model_field(img, p):
    inside_barcode_box = model_barcode_box_inside_pred(img, p)
    if inside_barcode_box is not None and model_box_looks_like_barcode(img, inside_barcode_box):
        text_box = model_text_box_above_barcode(img, p, inside_barcode_box)
        barcode_box = inside_barcode_box
        field_box = union_boxes(text_box, barcode_box)
        H, W = img.shape[:2]
        bx1, by1, bx2, by2 = barcode_box
        bw = bx2 - bx1
        bh = by2 - by1
        text_h = text_box[3] - text_box[1] if text_box is not None else 0
        text_top = int(field_box[1] - max(15, text_h * 0.34))
        band_top = int(barcode_box[1] - max(54, min(58, bh * 0.75)))
        field_box = (
            max(0, int(field_box[0] - max(8, bw * 0.05))),
            max(0, max(text_top, band_top)),
            min(W, int(field_box[2] + max(8, bw * 0.05))),
            min(H, int(barcode_box[3] + max(1, bh * 0.02))),
        )
        return trim_model_top_fragment(crop_from_box(img, field_box))
    else:
        text_box = model_text_line_box(img, p)
        barcode_box = model_barcode_box_below_pred(img, p, text_box=text_box)
        if barcode_box is None and model_box_looks_like_barcode(img, text_box):
            barcode_box = text_box
            text_box = model_text_box_above_barcode(img, p, barcode_box)
    field_box = union_boxes(text_box, barcode_box)
    if barcode_box is not None:
        H, W = img.shape[:2]
        bx1, by1, bx2, by2 = barcode_box
        bw = bx2 - bx1
        bh = by2 - by1
        field_box = (
            max(0, int(min(field_box[0], bx1 - max(14, bw * 0.12)))),
            max(0, int(min(field_box[1], by1 - max(4, bh * 0.10)))),
            min(W, int(max(field_box[2], bx2 + max(14, bw * 0.12)))),
            min(H, int(max(field_box[3], by2 + max(4, bh * 0.18)))),
        )
        field_box = add_slant_guard_to_box(img, field_box, STAGE2_SLANT_GUARD_MAX_PX)
    cropped = crop_from_box(img, field_box)
    if barcode_box is None:
        return trim_model_text_only_crop(cropped)
    return cropped


def fallback_model_crop_from_sn(img, sn_pred):
    if sn_pred is None:
        return None
    H, W = img.shape[:2]
    sx = float(sn_pred["x"])
    sy = float(sn_pred["y"])
    sw = float(sn_pred["width"])
    sh = float(sn_pred["height"])

    # Some newer labels (for example AR180Pro) miss the model detector but still
    # have a reliable SN detector. Stay inside the label crop and search the
    # model row above SN, never the original source photo.
    x1 = int(max(0, min(sx - sw * 0.56, W * 0.07)))
    x2 = int(min(W, max(sx + sw * 0.25, W * 0.48)))
    y1 = int(max(0, sy - sh * 3.05))
    y2 = int(max(y1 + 40, sy - sh * 2.0))
    if x2 <= x1 or y2 <= y1:
        return None

    box = text_component_line_box_near_y(
        img,
        x1,
        x2,
        y1,
        y2,
        target_y=(y1 + y2) / 2.0,
        min_span_w=max(80, int((x2 - x1) * 0.15)),
        pad_top_min=5,
        pad_bottom_min=8,
        pad_y_ratio=0.15,
    )
    if box is None:
        return None

    text_pred = {
        "x": (box[0] + box[2]) / 2.0,
        "y": (box[1] + box[3]) / 2.0,
        "width": box[2] - box[0],
        "height": box[3] - box[1],
    }
    sn_top = int(sy - sh / 2.0)
    search_y1 = max(box[3], int(box[3] - max(2, (box[3] - box[1]) * 0.08)))
    search_y2 = min(
        H,
        max(search_y1 + 1, sn_top - max(6, int(sh * 0.10))),
        int(box[3] + max(50, sh * 1.25)),
    )
    barcode_box = barcode_like_box_in_region(
        img,
        x1,
        search_y1,
        x2,
        search_y2,
        min_run_h=max(18, int(sh * 0.22)),
        min_span_w=max(100, int((x2 - x1) * 0.30)),
        row_trans_threshold=0.04,
        active_threshold=0.25,
    )
    if barcode_box is None:
        barcode_box = barcode_like_box_below(img, text_pred)
    if barcode_box is not None:
        max_gap = max(10, int((box[3] - box[1]) * 0.75))
        if barcode_box[1] - box[3] > max_gap:
            barcode_box = None
    field_box = union_boxes(box, barcode_box)
    return crop_from_box(img, add_slant_guard_to_box(img, field_box, STAGE2_SLANT_GUARD_MAX_PX))


def model_crop_min_width(label_img):
    if label_img is None:
        return MIN_W_MODEL
    _h, w = label_img.shape[:2]
    return max(MIN_W_MODEL, int(w * 0.22))


def model_crop_is_usable(crop_img, label_img=None):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < max(20, MIN_H_MODEL):
        return False
    if w < model_crop_min_width(label_img):
        return False
    return True


def crop_contains_1d_barcode(
    crop_img,
    min_span_ratio=0.28,
    row_trans_threshold=0.18,
    active_threshold=0.35,
):
    return (
        crop_1d_barcode_box(
            crop_img,
            min_span_ratio=min_span_ratio,
            row_trans_threshold=row_trans_threshold,
            active_threshold=active_threshold,
        )
        is not None
    )


def _crop_has_sn_barcode_stripes(crop_img):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 100:
        return False
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0

    min_vertical_run = max(18, int(h * 0.16))
    stripe = np.zeros(w, dtype=bool)
    for x in range(w):
        best = 0
        current = 0
        for y in range(h):
            if dark[y, x]:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        stripe[x] = best >= min_vertical_run

    active_idx = np.where(stripe)[0]
    if active_idx.size == 0:
        return False

    runs = []
    start = None
    for i, value in enumerate(stripe):
        if value and start is None:
            start = i
        at_end = i == len(stripe) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            runs.append((start, end))
            start = None

    widths = [end - start for start, end in runs]
    span_w = int(active_idx[-1] - active_idx[0] + 1)
    transitions = int(np.count_nonzero(stripe[1:] != stripe[:-1]))
    median_width = float(np.median(widths)) if widths else 0.0
    return (
        span_w >= max(85, int(w * 0.20))
        and transitions >= 30
        and len(runs) >= 15
        and median_width <= 12.0
    )


def crop_1d_barcode_box(
    crop_img,
    min_span_ratio=0.28,
    row_trans_threshold=0.18,
    active_threshold=0.35,
):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 18 or w < 80:
        return None
    return barcode_like_box_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(12, int(h * 0.12)),
        min_span_w=max(70, int(w * min_span_ratio)),
        row_trans_threshold=row_trans_threshold,
        active_threshold=active_threshold,
    )


def crop_has_complete_1d_barcode(
    crop_img,
    min_span_ratio=0.28,
    edge_guard_px=2,
    row_trans_threshold=0.18,
    active_threshold=0.35,
):
    box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=min_span_ratio,
        row_trans_threshold=row_trans_threshold,
        active_threshold=active_threshold,
    )
    if box is None:
        return False
    h, w = crop_img.shape[:2]
    x1, y1, x2, y2 = box
    edge_guard_x = max(2, int(edge_guard_px))
    return x1 > edge_guard_x and x2 < w - edge_guard_x and y1 > 1 and y2 < h - 1


def model_crop_barcode_box(crop_img):
    box = crop_1d_barcode_box(crop_img)
    if box is not None:
        return box
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    return barcode_like_box_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(12, int(h * 0.16)),
        min_span_w=max(70, int(w * 0.18)),
        row_trans_threshold=0.04,
        active_threshold=0.25,
    )


def model_crop_has_complete_1d_barcode(crop_img):
    box = model_crop_barcode_box(crop_img)
    if box is None:
        return False
    h, w = crop_img.shape[:2]
    x1, y1, x2, y2 = box
    return x1 > 2 and x2 < w - 2 and y1 > 1 and y2 < h - 1


def model_crop_has_text_above_barcode(crop_img):
    box = model_crop_barcode_box(crop_img)
    if box is None:
        return False
    h, w = crop_img.shape[:2]
    _x1, y1, _x2, _y2 = box
    text_y2 = max(0, int(y1) - 2)
    if text_y2 < max(12, int(h * 0.18)):
        return False
    text_box = text_component_line_box_near_y(
        crop_img,
        0,
        w,
        0,
        text_y2,
        target_y=text_y2 / 2.0,
        min_span_w=max(55, int(w * 0.10)),
        pad_top_min=1,
        pad_bottom_min=1,
        pad_y_ratio=0.0,
    )
    return text_box is not None


def part_no_crop_contains_1d_barcode(crop_img, min_span_ratio=0.16):
    return crop_contains_1d_barcode(
        crop_img,
        min_span_ratio=min_span_ratio,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )


def part_no_crop_has_complete_1d_barcode(crop_img, min_span_ratio=0.18):
    return crop_has_complete_1d_barcode(
        crop_img,
        min_span_ratio=min_span_ratio,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )


def part_no_crop_has_partial_1d_barcode(crop_img, min_span_ratio=0.20):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return False

    boxes = _part_no_barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(6, int(h * 0.03)),
        min_span_w=max(40, int(w * min_span_ratio)),
        row_trans_threshold=0.08,
        active_threshold=0.20,
    )
    if not boxes:
        return False

    box = boxes[0]["box"]
    x1, y1, x2, y2 = box
    span_w = x2 - x1
    run_h = y2 - y1
    if span_w < max(70, int(w * min_span_ratio)):
        return False
    if run_h < max(8, int(h * 0.03)):
        return False
    if y1 > h * 0.72:
        return False
    if y2 < h * 0.12:
        return False
    return True


def part_no_crop_has_edge_clipped_barcode(crop_img, min_span_ratio=0.16, edge_guard_px=2):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return False

    box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=min_span_ratio,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if box is None:
        bands = _part_no_barcode_like_boxes_in_region(
            crop_img,
            0,
            0,
            w,
            h,
            min_run_h=max(6, int(h * 0.03)),
            min_span_w=max(40, int(w * min_span_ratio)),
            row_trans_threshold=0.08,
            active_threshold=0.20,
        )
        if not bands:
            return False
        box = bands[0]["box"]

    x1, _y1, x2, _y2 = box
    edge_guard_x = max(2, int(edge_guard_px))
    return x1 <= edge_guard_x or x2 >= w - edge_guard_x


def part_no_crop_is_structurally_valid(crop_img, allow_partial=False):
    if crop_img is None:
        return False
    if not part_no_crop_has_text_above_barcode(crop_img):
        return False
    if part_no_crop_has_lower_neighbor_content(crop_img):
        return False
    if part_no_crop_has_edge_clipped_barcode(crop_img):
        return False
    if part_no_crop_has_complete_1d_barcode(crop_img, min_span_ratio=0.18):
        return True
    if allow_partial and part_no_crop_has_partial_1d_barcode(crop_img, min_span_ratio=0.20):
        return True
    return part_no_crop_contains_1d_barcode(crop_img, min_span_ratio=0.16)


def pad_part_no_crop_quiet_zone(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return crop_img

    barcode_box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.18,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if barcode_box is None:
        return crop_img

    x1, _y1, x2, y2 = barcode_box
    barcode_w = max(1, x2 - x1)
    barcode_h = max(1, y2 - _y1)
    required_side = max(16, int(barcode_w * 0.08))
    required_bottom = max(16, int(barcode_h * 0.35))
    current_left = x1
    current_right = w - x2
    current_bottom = h - y2
    pad_left = max(0, required_side - current_left)
    pad_right = max(0, required_side - current_right)
    pad_bottom = max(0, required_bottom - current_bottom)
    if pad_left == 0 and pad_right == 0 and pad_bottom == 0:
        return crop_img

    return cv2.copyMakeBorder(
        crop_img,
        0,
        int(pad_bottom),
        int(pad_left),
        int(pad_right),
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _part_no_selected_barcode_overlaps_band(crop_img, band_box):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return False

    barcode_box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.18,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if barcode_box is None:
        return False

    overlap = max(0, min(barcode_box[3], band_box[3]) - max(barcode_box[1], band_box[1]))
    band_h = max(1, band_box[3] - band_box[1])
    barcode_h = max(1, barcode_box[3] - barcode_box[1])
    return overlap >= max(4, int(min(band_h, barcode_h) * 0.45))


def _part_no_two_band_has_lower_neighbor(crop_img, first, second):
    h = crop_img.shape[0]
    gap = second[1] - first[3]
    if gap < max(8, int(h * 0.035)):
        return False
    if first[1] < max(18, int(h * 0.15)):
        return False
    if not _part_no_has_text_above_band(crop_img, first):
        return False
    return _part_no_selected_barcode_overlaps_band(crop_img, first)


def _part_no_has_text_above_band(crop_img, band_box):
    top = max(0, int(band_box[1]))
    if top < 8:
        return False
    region = crop_img[:top, :]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_dark = (bw > 0).mean(axis=1)
    active_rows = row_dark > 0.018
    min_run = max(3, int(crop_img.shape[0] * 0.025))
    start = None
    for i, active in enumerate(active_rows):
        if active and start is None:
            start = i
        at_end = i == len(active_rows) - 1
        if start is not None and ((not active) or at_end):
            end = i if not active else i + 1
            if end - start >= min_run:
                return True
            start = None
    return False


def _part_no_cutoff_before_two_band_neighbor(crop_img, first, second):
    h, _w = crop_img.shape[:2]
    gap_top = max(0, first[3])
    gap_bottom = max(gap_top, second[1])
    first_h = max(1, first[3] - first[1])
    min_text_gap = max(10, int(first_h * 0.50), int(h * 0.06))
    if gap_bottom - gap_top >= max(8, int(h * 0.035)):
        gap_img = crop_img[gap_top:gap_bottom]
        gray = cv2.cvtColor(gap_img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_dark = (bw > 0).mean(axis=1)
        active_rows = row_dark > 0.018
        min_run = max(3, int(h * 0.018))
        start = None
        for i, active in enumerate(active_rows):
            if active and start is None:
                start = i
            at_end = i == len(active_rows) - 1
            if start is not None and ((not active) or at_end):
                end = i if not active else i + 1
                if end - start >= min_run:
                    cutoff = gap_top + start - max(2, int(h * 0.015))
                    if cutoff - first[3] >= min_text_gap:
                        return max(first[3] + 2, min(h - 1, cutoff))
                start = None

    return max(first[3] + 2, second[1] - max(6, int(h * 0.025)))


def _part_no_primary_barcode_box(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    selected = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.18,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if selected is None:
        return None

    bands = _part_no_barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(8, int(h * 0.06)),
        min_span_w=max(60, int(w * 0.18)),
        row_trans_threshold=0.035,
        active_threshold=0.20,
    )
    best_box = None
    best_score = -1
    selected_h = max(1, selected[3] - selected[1])
    for band in bands:
        box = band["box"]
        overlap = max(0, min(selected[3], box[3]) - max(selected[1], box[1]))
        band_h = max(1, box[3] - box[1])
        if overlap < max(4, int(min(selected_h, band_h) * 0.35)):
            continue
        score = overlap * 10 + band.get("span_w", 0)
        if score > best_score:
            best_score = score
            best_box = box
    return best_box or selected


def part_no_crop_has_text_above_barcode(crop_img):
    barcode_box = _part_no_primary_barcode_box(crop_img)
    if barcode_box is None:
        return False
    return _part_no_has_text_above_band(crop_img, barcode_box)


def _part_no_crop_text_neighbor_top(crop_img, barcode_box):
    if crop_img is None or barcode_box is None:
        return None
    h, w = crop_img.shape[:2]
    _x1, _y1, _x2, y2 = barcode_box
    barcode_h = max(1, y2 - _y1)
    start_y = min(h, int(y2 + max(4, barcode_h * 0.08)))
    if start_y >= h - 3:
        return None

    region = crop_img[start_y:, :]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    row_dark = dark.mean(axis=1)
    active_rows = row_dark > 0.018
    min_run = max(3, int(h * 0.018))
    start = None
    for i, active in enumerate(active_rows):
        if active and start is None:
            start = i
        at_end = i == len(active_rows) - 1
        if start is not None and ((not active) or at_end):
            end = i if not active else i + 1
            if end - start >= min_run:
                return start_y + start
            start = None
    return None


def trim_part_no_crop_before_lower_text_neighbor(crop_img):
    if crop_img is None:
        return None
    h, _w = crop_img.shape[:2]
    if h < 24:
        return crop_img
    barcode_box = _part_no_primary_barcode_box(crop_img)
    if barcode_box is None:
        return crop_img
    text_top = _part_no_crop_text_neighbor_top(crop_img, barcode_box)
    if text_top is None:
        return crop_img
    min_cutoff = barcode_box[3] + 2
    cutoff = text_top - max(2, int(h * 0.015))
    if cutoff < min_cutoff and min_cutoff <= text_top:
        cutoff = min_cutoff
    if barcode_box[3] < cutoff < h:
        trimmed = crop_img[:cutoff, :].copy()
        if part_no_crop_contains_1d_barcode(trimmed, min_span_ratio=0.16):
            return trimmed
    return crop_img


def part_no_crop_has_lower_neighbor_content(crop_img):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return False

    bands = _part_no_barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(8, int(h * 0.06)),
        min_span_w=max(60, int(w * 0.18)),
        row_trans_threshold=0.035,
        active_threshold=0.20,
    )
    if len(bands) >= 3:
        part_no_pair_bottom = bands[1]["box"][3] if len(bands) >= 2 else bands[0]["box"][3]
        min_gap = max(8, int(h * 0.035))
        for band in bands[2:]:
            if band["box"][1] - part_no_pair_bottom >= min_gap:
                return True
    elif len(bands) == 2:
        first = bands[0]["box"]
        second = bands[1]["box"]
        first_h = first[3] - first[1]
        if first_h >= max(36, int(h * 0.28)) and second[1] - first[3] >= max(8, int(h * 0.035)):
            return True
        if _part_no_two_band_has_lower_neighbor(crop_img, first, second):
            return True

    box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.18,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if box is None:
        return False

    return False


def trim_part_no_crop_before_lower_neighbor(crop_img, trim_text_neighbor=False):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return crop_img

    bands = _part_no_barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(8, int(h * 0.06)),
        min_span_w=max(60, int(w * 0.18)),
        row_trans_threshold=0.035,
        active_threshold=0.20,
    )
    if len(bands) < 3:
        if len(bands) == 2:
            first = bands[0]["box"]
            second = bands[1]["box"]
            first_h = first[3] - first[1]
            if first_h >= max(36, int(h * 0.28)) and second[1] - first[3] >= max(8, int(h * 0.035)):
                cutoff = max(first[3] + 2, second[1] - max(6, int(h * 0.025)))
                if first[3] + 2 <= cutoff < h:
                    return crop_img[:cutoff, :].copy()
            if _part_no_two_band_has_lower_neighbor(crop_img, first, second):
                cutoff = _part_no_cutoff_before_two_band_neighbor(crop_img, first, second)
                if first[3] + 2 <= cutoff < h:
                    return crop_img[:cutoff, :].copy()
        if trim_text_neighbor:
            return trim_part_no_crop_before_lower_text_neighbor(crop_img)
        return crop_img

    part_no_pair_bottom = bands[1]["box"][3]
    lower_top = bands[2]["box"][1]
    if lower_top - part_no_pair_bottom < max(8, int(h * 0.035)):
        if trim_text_neighbor:
            return trim_part_no_crop_before_lower_text_neighbor(crop_img)
        return crop_img

    cutoff = max(part_no_pair_bottom + 2, lower_top - max(6, int(h * 0.025)))
    min_h = max(24, part_no_pair_bottom + 2)
    if cutoff < min_h or cutoff >= h:
        if trim_text_neighbor:
            return trim_part_no_crop_before_lower_text_neighbor(crop_img)
        return crop_img
    return crop_img[:cutoff, :].copy()


def polish_part_no_crop_for_scan_miss(crop_img):
    if crop_img is None:
        return None
    trimmed = trim_part_no_crop_before_lower_neighbor(crop_img, trim_text_neighbor=True)
    return pad_part_no_crop_quiet_zone(trimmed)


def _white_band_is_clean(band):
    if band is None or band.size == 0:
        return True
    return (band[:, :, 0] < 128).mean() < 0.01


def _part_no_crop_embedded_prefix(original_crop, polished_crop):
    oh, ow = original_crop.shape[:2]
    ph, pw = polished_crop.shape[:2]
    if ph < 24 or pw < 80 or pw < ow:
        return None

    max_prefix_h = min(oh, ph)
    min_prefix_h = max(24, int(min(oh, ph) * 0.45))
    for x0 in range(0, pw - ow + 1):
        for prefix_h in range(max_prefix_h, min_prefix_h - 1, -1):
            if not np.array_equal(polished_crop[:prefix_h, x0 : x0 + ow], original_crop[:prefix_h, :]):
                continue
            left_ok = _white_band_is_clean(polished_crop[:prefix_h, :x0])
            right_ok = _white_band_is_clean(polished_crop[:prefix_h, x0 + ow :])
            bottom_ok = _white_band_is_clean(polished_crop[prefix_h:, :])
            if left_ok and right_ok and bottom_ok:
                return x0, prefix_h
    return None


def part_no_polished_crop_is_safe(original_crop, polished_crop):
    if original_crop is None or polished_crop is None:
        return False
    if polished_crop.shape[:2] == original_crop.shape[:2]:
        return False
    if part_no_crop_has_edge_clipped_barcode(original_crop):
        return False
    if part_no_crop_has_edge_clipped_barcode(polished_crop):
        return False
    if not part_no_crop_has_text_above_barcode(polished_crop):
        return False
    if part_no_crop_has_lower_neighbor_content(polished_crop):
        return False
    if not (
        part_no_crop_contains_1d_barcode(polished_crop, min_span_ratio=0.16)
        or part_no_crop_contains_1d_barcode(original_crop, min_span_ratio=0.16)
    ):
        return False
    return _part_no_crop_embedded_prefix(original_crop, polished_crop) is not None


def _label_crop_match_in_original(original_img, label_img, min_score=0.98):
    if original_img is None or label_img is None:
        return None
    oh, ow = original_img.shape[:2]
    lh, lw = label_img.shape[:2]
    if lh < 40 or lw < 80 or lh >= oh or lw >= ow:
        return None

    scale = min(1.0, 1800.0 / float(max(oh, ow)), 900.0 / float(max(lh, lw)))
    if scale < 1.0:
        resized_original = cv2.resize(
            original_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        resized_label = cv2.resize(
            label_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized_original = original_img
        resized_label = label_img

    if (
        resized_label.shape[0] >= resized_original.shape[0]
        or resized_label.shape[1] >= resized_original.shape[1]
        or min(resized_label.shape[:2]) < 20
    ):
        return None

    original_gray = cv2.cvtColor(resized_original, cv2.COLOR_BGR2GRAY)
    label_gray = cv2.cvtColor(resized_label, cv2.COLOR_BGR2GRAY)
    try:
        result = cv2.matchTemplate(original_gray, label_gray, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None
    _min_val, coarse_max_val, _min_loc, coarse_max_loc = cv2.minMaxLoc(result)
    if float(coarse_max_val) < 0.70:
        return None
    if result.shape[0] > 1 and result.shape[1] > 1:
        suppressed = result.copy()
        th, tw = label_gray.shape[:2]
        mx, my = coarse_max_loc
        sx1 = max(0, mx - max(1, tw // 2))
        sx2 = min(result.shape[1], mx + max(1, tw // 2) + 1)
        sy1 = max(0, my - max(1, th // 2))
        sy2 = min(result.shape[0], my + max(1, th // 2) + 1)
        suppressed[sy1:sy2, sx1:sx2] = -1.0
        _second_min, second_val, _second_min_loc, _second_loc = cv2.minMaxLoc(suppressed)
        if float(second_val) >= 0.90 and float(coarse_max_val) - float(second_val) < 0.01:
            return None

    coarse_x = int(round(float(coarse_max_loc[0]) / scale))
    coarse_y = int(round(float(coarse_max_loc[1]) / scale))
    refine_pad = max(12, int(max(lw, lh) * 0.04))
    rx1 = max(0, coarse_x - refine_pad)
    ry1 = max(0, coarse_y - refine_pad)
    rx2 = min(ow, coarse_x + lw + refine_pad)
    ry2 = min(oh, coarse_y + lh + refine_pad)
    if rx2 - rx1 <= lw or ry2 - ry1 <= lh:
        return None

    refine_region = original_img[ry1:ry2, rx1:rx2]
    refine_gray = cv2.cvtColor(refine_region, cv2.COLOR_BGR2GRAY)
    full_label_gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
    try:
        refine_result = cv2.matchTemplate(refine_gray, full_label_gray, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None
    _rmin, refine_max_val, _rmin_loc, refine_max_loc = cv2.minMaxLoc(refine_result)
    if float(refine_max_val) < min_score:
        return None

    x = rx1 + int(refine_max_loc[0])
    y = ry1 + int(refine_max_loc[1])
    return (
        max(0, min(ow - 1, x)),
        max(0, min(oh - 1, y)),
        lw,
        lh,
        float(refine_max_val),
    )


def tighten_recovered_part_no_crop_width(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 24 or w < 160:
        return crop_img

    barcode_box = _part_no_primary_barcode_box(crop_img) or crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.12,
        row_trans_threshold=0.05,
        active_threshold=0.20,
    )
    if barcode_box is None:
        return crop_img

    _bx1, by1, _bx2, by2 = barcode_box
    band = crop_img[max(0, by1) : min(h, by2), :]
    if band.size == 0:
        return crop_img

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw_img > 0
    col_dark = dark.mean(axis=0)
    active_idx = np.where(col_dark > 0.18)[0]
    if active_idx.size == 0:
        return crop_img

    gap_threshold = max(12, int(w * 0.025), int((by2 - by1) * 0.75))
    runs = []
    start = int(active_idx[0])
    prev = int(active_idx[0])
    for idx_value in active_idx[1:]:
        idx = int(idx_value)
        if idx - prev > gap_threshold:
            runs.append((start, prev + 1))
            start = idx
        prev = idx
    runs.append((start, prev + 1))

    clusters = []
    for x1, x2 in runs:
        span = x2 - x1
        if span < max(80, int(w * 0.10)):
            continue
        active = col_dark[x1:x2] > 0.18
        transitions = int(np.count_nonzero(active[1:] != active[:-1])) if active.size > 1 else 0
        if transitions < 8:
            continue
        clusters.append((x1, x2, span, transitions))
    if not clusters:
        return crop_img

    selected = None
    for cluster in clusters:
        center_x = (cluster[0] + cluster[1]) / 2.0
        if center_x <= w * 0.72:
            selected = cluster
            break
    if selected is None:
        selected = clusters[0]

    x1, x2, span, _transitions = selected
    out_x1 = max(0, x1 - max(44, int(span * 0.24), int(w * 0.035)))
    out_x2 = min(w, x2 + max(28, int(span * 0.18), int(w * 0.025)))
    if out_x2 - out_x1 >= w * 0.92:
        return crop_img
    return crop_img[:, out_x1:out_x2].copy()


def crop_recovered_part_no_from_extended_label(crop_img):
    part_crop, _kind, _ok = _stage2_crop_part_no(crop_img, None)
    if part_crop is None:
        return None
    part_crop = trim_part_no_crop_before_lower_neighbor(part_crop, trim_text_neighbor=True)
    part_crop = tighten_recovered_part_no_crop_width(part_crop)
    return pad_part_no_crop_quiet_zone(part_crop)


def recover_part_no_crop_from_original_context(label_img_path, label_img):
    base = os.path.splitext(os.path.basename(label_img_path))[0]
    original_path = original_path_for_label_id(base)
    if not original_path or not os.path.isfile(original_path):
        return None, [], []
    original_img = read_image(original_path)
    match = _label_crop_match_in_original(original_img, label_img)
    if match is None:
        return None, [], []

    x, y, w, h, _score = match
    oh, ow = original_img.shape[:2]
    pad_x = max(0, int(w * 0.015))
    x1 = max(0, x - pad_x)
    x2 = min(ow, x + w + pad_x)
    for up_ratio in (0.18, 0.25, 0.12, 0.35):
        y1 = max(0, y - int(h * up_ratio))
        y2 = min(oh, y + h + max(2, int(h * 0.03)))
        if x2 <= x1 or y2 <= y1:
            continue
        extended = original_img[y1:y2, x1:x2]
        part_crop = crop_recovered_part_no_from_extended_label(extended)
        if part_crop is None:
            continue
        raw_codes = decode_raw_part_no_crop(part_crop, label_id=f"{base}.original_context")
        part_no_codes = normalize_part_no_codes(raw_codes)
        if part_no_codes:
            return part_crop, raw_codes, part_no_codes
    return None, [], []


PART_NO_CODE_RE = re.compile(
    r"(?:^|[^0-9A-Z])(?:1P|P/N|PN|PART\s*NO\.?\s*[:：]?)?\s*((?:500|980)\d{5})(?=[^0-9A-Z]|[A-Z]{2,}|$)",
    re.I,
)


def normalize_part_no_codes(codes):
    out = []
    seen = set()
    for code in codes or []:
        for match in PART_NO_CODE_RE.finditer(str(code or "").upper()):
            part_no = match.group(1)
            if part_no in seen:
                continue
            seen.add(part_no)
            out.append(part_no)
    return out


def decode_raw_part_no_crop(crop_img, label_id=""):
    if crop_img is None:
        return []

    fd, tmp_path = tempfile.mkstemp(prefix="part_no_scan_", suffix=".png")
    os.close(fd)
    try:
        save_png_required(tmp_path, crop_img, "part no scan candidate")
        import scan2 as scan2_module

        codes = scan2_module.read_part_no_barcodes(tmp_path, allow_pixel_repair=True)
    except Exception as exc:
        tag = f"{label_id} " if label_id else ""
        _log(f"WARN: Part No扫码失败：{tag}{exc.__class__.__name__}: {exc}", "debug")
        return []
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    out = []
    seen = set()
    for code in codes or []:
        code = str(code).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def decode_part_no_crop(crop_img, label_id=""):
    return normalize_part_no_codes(decode_raw_part_no_crop(crop_img, label_id=label_id))


def decode_model_crop(crop_img, label_id=""):
    if crop_img is None:
        return []

    fd, tmp_path = tempfile.mkstemp(prefix="model_scan_", suffix=".png")
    os.close(fd)
    try:
        save_png_required(tmp_path, crop_img, "model scan candidate")
        import scan2 as scan2_module

        model, _raw = scan2_module.try_model_from_barcode(tmp_path)
    except Exception as exc:
        tag = f"{label_id} " if label_id else ""
        _log(f"WARN: Model扫码失败：{tag}{exc.__class__.__name__}: {exc}", "debug")
        return []
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    model = str(model or "").strip()
    return [model] if model else []


def model_crop_has_decodable_barcode(crop_img, label_id=""):
    return bool(decode_model_crop(crop_img, label_id=label_id))


def crop_part_no_field(img, return_box=False):
    if img is None:
        return (None, None) if return_box else None
    H, W = img.shape[:2]
    if H < 20 or W < 80:
        return (None, None) if return_box else None

    boxes = [
        (0, 0, int(W * 0.62), int(H * 0.48)),
        (0, 0, int(W * 0.72), int(H * 0.60)),
    ]
    barcode_box = barcode_like_box_in_region(
        img,
        0,
        0,
        int(W * 0.68),
        int(H * 0.55),
        min_run_h=max(8, int(H * 0.025)),
        min_span_w=max(70, int(W * 0.12)),
        row_trans_threshold=0.055,
        active_threshold=0.30,
    )
    if barcode_box is not None and barcode_box[1] < H * 0.35:
        bh = barcode_box[3] - barcode_box[1]
        boxes.insert(
            1,
            (
                max(0, int(barcode_box[0] - max(8, W * 0.025))),
                max(0, int(barcode_box[1] - max(16, bh * 1.35))),
                min(W, int(barcode_box[2] + max(10, W * 0.025))),
                min(H, int(barcode_box[3] + max(6, bh * 0.45))),
            )
        )

    fallback = None
    fallback_box = None
    for box in boxes:
        part_crop = crop_from_box(img, box)
        if part_crop is None:
            continue
        if fallback is None:
            fallback = part_crop
            fallback_box = box
        if part_no_crop_has_complete_1d_barcode(part_crop, min_span_ratio=0.18):
            return (part_crop, box) if return_box else part_crop
    return (fallback, fallback_box) if return_box else fallback


def _stage2_part_no_field_box(img, best_part_no_pred=None):
    if img is None:
        return None
    if best_part_no_pred is not None:
        detected_box = _stage2_part_no_box_from_pred(img, best_part_no_pred)
        if detected_box is not None:
            return detected_box
    _fallback_crop, fallback_box = crop_part_no_field(img, return_box=True)
    if fallback_box is None:
        return None
    return add_slant_guard_to_box(
        img,
        fallback_box,
        STAGE2_TEXT_SLANT_GUARD_MAX_PX,
        top=True,
        bottom=False,
    )


def fallback_model_crop_from_part_no(img, best_part_no_pred=None):
    if img is None:
        return None
    part_no_box = _stage2_part_no_field_box(img, best_part_no_pred)
    if part_no_box is None:
        return None

    H, W = img.shape[:2]
    px1, py1, px2, py2 = part_no_box
    part_h = max(1, py2 - py1)
    part_w = max(1, px2 - px1)

    x1 = max(0, int(px1 - max(12, W * 0.018)))
    x2 = min(W, int(px2 + max(18, W * 0.025)))
    y1 = max(0, int(py2 - max(2, part_h * 0.04)))
    y2 = min(H, int(py2 + max(64, part_h * 1.55)))
    if x2 <= x1 or y2 <= y1:
        return None

    target_y = min(y2 - 1, int(py2 + max(16, part_h * 0.32)))
    text_box = text_component_line_box_near_y(
        img,
        x1,
        x2,
        y1,
        y2,
        target_y=target_y,
        min_span_w=max(84, int(part_w * 0.18)),
        pad_top_min=5,
        pad_bottom_min=8,
        pad_y_ratio=0.14,
    )
    if text_box is None:
        return None

    text_pred = {
        "x": (text_box[0] + text_box[2]) / 2.0,
        "y": (text_box[1] + text_box[3]) / 2.0,
        "width": text_box[2] - text_box[0],
        "height": text_box[3] - text_box[1],
    }
    search_y1 = max(text_box[3], int(text_box[3] - max(2, (text_box[3] - text_box[1]) * 0.08)))
    search_y2 = min(
        H,
        max(search_y1 + 1, int(text_box[3] + max(42, part_h * 0.95))),
    )
    barcode_box = barcode_like_box_in_region(
        img,
        x1,
        search_y1,
        x2,
        search_y2,
        min_run_h=max(16, int(part_h * 0.20)),
        min_span_w=max(96, int((x2 - x1) * 0.28)),
        row_trans_threshold=0.04,
        active_threshold=0.24,
    )
    if barcode_box is None:
        barcode_box = barcode_like_box_below(img, text_pred)
    if barcode_box is not None:
        max_gap = max(12, int((text_box[3] - text_box[1]) * 0.85))
        if barcode_box[1] - text_box[3] > max_gap:
            barcode_box = None

    if barcode_box is None:
        text_only_box = add_slant_guard_to_box(img, text_box, STAGE2_SLANT_GUARD_MAX_PX)
        return trim_model_text_only_crop(crop_from_box(img, text_only_box))

    field_box = union_boxes(text_box, barcode_box)
    return crop_from_box(img, add_slant_guard_to_box(img, field_box, STAGE2_SLANT_GUARD_MAX_PX))


def _part_no_barcode_like_boxes_in_region(
    img,
    sx1,
    sy1,
    sx2,
    sy2,
    min_run_h,
    min_span_w,
    row_trans_threshold=0.035,
    active_threshold=0.20,
):
    H, W = img.shape[:2]
    sx1 = max(0, min(W, int(sx1)))
    sx2 = max(0, min(W, int(sx2)))
    sy1 = max(0, min(H, int(sy1)))
    sy2 = max(0, min(H, int(sy2)))
    if sx2 <= sx1 or sy2 <= sy1:
        return []

    roi = img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    if dark.shape[0] < 8 or dark.shape[1] < 80:
        return []

    row_dark = dark.mean(axis=1)
    row_trans = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1) / max(1, dark.shape[1] - 1)
    candidate_rows = (row_dark > 0.08) & (row_dark < 0.75) & (row_trans > row_trans_threshold)

    boxes = []
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            run_h = end - start
            if run_h >= min_run_h:
                band = dark[start:end]
                col_dark = band.mean(axis=0)
                active = col_dark > active_threshold
                active_idx = np.where(active)[0]
                if active_idx.size:
                    span_w = int(active_idx[-1] - active_idx[0] + 1)
                    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
                    aspect = span_w / float(max(run_h, 1))
                    if span_w >= min_span_w and transitions >= 18 and aspect >= 3.0:
                        boxes.append(
                            {
                                "box": (
                                    sx1 + int(active_idx[0]),
                                    sy1 + start,
                                    sx1 + int(active_idx[-1]) + 1,
                                    sy1 + end,
                                ),
                                "run_h": run_h,
                                "span_w": span_w,
                                "transitions": transitions,
                                "aspect": aspect,
                            }
                        )
            start = None

    return sorted(boxes, key=lambda item: (item["box"][1], -item["span_w"]))


def _part_no_field_box_from_text_and_barcode(img, text_box, barcode_box):
    H, W = img.shape[:2]
    field_box = union_boxes(text_box, barcode_box)
    if field_box is None:
        return None
    fx1, fy1, fx2, fy2 = field_box
    bw = barcode_box[2] - barcode_box[0]
    bh = barcode_box[3] - barcode_box[1]
    pad_left = max(44, int(bw * 0.42), int(W * 0.035))
    pad_right = max(24, int(bw * 0.22), int(W * 0.020))
    pad_top = max(4, int((text_box[3] - text_box[1]) * 0.22))
    pad_bottom = max(12, int(bh * 0.22))
    return (
        max(0, int(fx1 - pad_left)),
        max(0, int(fy1 - pad_top)),
        min(W, int(fx2 + pad_right)),
        min(H, int(fy2 + pad_bottom)),
    )


def _stage2_part_no_box_from_pred(img, best_part_no_pred):
    if best_part_no_pred is None:
        return None
    H, W = img.shape[:2]
    x = float(best_part_no_pred["x"])
    y = float(best_part_no_pred["y"])
    w = float(best_part_no_pred["width"])
    h = float(best_part_no_pred["height"])

    sx1 = max(0, int(x - w / 2 - max(w * 0.85, W * 0.035)))
    sx2 = min(W, int(x + w / 2 + max(w * 0.45, W * 0.030)))
    sy1 = max(0, int(y - h / 2 - max(h * 0.80, H * 0.025)))
    sy2 = min(H, int(y + h / 2 + max(h * 0.45, H * 0.025)))
    band_boxes = _part_no_barcode_like_boxes_in_region(
        img,
        sx1,
        sy1,
        sx2,
        sy2,
        min_run_h=max(8, int(h * 0.14)),
        min_span_w=max(60, int(w * 0.24)),
    )
    for idx in range(len(band_boxes) - 1):
        text_box = band_boxes[idx]["box"]
        barcode_box = band_boxes[idx + 1]["box"]
        gap = barcode_box[1] - text_box[3]
        if gap < 0 or gap > max(16, int(h * 0.30)):
            continue
        if barcode_box[3] > y + h * 0.75:
            continue
        field_box = _part_no_field_box_from_text_and_barcode(img, text_box, barcode_box)
        if field_box is not None:
            if idx + 2 < len(band_boxes):
                next_top = band_boxes[idx + 2]["box"][1]
                if next_top > barcode_box[3]:
                    field_box = (
                        field_box[0],
                        field_box[1],
                        field_box[2],
                        min(field_box[3], max(barcode_box[3] + 2, next_top - 3)),
                    )
            return add_slant_guard_to_box(img, field_box, STAGE2_TEXT_SLANT_GUARD_MAX_PX, top=True, bottom=False)

    # Keep the direct barcode search close to the detector box. If this reaches
    # the model row below, barcode readers may decode the model barcode instead.
    sy2 = min(H, int(y + h / 2 + max(h * 0.25, H * 0.015)))
    barcode_box = barcode_like_box_in_region(
        img,
        sx1,
        sy1,
        sx2,
        sy2,
        min_run_h=max(10, int(h * 0.20)),
        min_span_w=max(70, int(w * 0.38)),
        row_trans_threshold=0.07,
        active_threshold=0.32,
    )
    if barcode_box is None:
        return None

    bx1, by1, bx2, by2 = barcode_box
    bw = bx2 - bx1
    bh = by2 - by1
    text_box = text_component_line_box_near_y(
        img,
        max(0, int(bx1 - max(28, bw * 0.18))),
        min(W, int(bx2 + max(20, bw * 0.12))),
        max(0, int(by1 - max(48, h * 1.25, bh * 1.10))),
        min(H, max(int(by1 + max(4, h * 0.12)), by1 + 1)),
        target_y=by1 - max(10.0, h * 0.32),
        min_span_w=max(60, int(min(w, bw) * 0.22)),
        pad_top_min=3,
        pad_bottom_min=3,
        pad_y_ratio=0.16,
    )
    if text_box is not None:
        field_box = _part_no_field_box_from_text_and_barcode(img, text_box, barcode_box)
        if field_box is not None:
            return add_slant_guard_to_box(
                img,
                field_box,
                STAGE2_TEXT_SLANT_GUARD_MAX_PX,
                top=True,
                bottom=False,
            )

    box = (
        max(0, int(bx1 - max(44, bw * 0.42, W * 0.035))),
        max(0, int(by1 - max(22, bh * 1.35))),
        min(W, int(bx2 + max(24, bw * 0.22, W * 0.02))),
        min(H, int(by2 + max(6, bh * 0.22))),
    )
    return add_slant_guard_to_box(img, box, STAGE2_TEXT_SLANT_GUARD_MAX_PX, top=True, bottom=False)


def model_crop_satisfies_target(crop_img, label_img, model_pred=None):
    if not model_crop_is_usable(crop_img, label_img):
        return False
    if model_pred is None:
        expected_barcode = crop_contains_1d_barcode(crop_img, min_span_ratio=0.20)
    else:
        expected_barcode = model_barcode_box_near_pred(label_img, model_pred)
    if expected_barcode is None or expected_barcode is False:
        return True
    if not model_crop_has_text_above_barcode(crop_img):
        return False
    if not model_crop_has_decodable_barcode(crop_img):
        return False
    return True


def sn_barcode_box_near_pred(img, sn_pred):
    if sn_pred is None:
        return None
    H, W = img.shape[:2]
    x = float(sn_pred["x"])
    y = float(sn_pred["y"])
    w = float(sn_pred["width"])
    h = float(sn_pred["height"])
    pred_top = y - h / 2.0
    pred_bottom = y + h / 2.0
    sx1 = max(0, int(x - w * 0.85))
    sx2 = min(W, int(x + w * 1.15))
    sy1 = max(0, int(pred_top - h * 0.35))
    sy2 = min(H, int(pred_bottom + max(48.0, h * 0.85)))
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    candidates = _barcode_like_boxes_in_region(
        img,
        sx1,
        sy1,
        sx2,
        sy2,
        min_run_h=max(16, int(h * 0.18)),
        min_span_w=max(110, int(w * 0.38)),
        row_trans_threshold=0.12,
        active_threshold=0.28,
    )
    if not candidates:
        return None

    striped_candidates = []
    for box, score in candidates:
        for split_box in split_sn_barcode_segments(img, box):
            candidate_crop = crop_from_box(img, split_box)
            if _crop_has_sn_barcode_stripes(candidate_crop):
                area = (split_box[2] - split_box[0]) * (split_box[3] - split_box[1])
                striped_candidates.append((split_box, max(score, area)))
    candidates = striped_candidates
    if not candidates:
        return None

    min_plausible_top = max(0, int(pred_top - h * 0.10))
    plausible = [item for item in candidates if item[0][1] >= min_plausible_top]
    if not plausible:
        plausible = candidates
    # The SN barcode is the first one-dimensional band below the SN text.
    # Lower bands are usually EAN/UPC/PartNo neighbors and should not drive the crop.
    return min(plausible, key=lambda item: (item[0][1], item[0][0], -item[1]))[0]


def split_sn_barcode_segments(img, box):
    crop_img = crop_from_box(img, box)
    if crop_img is None:
        return []
    h, w = crop_img.shape[:2]
    if h < 18 or w < 80:
        return []
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0

    min_vertical_run = max(16, int(h * 0.22))
    stripe = np.zeros(w, dtype=bool)
    for x in range(w):
        best = 0
        current = 0
        for y in range(h):
            if dark[y, x]:
                current += 1
                best = max(best, current)
            else:
                current = 0
        stripe[x] = best >= min_vertical_run

    active_idx = np.where(stripe)[0]
    if active_idx.size == 0:
        return []

    gap_threshold = max(18, int(w * 0.035))
    segments = []
    start = int(active_idx[0])
    previous = int(active_idx[0])
    for value in active_idx[1:]:
        value = int(value)
        if value - previous > gap_threshold:
            segments.append((start, previous + 1))
            start = value
        previous = value
    segments.append((start, previous + 1))

    x1, y1, x2, y2 = box
    out = []
    for seg_x1, seg_x2 in segments:
        seg_w = seg_x2 - seg_x1
        if seg_w < max(80, int(w * 0.12)):
            continue
        seg_stripe = stripe[seg_x1:seg_x2]
        transitions = int(np.count_nonzero(seg_stripe[1:] != seg_stripe[:-1])) if len(seg_stripe) > 1 else 0
        if transitions < 18:
            continue
        pad_x = max(10, int(seg_w * 0.08))
        out.append(
            (
                max(0, x1 + seg_x1 - pad_x),
                y1,
                min(img.shape[1], x1 + seg_x2 + pad_x),
                y2,
            )
        )
    return out or [box]


def sn_text_box_above_barcode(img, sn_pred, barcode_box):
    if sn_pred is None or barcode_box is None:
        return None
    H, W = img.shape[:2]
    x = float(sn_pred["x"])
    y = float(sn_pred["y"])
    w = float(sn_pred["width"])
    h = float(sn_pred["height"])
    bx1, by1, bx2, _by2 = barcode_box
    bw = max(1, bx2 - bx1)

    x1 = max(0, int(min(x - w * 0.45, bx1 - max(18.0, bw * 0.08))))
    # Keep the text search tied to the selected SN barcode segment. Wide detector
    # boxes on AP-style labels can span into MAC/EAN neighbors on the same row.
    x2 = min(W, int(bx2 + max(24.0, bw * 0.08)))
    search_y1 = max(0, int(min(y - h / 2.0, by1 - max(52.0, h * 1.25))))
    search_y2 = min(H, max(search_y1 + 1, int(by1 + max(4.0, h * 0.12))))

    text_box = text_component_line_box_near_y(
        img,
        x1,
        x2,
        search_y1,
        search_y2,
        target_y=by1 - max(10.0, h * 0.30),
        min_span_w=max(70, int(min(w, bw) * 0.28)),
        pad_top_min=3,
        pad_bottom_min=3,
        pad_y_ratio=0.16,
    )
    if text_box is not None and (text_box[3] - text_box[1]) <= max(46, int(h * 0.85)):
        return add_slant_guard_to_box(img, text_box, STAGE2_TEXT_SLANT_GUARD_MAX_PX)

    fallback = (
        x1,
        max(0, int(y - h / 2.0 - h * 0.15)),
        x2,
        max(0, min(H, int(by1))),
    )
    if fallback[3] <= fallback[1]:
        fallback = (
            x1,
            max(0, int(by1 - max(42.0, h * 1.10))),
            x2,
            max(0, int(by1)),
        )
    return add_slant_guard_to_box(img, fallback, STAGE2_TEXT_SLANT_GUARD_MAX_PX)


def trim_sn_crop_to_selected_barcode(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 40 or w < 120:
        return crop_img

    candidates = []
    for box, score in _barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(14, int(h * 0.12)),
        min_span_w=max(90, int(w * 0.20)),
        row_trans_threshold=0.12,
        active_threshold=0.25,
    ):
        for split_box in split_sn_barcode_segments(crop_img, box):
            candidate_crop = crop_from_box(crop_img, split_box)
            if _crop_has_sn_barcode_stripes(candidate_crop):
                area = (split_box[2] - split_box[0]) * (split_box[3] - split_box[1])
                candidates.append((split_box, max(score, area)))
    if not candidates:
        return crop_img

    barcode_box = min(candidates, key=lambda item: (item[0][1], item[0][0], -item[1]))[0]
    bx1, by1, bx2, by2 = barcode_box
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    pad_x = max(16, int(bw * 0.08))
    pad_top = max(24, int(bh * 0.85))
    pad_bottom = max(8, int(bh * 0.24))

    search_x1 = max(0, bx1 - pad_x)
    search_x2 = min(w, bx2 + pad_x)
    search_y1 = max(0, by1 - max(58, int(bh * 1.35)))
    search_y2 = min(h, by1 + max(4, int(bh * 0.12)))
    text_box = text_component_line_box_near_y(
        crop_img,
        search_x1,
        search_x2,
        search_y1,
        search_y2,
        target_y=by1 - max(10, int(bh * 0.35)),
        min_span_w=max(50, int(bw * 0.25)),
        pad_top_min=3,
        pad_bottom_min=3,
        pad_y_ratio=0.16,
    )
    field_box = union_boxes(text_box, barcode_box)
    if field_box is None:
        field_box = (
            search_x1,
            max(0, by1 - pad_top),
            search_x2,
            min(h, by2 + pad_bottom),
        )
    else:
        field_box = (
            min(field_box[0], search_x1),
            min(field_box[1], max(0, by1 - pad_top)),
            max(field_box[2], search_x2),
            max(field_box[3], min(h, by2 + pad_bottom)),
        )

    refined = crop_from_box(crop_img, field_box)
    if refined is not None and _crop_has_sn_barcode_stripes(refined):
        return refined
    return crop_img


def normalize_sn_crop_for_barcode(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h <= 0 or w <= 0:
        return crop_img
    resized = cv2.resize(crop_img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(resized, 1.7, blurred, -0.7, 0)
    if sharpened.shape[1] >= 1300:
        return sharpened
    padded = cv2.copyMakeBorder(
        sharpened,
        48,
        48,
        48,
        48,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    blurred = cv2.GaussianBlur(padded, (0, 0), 1.1)
    return cv2.addWeighted(padded, 1.9, blurred, -0.9, 0)


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
    idxs = None
    try:
        if hasattr(cv2, "dnn") and hasattr(cv2.dnn, "NMSBoxes"):
            idxs = cv2.dnn.NMSBoxes(boxes, confs, min_conf, nms_thresh)
    except Exception:
        idxs = None
    if idxs is None or len(idxs) == 0:
        return [kept[i] for i in _manual_nms_indices(boxes, confs, nms_thresh)]
    idxs = idxs.flatten().tolist() if hasattr(idxs, "flatten") else list(idxs)
    return [kept[i] for i in idxs]


def _manual_nms_indices(boxes, confs, nms_thresh):
    order = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    selected = []
    while order:
        current = order.pop(0)
        selected.append(current)
        remaining = []
        for idx in order:
            if _box_iou_xywh(boxes[current], boxes[idx]) <= nms_thresh:
                remaining.append(idx)
        order = remaining
    return selected


def _box_iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in exts:
                images.append(os.path.join(dirpath, name))
    return sorted(images, key=lambda p: os.path.normcase(os.path.relpath(p, folder)))


def input_label_base(img_path):
    try:
        rel = os.path.relpath(os.path.abspath(img_path), os.path.abspath(INPUT_DIR))
    except ValueError:
        rel = os.path.basename(img_path)
    if rel.startswith(".."):
        rel = os.path.basename(img_path)

    rel = rel.replace("/", os.sep).replace("\\", os.sep)
    if os.sep not in rel:
        base = os.path.basename(img_path)
    else:
        safe_rel = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "__", rel).strip(" .")
        digest = hashlib.sha1(rel.encode("utf-8", errors="ignore")).hexdigest()[:8]
        base = f"{safe_rel}__{digest}"

    LABEL_ID_ORIGINAL_PATHS[base] = os.path.abspath(img_path)
    return base


def original_path_for_label_id(label_id):
    original_name = str(label_id or "").split("__label_", 1)[0]
    if not original_name:
        return ""
    mapped = LABEL_ID_ORIGINAL_PATHS.get(original_name)
    if mapped and os.path.isfile(mapped):
        return mapped
    candidate = os.path.join(INPUT_DIR, original_name)
    return candidate if os.path.isfile(candidate) else ""

# ==================== Inference Client ====================
client = None
client_backend = None
auto_fallback_backend = None
client_lock = threading.Lock()
client_local = threading.local()

def inference_backend():
    return os.environ.get("CROP_INFERENCE_BACKEND", "local").strip().lower() or "local"


def _is_local_inference_backend(backend):
    return backend in {"local", "onnx", "yolo", "local-yolo", "local_yolo"}


def _is_cloud_inference_backend(backend):
    return backend in {"roboflow", "cloud", "remote"}


def _is_auto_inference_backend(backend):
    return backend in {"auto", "roboflow-local", "cloud-local"}


def _local_yolo_cuda_available() -> bool:
    requested = os.environ.get("LOCAL_YOLO_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "-1", "none", "false", "off"}:
        return False
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return "CUDAExecutionProvider" in set(ort.get_available_providers())


def _auto_crop_worker_count(backend):
    cpu_count = os.cpu_count() or 2
    if _is_cloud_inference_backend(backend):
        return min(max(4, cpu_count), DEFAULT_CLOUD_MAX_WORKERS)
    if _local_yolo_cuda_available():
        return min(max(2, cpu_count // 2), DEFAULT_LOCAL_MAX_WORKERS)
    return 2


def _thread_client_cache():
    cache = getattr(client_local, "clients", None)
    if cache is None:
        cache = {}
        client_local.clients = cache
    return cache


def get_inference_client(backend=None):
    global client, client_backend, API_KEY
    backend = (backend or inference_backend()).strip().lower() or "local"
    if _is_auto_inference_backend(backend):
        backend = auto_fallback_backend or "roboflow"

    cache = _thread_client_cache()
    if backend in cache:
        client = cache[backend]
        client_backend = backend
        return client

    if _is_local_inference_backend(backend):
        from local_yolo import LocalYoloClient

        created = LocalYoloClient()
        cache[backend] = created
        client = created
        client_backend = backend
        return created

    if not _is_cloud_inference_backend(backend):
        raise RuntimeError(
            "Unknown CROP_INFERENCE_BACKEND. Use 'roboflow', 'local', or 'auto'."
        )

    if InferenceHTTPClient is None:
        raise RuntimeError(
            "inference-sdk is not installed. Use CROP_INFERENCE_BACKEND=local "
            "or install inference-sdk for Roboflow."
        )

    with client_lock:
        API_KEY = os.environ.get("API_KEY", API_KEY)
    if not API_KEY:
        raise RuntimeError("API_KEY is missing. Set it in .env or environment variables.")
    created = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)
    cache[backend] = created
    client = created
    client_backend = backend
    return created


def _inference_status_code(exc):
    cur = exc
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = getattr(cur, "status_code", None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                return status
        cur = getattr(cur, "inner_error", None) or getattr(cur, "__cause__", None)
    return None


def _strip_html_message(text):
    if not text:
        return ""
    text = str(text).strip()
    lower = text.lower()
    if "<!doctype html" in lower or "<html" in lower:
        return "server returned an HTML error page"
    return " ".join(text.split())


def _error_text_contains(exc, needles):
    cur = exc
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chunks = [
            getattr(cur, "api_message", ""),
            getattr(cur, "description", ""),
            str(cur),
        ]
        haystack = "\n".join(str(chunk).lower() for chunk in chunks if chunk)
        if any(needle in haystack for needle in needles):
            return True
        cur = getattr(cur, "inner_error", None) or getattr(cur, "__cause__", None)
    return False


def _is_cloudflare_block(exc):
    return (
        _inference_status_code(exc) == 403
        and _error_text_contains(exc, {"cloudflare", "you have been blocked", "enable cookies"})
    )


def _format_inference_error(exc, model_id, backend=None):
    status = _inference_status_code(exc)
    api_message = _strip_html_message(getattr(exc, "api_message", ""))
    description = _strip_html_message(getattr(exc, "description", "") or str(exc))
    selected_backend = backend or client_backend or inference_backend()

    if _is_local_inference_backend(selected_backend):
        detail = api_message or description
        return f"本地 YOLO 检测失败（model_id={model_id}）：{detail or type(exc).__name__}"

    if _is_cloudflare_block(exc):
        return (
            f"Roboflow 检测接口被 Cloudflare 拦截（HTTP 403，model_id={model_id}）。"
            "这是 Roboflow 网站/网络侧拦截，不是本地 OCR 或条码识别错误。"
            "可设置 CROP_INFERENCE_BACKEND=auto 自动切换本地 YOLO，"
            "或联系 Roboflow 确认当前 IP/API 访问是否被安全策略阻断。"
        )
    if status in {401, 403}:
        return (
            f"Roboflow 检测接口认证失败或无模型权限（HTTP {status}，model_id={model_id}）。"
            "请检查 .env / _internal\\.env 里的 API_KEY 是否为空、过期，"
            "并确认这个 key 有 huawei-2ha7t 和 sn_model 项目的访问权限。"
        )
    if status == 404:
        return f"Roboflow 检测模型不存在或版本号不对（HTTP 404，model_id={model_id}）。请检查 crop.py 里的 MODEL*_ID。"
    if status == 413:
        return f"Roboflow 检测图片过大（HTTP 413，model_id={model_id}）。"
    if status:
        detail = api_message or description
        suffix = f"：{detail}" if detail else ""
        return f"Roboflow 检测请求失败（HTTP {status}，model_id={model_id}）{suffix}"

    detail = api_message or description
    return f"Roboflow 检测请求失败（model_id={model_id}）：{detail or type(exc).__name__}"


def _is_retryable_inference_error(exc):
    status = _inference_status_code(exc)
    return status in {408, 413, 429, 500, 502, 503, 504} or status is None


def crop_worker_count(stage):
    stage_key = f"CROP_{str(stage).upper()}_WORKERS"
    raw = os.environ.get(stage_key) or os.environ.get("CROP_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            raise RuntimeError(f"{stage_key} / CROP_WORKERS must be a positive integer.")

    backend = inference_backend()
    if _is_auto_inference_backend(backend):
        return _auto_crop_worker_count("local")
    return _auto_crop_worker_count(backend)


def _map_ordered(items, fn, workers):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]

    results = [None] * len(items)
    max_workers = min(workers, len(items))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(fn, item): index
            for index, item in enumerate(items)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return results


def _infer_remote_or_local(tmp_path, model_id):
    global auto_fallback_backend
    backend = inference_backend()
    if not _is_auto_inference_backend(backend):
        return get_inference_client().infer(tmp_path, model_id=model_id)

    if auto_fallback_backend:
        return get_inference_client(auto_fallback_backend).infer(tmp_path, model_id=model_id)

    try:
        return get_inference_client("roboflow").infer(tmp_path, model_id=model_id)
    except Exception as exc:
        if not _is_cloudflare_block(exc):
            raise
        auto_fallback_backend = "local"
        _log(
            f"WARN: Roboflow Cloudflare 403 for {model_id}; using local YOLO for this run.",
            "warn",
        )
        return get_inference_client(auto_fallback_backend).infer(tmp_path, model_id=model_id)


def _can_use_local_original_path_inference(model_id):
    if model_id not in {MODEL1_ID, MODEL1_HARDCASE_SUPPLEMENT_ID}:
        return False
    if not stage1_local_direct_source_enabled():
        return False
    backend = inference_backend()
    if _is_local_inference_backend(backend):
        return True
    if _is_auto_inference_backend(backend):
        return auto_fallback_backend == "local"
    return False


def _infer_local_original_path_if_supported(original_img_path, model_id):
    if not _can_use_local_original_path_inference(model_id):
        return None
    if not original_img_path or not os.path.isfile(original_img_path):
        return None
    client = get_inference_client("local")
    if not hasattr(client, "supports_original_path_inference"):
        return None
    try:
        if not client.supports_original_path_inference(model_id=model_id):
            return None
    except Exception:
        return None
    res = client.infer_original_path(original_img_path, model_id=model_id)
    return (res.get("predictions", []) if isinstance(res, dict) else res) or []

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
    direct_preds = _infer_local_original_path_if_supported(original_img_path, model_id)
    if direct_preds is not None:
        return direct_preds

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

        os.makedirs(TMP_DIR, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(prefix="infer_", suffix=".jpg", dir=TMP_DIR, delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            if not _write_tmp_jpg(resized, tmp_path, quality=q):
                raise RuntimeError(f"Failed to write temporary inference image: {tmp_path}")
            res = _infer_remote_or_local(tmp_path, model_id)
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

            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return preds

        except Exception as e:
            last_err = e
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if not _is_retryable_inference_error(e):
                raise RuntimeError(_format_inference_error(e, model_id))
            # 继续下一轮降质量/降分辨率
            continue

    # 两次都失败，抛出最后一次错误
    raise RuntimeError(_format_inference_error(last_err, model_id))

# ==================== Stage 1：大图 -> 标签小图 ====================
def _stage1_prepare_product_label_crop(crop_img, allow_orientation_variants=False):
    if crop_img is None:
        return None

    rotations = (0, 90, 270, 180) if allow_orientation_variants else (0,)
    for rotation in rotations:
        candidate = crop_img if rotation == 0 else rotate_image(crop_img, rotation)
        tightened = stage1_tighten_label_crop(candidate)
        if tightened is not None and stage1_is_product_label_crop(tightened):
            return stage1_normalize_label_orientation(tightened)
        if stage1_is_product_label_crop(candidate):
            return stage1_normalize_label_orientation(candidate)
    return None


def _stage1_has_fallback_field_evidence(crop_img):
    if crop_img is None:
        return False
    preds = infer_with_resize(
        crop_img,
        "__stage1_fallback_field_check__.png",
        MODEL2_ID,
        max_side=1600,
    )
    _model_preds, part_no_preds, sn_preds = _stage2_parse_preds(preds)
    return bool(part_no_preds and sn_preds)


def _stage1_has_relaxed_single_label_field_evidence(crop_img):
    if crop_img is None:
        return False
    preds = infer_with_resize(
        crop_img,
        "__stage1_relaxed_single_label_field_check__.png",
        MODEL2_ID,
        max_side=1600,
    )
    model_preds, part_no_preds, sn_preds = _stage2_parse_preds(preds)
    return _stage1_field_layout_is_relaxed_plausible(crop_img, model_preds, part_no_preds, sn_preds)


def _stage1_has_single_label_field_evidence(crop_img):
    if crop_img is None:
        return False
    preds = infer_with_resize(
        crop_img,
        "__stage1_single_label_field_check__.png",
        MODEL2_ID,
        max_side=1600,
    )
    model_preds, part_no_preds, sn_preds = _stage2_parse_preds(preds)
    return _stage1_field_layout_is_plausible(crop_img, model_preds, part_no_preds, sn_preds)


def _stage1_has_relaxed_area_field_evidence(crop_img):
    if crop_img is None:
        return False
    if _stage1_has_single_label_field_evidence(crop_img):
        return True
    return _stage1_has_fallback_field_evidence(crop_img)


def _stage1_maybe_retry_upward_crop(
    img,
    pred,
    box,
    crop_img,
    allow_orientation_variants=False,
):
    if img is None or pred is None or box is None or crop_img is None:
        return box, crop_img

    box_h = max(1, box[3] - box[1])
    box_w = max(1, box[2] - box[0])
    expanded_box = box_from_pred_asym(
        img,
        pred,
        PADDING_1,
        PADDING_1 + 0.16,
        PADDING_1,
        slant_guard_max_px=STAGE1_SLANT_GUARD_MAX_PX,
    )
    if expanded_box is None:
        return box, crop_img
    if expanded_box[1] >= box[1] - max(10, int(box_h * 0.06)):
        return box, crop_img
    if expanded_box[2] - expanded_box[0] > int(box_w * 1.18):
        return box, crop_img

    expanded_crop = crop_from_box(img, expanded_box)
    if expanded_crop is None:
        return box, crop_img
    expanded_crop = _stage1_prepare_product_label_crop(
        expanded_crop,
        allow_orientation_variants=allow_orientation_variants,
    )
    if expanded_crop is None:
        return box, crop_img
    if not _stage1_has_single_label_field_evidence(expanded_crop):
        return box, crop_img

    current_single_ok = _stage1_has_single_label_field_evidence(crop_img)
    if current_single_ok and expanded_crop.shape[0] <= crop_img.shape[0] + max(12, int(crop_img.shape[0] * 0.08)):
        return box, crop_img
    return expanded_box, expanded_crop


def _stage1_maybe_retry_edge_recovery_crop(
    img,
    pred,
    box,
    crop_img,
    allow_orientation_variants=False,
):
    if img is None or pred is None or box is None or crop_img is None:
        return box, crop_img

    edge_ratios = _stage1_label_crop_edge_ink_ratios(crop_img)
    need_top = edge_ratios["top"] > 0.050
    need_left = edge_ratios["left"] > 0.110
    need_right = edge_ratios["right"] > 0.110
    if not (need_top or need_left or need_right):
        return box, crop_img

    box_h = max(1, box[3] - box[1])
    box_w = max(1, box[2] - box[0])
    expanded_box = box_from_pred_xy_asym(
        img,
        pred,
        PADDING_1 + (0.18 if need_left else 0.04),
        PADDING_1 + (0.12 if need_right else 0.04),
        PADDING_1 + (0.24 if need_top else 0.08),
        PADDING_1 + 0.08,
        slant_guard_max_px=STAGE1_SLANT_GUARD_MAX_PX,
    )
    if expanded_box is None:
        return box, crop_img
    if expanded_box[2] - expanded_box[0] > int(box_w * 1.24):
        return box, crop_img
    if expanded_box[3] - expanded_box[1] > int(box_h * 1.30):
        return box, crop_img
    if need_left and expanded_box[0] >= box[0] - max(10, int(box_w * 0.06)):
        return box, crop_img
    if need_right and expanded_box[2] <= box[2] + max(10, int(box_w * 0.06)):
        return box, crop_img
    if need_top and expanded_box[1] >= box[1] - max(10, int(box_h * 0.06)):
        return box, crop_img

    expanded_crop = crop_from_box(img, expanded_box)
    if expanded_crop is None:
        return box, crop_img
    expanded_crop = _stage1_prepare_product_label_crop(
        expanded_crop,
        allow_orientation_variants=allow_orientation_variants,
    )
    if expanded_crop is None:
        return box, crop_img
    expanded_single_ok = _stage1_has_single_label_field_evidence(expanded_crop)
    expanded_relaxed_ok = expanded_single_ok
    if not expanded_relaxed_ok and need_top:
        expanded_relaxed_ok = (
            _stage1_has_relaxed_single_label_field_evidence(expanded_crop)
            and stage1_is_partial_product_label_crop(expanded_crop)
        )
    if not expanded_relaxed_ok:
        return box, crop_img

    current_single_ok = _stage1_has_single_label_field_evidence(crop_img)
    current_relaxed_ok = current_single_ok
    if not current_relaxed_ok and need_top:
        current_relaxed_ok = _stage1_has_relaxed_area_field_evidence(crop_img)
    if current_single_ok:
        taller_gain = expanded_crop.shape[0] - crop_img.shape[0]
        wider_gain = expanded_crop.shape[1] - crop_img.shape[1]
        if need_top and taller_gain < max(12, int(crop_img.shape[0] * 0.07)):
            return box, crop_img
        if (need_left or need_right) and wider_gain < max(14, int(crop_img.shape[1] * 0.05)):
            return box, crop_img
    elif current_relaxed_ok and need_top:
        taller_gain = expanded_crop.shape[0] - crop_img.shape[0]
        if taller_gain < max(18, int(crop_img.shape[0] * 0.10)):
            return box, crop_img

    return expanded_box, expanded_crop


def _stage1_box_conflicts_with_existing(box, existing_boxes):
    if box is None:
        return False
    overlap_hits = 0
    box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    for existing_box in existing_boxes:
        if existing_box is None:
            continue
        if box_iou(box, existing_box) >= STAGE1_SUPPLEMENT_BOX_IOU:
            return True
        overlap_ratio = box_overlap_ratio(box, existing_box)
        if overlap_ratio >= 0.85:
            return True
        existing_area = max(1, (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1]))
        size_ratio = max(box_area, existing_area) / float(min(box_area, existing_area))
        if overlap_ratio >= 0.60 and size_ratio >= 1.35:
            return True
        if overlap_ratio >= 0.50 and size_ratio >= 1.80:
            return True
        if overlap_ratio >= 0.30:
            overlap_hits += 1
            if overlap_hits >= 2:
                return True
    return False


def _stage1_box_aspect(box):
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(1, y2 - y1)
    return float(width) / float(height)


def _stage1_should_reject_raw_edge_false_positive(box, image_shape):
    if box is None or image_shape is None:
        return False
    if not _stage1_box_touches_image_edge(box, image_shape):
        return False
    max_aspect = _env_float_default(
        "CROP_STAGE1_RAW_EDGE_FALSEPOS_MAX_ASPECT",
        STAGE1_RAW_EDGE_FALSEPOS_MAX_ASPECT,
    )
    return _stage1_box_aspect(box) < max_aspect


def _stage1_hardcase_model_available():
    if not _env_flag_default("CROP_STAGE1_HARDCASE_MODEL_SUPPLEMENT", True):
        return False
    try:
        from local_yolo import DEFAULT_MODEL_SPECS, model_ids_share_same_path
    except Exception:
        return False
    if model_ids_share_same_path(MODEL1_ID, MODEL1_HARDCASE_SUPPLEMENT_ID):
        return False
    spec = DEFAULT_MODEL_SPECS.get(MODEL1_HARDCASE_SUPPLEMENT_ID)
    return bool(spec and os.path.exists(spec.path))


def _stage1_collect_secondary_model_entries(img, existing_boxes, img_path=""):
    if not _stage1_hardcase_model_available():
        return [], 0

    secondary_preds = infer_with_resize(
        img,
        "__stage1_hardcase_model_supplement__.png",
        MODEL1_HARDCASE_SUPPLEMENT_ID,
    )
    label_preds = sorted(
        (
            p
            for p in (secondary_preds or [])
            if pred_class(p) == MODEL1_LABEL_CLASS
            and float(p.get("confidence", 0.0)) >= STAGE1_SUPPLEMENT_MIN_CONF
        ),
        key=lambda p: float(p.get("confidence", 0.0)),
        reverse=True,
    )

    accepted = []
    dropped = 0
    for candidate_index, p in enumerate(label_preds, start=1):
        box = box_from_pred(img, p, PADDING_1, slant_guard_max_px=STAGE1_SLANT_GUARD_MAX_PX)
        if _stage1_box_aspect(box) < STAGE1_PARTIAL_MIN_ASPECT:
            dropped += 1
            record_stage1_reject("aspect", img_path=img_path, index=candidate_index)
            continue
        if _stage1_box_conflicts_with_existing(box, existing_boxes):
            record_stage1_reject("hardcase_box_conflict", img_path=img_path, index=candidate_index)
            continue
        candidate_entries, candidate_dropped = _stage1_collect_product_label_entries(
            img,
            [p],
            allow_orientation_variants=True,
            require_single_label_field_evidence=True,
            max_crops=1,
            img_path=img_path,
        )
        dropped += candidate_dropped
        if not candidate_entries:
            continue
        entry = candidate_entries[0]
        accepted.append(entry)
        if entry.get("box") is not None:
            existing_boxes.append(entry["box"])
        if len(accepted) >= STAGE1_HARDCASE_SUPPLEMENT_MAX_CROPS:
            break
    return accepted, dropped


def _stage1_collect_product_label_entries(
    img,
    label_preds,
    allow_orientation_variants=False,
    require_field_evidence=False,
    require_single_label_field_evidence=False,
    enforce_relaxed_area_evidence=False,
    max_crops=None,
    img_path="",
):
    accepted = []
    dropped = 0
    accepted_boxes = []
    candidate_index = 0

    def _stage1_candidate_has_required_field_evidence(candidate_crop):
        if candidate_crop is None:
            return False
        if require_single_label_field_evidence:
            if _stage1_has_single_label_field_evidence(candidate_crop):
                return True
            return (
                stage1_is_partial_product_label_crop(candidate_crop)
                and _stage1_has_relaxed_single_label_field_evidence(candidate_crop)
            )
        if enforce_relaxed_area_evidence:
            return _stage1_has_relaxed_area_field_evidence(candidate_crop)
        if require_field_evidence:
            return _stage1_has_fallback_field_evidence(candidate_crop)
        return True

    for p in sorted(
        label_preds,
        key=lambda item: float((item or {}).get("confidence", 0.0)),
        reverse=True,
    ):
        candidate_index += 1
        box = box_from_pred(img, p, PADDING_1, slant_guard_max_px=STAGE1_SLANT_GUARD_MAX_PX)
        crop_img = crop_from_pred(img, p, PADDING_1, slant_guard_max_px=STAGE1_SLANT_GUARD_MAX_PX)
        if crop_img is None:
            record_stage1_reject("crop_failed", img_path=img_path, index=candidate_index)
            continue
        raw_crop_img = crop_img
        crop_img = _stage1_prepare_product_label_crop(
            crop_img,
            allow_orientation_variants=allow_orientation_variants,
        )
        if crop_img is None:
            dropped += 1
            record_stage1_reject("prepare_failed", raw_crop_img, img_path=img_path, index=candidate_index)
            continue
        box, crop_img = _stage1_maybe_retry_upward_crop(
            img,
            p,
            box,
            crop_img,
            allow_orientation_variants=allow_orientation_variants,
        )
        box, crop_img = _stage1_maybe_retry_edge_recovery_crop(
            img,
            p,
            box,
            crop_img,
            allow_orientation_variants=allow_orientation_variants,
        )
        if box is not None and img is not None:
            img_h, img_w = img.shape[:2]
            img_area = max(1, img_h * img_w)
            box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            box_area_ratio = box_area / float(img_area)
            if box_area_ratio < STAGE1_RELAXED_MIN_BOX_AREA_RATIO:
                dropped += 1
                record_stage1_reject("area_too_small_relaxed", crop_img, img_path=img_path, index=candidate_index)
                continue
            if (
                box_area_ratio < STAGE1_MIN_BOX_AREA_RATIO
                and not (
                    enforce_relaxed_area_evidence
                    or require_field_evidence
                    or require_single_label_field_evidence
                )
            ):
                dropped += 1
                record_stage1_reject("area_too_small", crop_img, img_path=img_path, index=candidate_index)
                continue
            if _stage1_box_touches_image_edge(box, img.shape):
                edge_single_ok = _stage1_has_single_label_field_evidence(crop_img)
                if not edge_single_ok and stage1_is_partial_product_label_crop(crop_img):
                    edge_single_ok = _stage1_has_relaxed_single_label_field_evidence(crop_img)
                if not edge_single_ok:
                    dropped += 1
                    record_stage1_reject("edge_touch_no_evidence", crop_img, img_path=img_path, index=candidate_index)
                    continue
        if not _stage1_candidate_has_required_field_evidence(crop_img):
            dropped += 1
            record_stage1_reject("no_field_evidence", crop_img, img_path=img_path, index=candidate_index)
            continue
        if _stage1_should_reject_blurry_small_background_label(
            box,
            crop_img,
            img.shape,
            confidence=float((p or {}).get("confidence", 0.0)),
            reference_box_areas=[
                max(0, accepted_box[2] - accepted_box[0]) * max(0, accepted_box[3] - accepted_box[1])
                for accepted_box in accepted_boxes
                if accepted_box is not None
            ],
        ):
            dropped += 1
            record_stage1_reject("blurry_background", crop_img, img_path=img_path, index=candidate_index)
            continue
        if _stage1_box_conflicts_with_existing(box, accepted_boxes):
            dropped += 1
            record_stage1_reject("box_conflict", crop_img, img_path=img_path, index=candidate_index)
            continue
        accepted.append(
            {
                "confidence": float(p.get("confidence", 1.0)),
                "crop": crop_img,
                "box": box,
            }
        )
        if box is not None:
            accepted_boxes.append(box)

    if max_crops is not None and len(accepted) > max_crops:
        accepted = sorted(accepted, key=lambda item: item["confidence"], reverse=True)[:max_crops]

    return accepted, dropped


def _stage1_collect_product_label_crops(
    img,
    label_preds,
    allow_orientation_variants=False,
    require_field_evidence=False,
    require_single_label_field_evidence=False,
    enforce_relaxed_area_evidence=False,
    max_crops=None,
    img_path="",
):
    accepted, dropped = _stage1_collect_product_label_entries(
        img,
        label_preds,
        allow_orientation_variants=allow_orientation_variants,
        require_field_evidence=require_field_evidence,
        require_single_label_field_evidence=require_single_label_field_evidence,
        enforce_relaxed_area_evidence=enforce_relaxed_area_evidence,
        max_crops=max_crops,
        img_path=img_path,
    )
    return [item["crop"] for item in accepted], dropped


def _stage1_collect_raw_label_entries(img, label_preds, img_path=""):
    accepted = []
    accepted_boxes = []
    candidate_index = 0
    for p in sorted(
        label_preds,
        key=lambda item: float((item or {}).get("confidence", 0.0)),
        reverse=True,
    ):
        candidate_index += 1
        box = box_from_pred(img, p, PADDING_1, slant_guard_max_px=STAGE1_SLANT_GUARD_MAX_PX)
        if box is None:
            record_stage1_reject("raw_box_failed", img_path=img_path, index=candidate_index)
            continue
        if _stage1_should_reject_raw_edge_false_positive(box, img.shape):
            debug_crop = crop_from_box(img, box) if LOG_LEVEL == "debug" else None
            record_stage1_reject("raw_edge_false_positive", debug_crop, img_path=img_path, index=candidate_index)
            continue
        if _stage1_box_conflicts_with_existing(box, accepted_boxes):
            debug_crop = crop_from_box(img, box) if LOG_LEVEL == "debug" else None
            record_stage1_reject("raw_box_conflict", debug_crop, img_path=img_path, index=candidate_index)
            continue
        crop_img = crop_from_box(img, box)
        if crop_img is None or crop_img.size == 0:
            record_stage1_reject("raw_crop_failed", img_path=img_path, index=candidate_index)
            continue
        accepted.append(
            {
                "confidence": float((p or {}).get("confidence", 1.0)),
                "crop": crop_img,
                "box": box,
            }
        )
        accepted_boxes.append(box)
    return accepted, 0


def _stage1_collect_raw_label_crops(img, raw_preds):
    label_preds = [p for p in (raw_preds or []) if pred_class(p) == MODEL1_LABEL_CLASS]
    accepted_entries, dropped = _stage1_collect_raw_label_entries(img, label_preds)
    return [item["crop"] for item in accepted_entries], dropped


def _stage1_collect_raw_entries_from_preds(img, raw_preds, img_path=""):
    label_preds = [p for p in (raw_preds or []) if pred_class(p) == MODEL1_LABEL_CLASS]
    return _stage1_collect_raw_label_entries(img, label_preds, img_path=img_path)


def _stage1_collect_label_entries(img, raw_preds, img_path=""):
    label_preds = [p for p in (raw_preds or []) if pred_class(p) == MODEL1_LABEL_CLASS]
    final_preds = nms(label_preds, MIN_CONF_1, NMS_1)

    accepted_entries, dropped = _stage1_collect_product_label_entries(
        img,
        final_preds,
        img_path=img_path,
    )
    if accepted_entries:
        existing_boxes = [item["box"] for item in accepted_entries if item.get("box") is not None]
        supplement_entries = []
        low_conf_label_preds = sorted(
            (
                p
                for p in label_preds
                if float(p.get("confidence", 0.0)) >= STAGE1_SUPPLEMENT_MIN_CONF
            ),
            key=lambda p: float(p.get("confidence", 0.0)),
            reverse=True,
        )
        for p in low_conf_label_preds:
            conf = float(p.get("confidence", 0.0))
            if conf >= MIN_CONF_1:
                continue
            box = box_from_pred(img, p, PADDING_1, slant_guard_max_px=STAGE1_SLANT_GUARD_MAX_PX)
            if _stage1_box_conflicts_with_existing(box, existing_boxes):
                continue
            supplement_candidate_entries, supplement_dropped = _stage1_collect_product_label_entries(
                img,
                [p],
                allow_orientation_variants=True,
                require_single_label_field_evidence=True,
                max_crops=1,
                img_path=img_path,
            )
            dropped += supplement_dropped
            if not supplement_candidate_entries:
                continue
            supplement_entry = supplement_candidate_entries[0]
            supplement_entries.append(supplement_entry)
            if supplement_entry.get("box") is not None:
                existing_boxes.append(supplement_entry["box"])
            if len(supplement_entries) >= STAGE1_FALLBACK_MAX_CROPS:
                break
        if supplement_entries:
            accepted_entries.extend(supplement_entries)
        secondary_entries, secondary_dropped = _stage1_collect_secondary_model_entries(img, existing_boxes, img_path=img_path)
        dropped += secondary_dropped
        if secondary_entries:
            accepted_entries.extend(secondary_entries)
        return accepted_entries, dropped

    fallback_preds = nms(label_preds, MIN_CONF_1_FALLBACK, NMS_1)
    fallback_entries, fallback_dropped = _stage1_collect_product_label_entries(
        img,
        fallback_preds,
        allow_orientation_variants=True,
        require_field_evidence=True,
        max_crops=STAGE1_FALLBACK_MAX_CROPS,
        img_path=img_path,
    )
    if fallback_entries:
        return fallback_entries, fallback_dropped
    return [], dropped


def _stage1_collect_label_crops(img, raw_preds):
    accepted_entries, dropped = _stage1_collect_label_entries(img, raw_preds)
    return [item["crop"] for item in accepted_entries], dropped


def stage1_crop_labels(img_path):
    img = read_image(img_path)
    if img is None:
        _log(f"❌ 读图失败: {img_path}", "error")
        return []

    # 先按原图方向推理；若过滤后没有产品标签，再用旋转后的整图重试 stage1。
    preds = infer_with_resize(img, img_path, model_id=MODEL1_ID)
    entry_collector = _stage1_collect_raw_entries_from_preds if stage1_raw_label_mode_enabled() else _stage1_collect_label_entries
    label_entries, dropped = entry_collector(img, preds, img_path=img_path)
    stage1_rotation = 0
    preview_img = img

    if not label_entries and stage1_rotation_retry_enabled():
        for rotation in stage1_rotation_retry_angles():
            rotated_img = stage1_rotated_image(img, rotation)
            rotated_preds = infer_with_resize(rotated_img, img_path, model_id=MODEL1_ID)
            label_entries, rotated_dropped = entry_collector(rotated_img, rotated_preds, img_path=img_path)
            dropped += rotated_dropped
            if label_entries:
                stage1_rotation = rotation
                preview_img = rotated_img
                break

    base = input_label_base(img_path)
    out_paths = []

    for entry in label_entries:
        crop = entry.get("crop")
        if crop is None:
            continue
        out_name = f"{base}__label_{len(out_paths) + 1}.png"
        out_path = os.path.join(STAGE1_DIR, out_name)
        out_paths.append(save_png_required(out_path, crop, "stage1 label crop"))
    _stage1_save_preview_image(preview_img, img_path, label_entries, rotation_deg=stage1_rotation)

    _log(
        f"标签裁剪：{os.path.basename(img_path)} -> {len(out_paths)} 个标签小图"
        + (f"（stage1 方向重试 {stage1_rotation}°）" if stage1_rotation else ""),
        "info",
    )
    return out_paths


def _remove_stage1_orphan_label(label_path):
    try:
        stage1_root = os.path.abspath(STAGE1_DIR)
        target = os.path.abspath(label_path)
        if os.path.commonpath([stage1_root, target]) != stage1_root:
            return False
        if os.path.isfile(target):
            os.remove(target)
            return True
    except Exception as e:
        _log(f"警告：删除无效 Stage1 标签裁剪失败 {label_path}: {e}", "warn")
    return False

# ==================== Stage 2：标签小图 -> model/sn 小块 ====================
def _stage2_parse_preds(raw_preds):
    by_cls = {
        MODEL2_MODEL_CLASS: [],
        MODEL2_PART_NO_CLASS: [],
        MODEL2_SN_CLASS: [],
    }
    for p in raw_preds or []:
        if not isinstance(p, dict):
            continue
        if "x" not in p:
            continue
        c = pred_class(p)
        if c in by_cls:
            by_cls[c].append(p)
    return (
        by_cls[MODEL2_MODEL_CLASS],
        by_cls[MODEL2_PART_NO_CLASS],
        by_cls[MODEL2_SN_CLASS],
    )


def _stage2_best_pred(pred_list):
    if not pred_list:
        return None
    return max(pred_list, key=lambda x: float(x.get("confidence", 1.0)))


def _stage2_infer_field_preds(img, label_img_path, debug_suffix=""):
    preds1 = infer_with_resize(img, label_img_path, MODEL2_ID, max_side=1600)

    bn = os.path.basename(label_img_path)
    if bn in {"1__label_6.png", "2__label_6.png"} or bn.startswith("2dc0ee5e") or bn.startswith("358d508d"):
        _log(
            f"DEBUG {bn}{debug_suffix} RAW PREDS (R1): "
            f"{json.dumps(preds1, ensure_ascii=False)[:2000]}",
            "debug",
        )

    model_preds, part_no_preds, sn_preds = _stage2_parse_preds(preds1)
    model_required = stage2_save_model_crops_enabled()
    if part_no_preds and sn_preds and (model_preds or not model_required):
        return preds1 or []

    preds2 = infer_with_resize(img, label_img_path, MODEL2_ID, max_side=2048)
    return (preds1 or []) + (preds2 or [])


def _stage2_crop_sn(img, best_sn_pred):
    if best_sn_pred is None:
        return None
    barcode_box = sn_barcode_box_near_pred(img, best_sn_pred)
    base_box = box_from_pred_asym(
        img,
        best_sn_pred,
        max(0.12, PADDING_2_SN_X),
        max(0.12, PADDING_2_SN_TOP),
        max(0.70, PADDING_2_SN_BOTTOM),
        slant_guard_max_px=STAGE2_SLANT_GUARD_MAX_PX,
    )
    if barcode_box is None:
        crop_s = crop_from_box(img, base_box)
        if _crop_has_sn_barcode_stripes(crop_s):
            return normalize_sn_crop_for_barcode(trim_sn_crop_to_selected_barcode(crop_s))
        return None

    candidate_boxes = []
    text_box = sn_text_box_above_barcode(img, best_sn_pred, barcode_box)
    field_box = union_boxes(text_box, barcode_box)
    if field_box is not None:
        bw = barcode_box[2] - barcode_box[0]
        bh = barcode_box[3] - barcode_box[1]
        for pad_x, pad_y in (
            (max(12, int(bw * 0.08)), max(5, int(bh * 0.16))),
            (max(20, int(bw * 0.10)), max(7, int(bh * 0.22))),
        ):
            candidate_boxes.append(expand_box_pixels(img, field_box, pad_x=pad_x, pad_y=pad_y))
    candidate_boxes.append(base_box)
    candidate_boxes.append(
        box_from_pred_asym(
            img,
            best_sn_pred,
            0.28,
            0.15,
            0.85,
            slant_guard_max_px=STAGE2_SLANT_GUARD_MAX_PX,
        )
    )

    fallback_crop = None
    for field_box in candidate_boxes:
        crop_s = crop_from_box(img, field_box)
        if crop_has_complete_1d_barcode(
            crop_s,
            min_span_ratio=0.18,
            edge_guard_px=4,
            row_trans_threshold=0.12,
            active_threshold=0.25,
        ):
            return normalize_sn_crop_for_barcode(trim_sn_crop_to_selected_barcode(crop_s))
        if fallback_crop is None:
            if crop_contains_1d_barcode(
                crop_s,
                min_span_ratio=0.16,
                row_trans_threshold=0.12,
                active_threshold=0.25,
            ) or _crop_has_sn_barcode_stripes(crop_s):
                fallback_crop = normalize_sn_crop_for_barcode(trim_sn_crop_to_selected_barcode(crop_s))
    return fallback_crop


def _stage2_crop_part_no_from_pred(img, best_part_no_pred):
    if best_part_no_pred is None:
        return None

    barcode_box = _stage2_part_no_box_from_pred(img, best_part_no_pred)
    candidates = [trim_part_no_crop_before_lower_neighbor(crop_from_box(img, barcode_box))]

    base_box = box_from_pred_asym(
        img,
        best_part_no_pred,
        0.08,
        0.18,
        0.28,
        slant_guard_max_px=STAGE2_SLANT_GUARD_MAX_PX,
    )
    candidates.append(trim_part_no_crop_before_lower_neighbor(crop_from_box(img, base_box)))

    wide_box = box_from_pred_asym(
        img,
        best_part_no_pred,
        0.14,
        0.28,
        0.42,
        slant_guard_max_px=STAGE2_SLANT_GUARD_MAX_PX,
    )
    candidates.append(trim_part_no_crop_before_lower_neighbor(crop_from_box(img, wide_box)))

    for crop_p in candidates:
        if (
            part_no_crop_has_complete_1d_barcode(crop_p, min_span_ratio=0.18)
            and not part_no_crop_has_lower_neighbor_content(crop_p)
        ):
            return crop_p

    for crop_p in candidates:
        if (
            part_no_crop_contains_1d_barcode(crop_p, min_span_ratio=0.16)
            and not part_no_crop_has_lower_neighbor_content(crop_p)
        ):
            return crop_p

    for crop_p in candidates:
        if part_no_crop_has_complete_1d_barcode(crop_p, min_span_ratio=0.18):
            return crop_p

    for crop_p in candidates:
        if part_no_crop_contains_1d_barcode(crop_p, min_span_ratio=0.16):
            return crop_p

    for crop_p in candidates:
        if crop_p is not None:
            return crop_p
    return None


def _stage2_crop_part_no(img, best_part_no_pred):
    detected = _stage2_crop_part_no_from_pred(img, best_part_no_pred)
    detected = trim_part_no_crop_before_lower_neighbor(detected)
    detected_kind = "detector" if detected is not None else ""
    if detected is not None:
        ok = part_no_crop_has_complete_1d_barcode(detected, min_span_ratio=0.18)
        if part_no_crop_is_structurally_valid(detected):
            if ok:
                return detected, "detector", True
            if part_no_crop_contains_1d_barcode(detected, min_span_ratio=0.16):
                detected_kind = "detector"
        if (
            part_no_crop_has_text_above_barcode(detected)
            and not part_no_crop_has_lower_neighbor_content(detected)
            and part_no_crop_contains_1d_barcode(detected, min_span_ratio=0.12)
        ):
            padded_detected = pad_part_no_crop_quiet_zone(detected)
            if (
                padded_detected is not None
                and not part_no_crop_has_lower_neighbor_content(padded_detected)
                and part_no_crop_contains_1d_barcode(padded_detected, min_span_ratio=0.12)
            ):
                padded_ok = part_no_crop_has_complete_1d_barcode(padded_detected, min_span_ratio=0.18)
                if padded_ok:
                    return padded_detected, "detector_padded", True
                detected = padded_detected
                detected_kind = "detector_padded"
        if (
            part_no_crop_has_text_above_barcode(detected)
            and not part_no_crop_has_edge_clipped_barcode(detected)
            and
            part_no_crop_has_partial_1d_barcode(detected, min_span_ratio=0.20)
            and not part_no_crop_has_lower_neighbor_content(detected)
        ):
            partial_detected = pad_part_no_crop_quiet_zone(detected)
            if partial_detected is not None:
                detected = partial_detected
                detected_kind = "detector_partial"
        if best_part_no_pred is not None and not part_no_crop_has_lower_neighbor_content(detected):
            h, w = detected.shape[:2]
            if (
                h >= 40
                and w >= 120
                and part_no_crop_has_text_above_barcode(detected)
                and not part_no_crop_has_edge_clipped_barcode(detected)
                and part_no_crop_contains_1d_barcode(detected, min_span_ratio=0.12)
            ):
                detected_kind = "detector_field"

    fallback = crop_part_no_field(img)
    fallback = trim_part_no_crop_before_lower_neighbor(fallback)
    fallback_ok = fallback is not None and part_no_crop_has_complete_1d_barcode(
        fallback,
        min_span_ratio=0.18,
    )
    if fallback is not None and part_no_crop_is_structurally_valid(fallback):
        return fallback, "heuristic", fallback_ok
    if (
        fallback is not None
        and part_no_crop_has_text_above_barcode(fallback)
        and not part_no_crop_has_lower_neighbor_content(fallback)
        and part_no_crop_contains_1d_barcode(fallback, min_span_ratio=0.12)
    ):
        padded_fallback = pad_part_no_crop_quiet_zone(fallback)
        if (
            padded_fallback is not None
            and not part_no_crop_has_lower_neighbor_content(padded_fallback)
            and part_no_crop_contains_1d_barcode(padded_fallback, min_span_ratio=0.12)
        ):
            padded_ok = part_no_crop_has_complete_1d_barcode(padded_fallback, min_span_ratio=0.18)
            return padded_fallback, "heuristic_padded", padded_ok
    if (
        best_part_no_pred is None
        and fallback is not None
        and part_no_crop_has_text_above_barcode(fallback)
        and not part_no_crop_has_lower_neighbor_content(fallback)
        and part_no_crop_contains_1d_barcode(fallback, min_span_ratio=0.12)
    ):
        return fallback, "heuristic_original_context", fallback_ok

    return detected, detected_kind, False


def _stage2_build_candidate(img, raw_preds, rotation):
    model_preds, part_no_preds, sn_preds = _stage2_parse_preds(raw_preds)
    best_model_pred = _stage2_best_pred(model_preds)
    best_part_no_pred = _stage2_best_pred(part_no_preds)
    best_sn_pred = _stage2_best_pred(sn_preds)
    part_no_crop, part_no_kind, _part_no_visual_ok = _stage2_crop_part_no(img, best_part_no_pred)
    part_no_raw_codes = decode_raw_part_no_crop(part_no_crop)
    part_no_codes = normalize_part_no_codes(part_no_raw_codes)
    if part_no_crop is not None and not part_no_codes:
        polished_part_no_crop = polish_part_no_crop_for_scan_miss(part_no_crop)
        if part_no_polished_crop_is_safe(part_no_crop, polished_part_no_crop):
            polished_raw_codes = decode_raw_part_no_crop(polished_part_no_crop)
            polished_codes = normalize_part_no_codes(polished_raw_codes)
            if polished_codes or not part_no_raw_codes:
                part_no_crop = polished_part_no_crop
                part_no_raw_codes = polished_raw_codes
                part_no_codes = polished_codes
    part_no_ok = bool(part_no_codes)
    model_required = stage2_save_model_crops_enabled()

    model_crop = None
    model_kind = ""
    model_conf = None
    if model_required and best_model_pred is not None:
        direct_crop = crop_model_field(img, best_model_pred)
        if model_crop_satisfies_target(direct_crop, img, best_model_pred):
            model_crop = direct_crop
            model_kind = "direct"
            model_conf = float(best_model_pred.get("confidence", 0))

    if model_required and model_crop is None:
        fallback_crop = fallback_model_crop_from_part_no(img, best_part_no_pred)
        if model_crop_satisfies_target(fallback_crop, img, None):
            model_crop = fallback_crop
            model_kind = "fallback_from_part_no"
            model_conf = 0.0

    if model_required and model_crop is None and best_sn_pred is not None:
        fallback_crop = fallback_model_crop_from_sn(img, best_sn_pred)
        if model_crop_satisfies_target(fallback_crop, img, None):
            model_crop = fallback_crop
            model_kind = "fallback_from_sn"
            model_conf = 0.0

    sn_crop = _stage2_crop_sn(img, best_sn_pred)
    sn_conf = float(best_sn_pred.get("confidence", 0)) if best_sn_pred is not None else None
    candidate = {
        "rotation": rotation,
        "raw_preds": raw_preds or [],
        "best_model_pred": best_model_pred,
        "best_part_no_pred": best_part_no_pred,
        "best_sn_pred": best_sn_pred,
        "model_crop": model_crop,
        "model_kind": model_kind,
        "model_conf": model_conf,
        "sn_crop": sn_crop,
        "sn_conf": sn_conf,
        "part_no_crop": part_no_crop,
        "part_no_kind": part_no_kind,
        "part_no_conf": float(best_part_no_pred.get("confidence", 0)) if best_part_no_pred is not None else None,
        "part_no_ok": part_no_ok,
        "part_no_raw_codes": part_no_raw_codes,
        "part_no_codes": part_no_codes,
        "model_required": model_required,
        "label_shape": img.shape[:2],
    }
    candidate["score"] = _stage2_candidate_score(candidate)
    return candidate


def _stage2_layout_score(candidate):
    score = 0.0
    part_no_pred = candidate.get("best_part_no_pred")
    model_pred = candidate.get("best_model_pred")
    sn_pred = candidate.get("best_sn_pred")
    label_shape = candidate.get("label_shape") or (0, 0)
    label_h = float(label_shape[0] or 0)

    if part_no_pred is not None and label_h > 0:
        part_y = float(part_no_pred.get("y", 0.0))
        if part_y <= label_h * 0.46:
            score += 45.0
        else:
            score -= 160.0

    if part_no_pred is not None and model_pred is not None:
        if float(part_no_pred.get("y", 0.0)) < float(model_pred.get("y", 0.0)):
            score += 55.0
        else:
            score -= 95.0

    if part_no_pred is not None and sn_pred is not None:
        if float(part_no_pred.get("y", 0.0)) < float(sn_pred.get("y", 0.0)):
            score += 75.0
        else:
            score -= 130.0

    if model_pred is not None and sn_pred is not None:
        if float(model_pred.get("y", 0.0)) < float(sn_pred.get("y", 0.0)):
            score += 35.0
        else:
            score -= 55.0

    return score


def _stage2_candidate_score(candidate):
    score = 0.0
    score += _stage2_layout_score(candidate)
    model_crop = candidate.get("model_crop")
    if model_crop is not None:
        h, w = model_crop.shape[:2]
        if candidate.get("model_kind") == "direct":
            score += 700.0
        else:
            score += 350.0
        score += min(w, 600) / 10.0
        score += min(h, 160) / 20.0
        score += float(candidate.get("model_conf") or 0.0) * 100.0
    elif candidate.get("model_required"):
        score -= 500.0

    if candidate.get("sn_crop") is not None:
        score += 500.0 + float(candidate.get("sn_conf") or 0.0) * 100.0
    elif candidate.get("best_sn_pred") is not None:
        score += 50.0 + float(candidate.get("sn_conf") or 0.0) * 20.0

    if candidate.get("part_no_ok"):
        score += 300.0
        score += float(candidate.get("part_no_conf") or 0.0) * 80.0
        if candidate.get("part_no_kind") == "detector":
            score += 120.0
    elif candidate.get("part_no_crop") is not None:
        score -= 90.0
        if candidate.get("part_no_kind") == "detector":
            score -= 35.0

    if candidate.get("rotation") == 180:
        score -= 1.0
    return score


def _stage2_should_retry_rot180(candidate):
    if os.environ.get("CROP_STAGE2_ROTATION_RETRY", "1").strip().lower() in {"0", "false", "no"}:
        return False
    if not candidate.get("model_required", True):
        if not candidate.get("part_no_ok"):
            return True
        if candidate.get("sn_crop") is None:
            sn_conf = candidate.get("sn_conf")
            return sn_conf is None or float(sn_conf or 0.0) < 0.55
        return False
    if candidate.get("model_crop") is None:
        return True
    model_kind = candidate.get("model_kind")
    sn_conf = candidate.get("sn_conf")
    model_conf = candidate.get("model_conf")
    if model_kind == "direct":
        if not candidate.get("part_no_ok") and candidate.get("sn_crop") is None:
            return True
        if candidate.get("sn_crop") is None and float(model_conf or 0.0) < 0.70:
            return True
        return False
    if model_kind == "fallback_from_sn":
        return sn_conf is None or float(sn_conf) < 0.55 or not candidate.get("part_no_ok")
    return False


def stage2_crop_fields(label_img_path):
    img = read_image(label_img_path)
    if img is None:
        return None

    preds = _stage2_infer_field_preds(img, label_img_path)
    candidates = [_stage2_build_candidate(img, preds, rotation=0)]

    if _stage2_should_retry_rot180(candidates[0]):
        rotated_img = rotate_image(img, 180)
        rotated_preds = _stage2_infer_field_preds(rotated_img, label_img_path, debug_suffix=" ROT180")
        candidates.append(_stage2_build_candidate(rotated_img, rotated_preds, rotation=180))

    selected = max(candidates, key=lambda item: item["score"])

    base = os.path.splitext(os.path.basename(label_img_path))[0]
    out = {
        "label_id": base,
        "label_crop": label_img_path,
        "original_image_path": original_path_for_label_id(base),
        "model_path": None,
        "sn_path": None,
        "part_no_path": None,
        "model_conf": None,
        "sn_conf": None,
        "part_no_conf": None,
        "part_no": "",
        "part_no_codes": [],
        "stage2_rotation": selected["rotation"],
        "model_crop_source": selected["model_kind"],
        "part_no_crop_source": selected.get("part_no_kind", ""),
    }

    part_no_crop = selected.get("part_no_crop")
    part_no_codes = list(selected.get("part_no_codes") or [])
    part_no_raw_codes = list(selected.get("part_no_raw_codes") or [])
    if part_no_crop is not None and not part_no_raw_codes and not part_no_codes:
        part_no_raw_codes = decode_raw_part_no_crop(part_no_crop, label_id=base)
        part_no_codes = normalize_part_no_codes(part_no_raw_codes)
    if not part_no_codes:
        recovered_crop, recovered_raw_codes, recovered_codes = recover_part_no_crop_from_original_context(
            label_img_path,
            img,
        )
        if recovered_codes:
            part_no_crop = recovered_crop
            part_no_raw_codes = recovered_raw_codes
            part_no_codes = recovered_codes
            out["part_no_crop_source"] = "original_context"
    if (
        not part_no_codes
        and selected.get("sn_crop") is None
        and selected.get("model_crop") is None
    ):
        fail_path = os.path.join(FAILED_DIR, f"{base}__FAILED.png")
        save_png_required(fail_path, img, "failed label crop")
        return None
    should_save_part_no_crop = part_no_crop is not None and (
        bool(part_no_codes)
        or (
            not part_no_raw_codes
            and part_no_crop_is_structurally_valid(
                part_no_crop,
                allow_partial=selected.get("part_no_kind") == "detector_partial",
            )
        )
    )
    if should_save_part_no_crop:
        pp = os.path.join(OUT_PART_NO_DIR, f"{base}__part_no.png")
        save_png_required(pp, part_no_crop, "part no crop")
        out["part_no_path"] = pp
        out["part_no_conf"] = selected.get("part_no_conf")
        out["part_no_codes"] = part_no_codes
        out["part_no"] = part_no_codes[0] if part_no_codes else ""

    if stage2_save_model_crops_enabled() and selected["model_crop"] is not None:
        mp = os.path.join(OUT_MODEL_DIR, f"{base}__model.png")
        save_png_required(mp, selected["model_crop"], "model crop")
        out["model_path"] = mp
        out["model_conf"] = selected["model_conf"]

    if selected["sn_crop"] is not None:
        sp = os.path.join(OUT_SN_DIR, f"{base}__sn.png")
        save_png_required(sp, selected["sn_crop"], "sn crop")
        out["sn_path"] = sp
        out["sn_conf"] = selected["sn_conf"]

    if not out["model_path"] and not out["sn_path"] and not out["part_no_path"]:
        fail_path = os.path.join(FAILED_DIR, f"{base}__FAILED.png")
        save_png_required(fail_path, img, "failed label crop")
        return None

    return out


def _stage2_build_model_only_candidate(img, raw_preds, rotation):
    model_preds, part_no_preds, sn_preds = _stage2_parse_preds(raw_preds)
    best_model_pred = _stage2_best_pred(model_preds)
    best_part_no_pred = _stage2_best_pred(part_no_preds)
    best_sn_pred = _stage2_best_pred(sn_preds)

    model_crop = None
    model_kind = ""
    model_conf = None
    if best_model_pred is not None:
        direct_crop = crop_model_field(img, best_model_pred)
        if model_crop_satisfies_target(direct_crop, img, best_model_pred):
            model_crop = direct_crop
            model_kind = "direct"
            model_conf = float(best_model_pred.get("confidence", 0))

    if model_crop is None:
        fallback_crop = fallback_model_crop_from_part_no(img, best_part_no_pred)
        if model_crop_satisfies_target(fallback_crop, img, None):
            model_crop = fallback_crop
            model_kind = "fallback_from_part_no"
            model_conf = 0.0

    if model_crop is None and best_sn_pred is not None:
        fallback_crop = fallback_model_crop_from_sn(img, best_sn_pred)
        if model_crop_satisfies_target(fallback_crop, img, None):
            model_crop = fallback_crop
            model_kind = "fallback_from_sn"
            model_conf = 0.0

    score = 0.0
    if model_crop is not None:
        h, w = model_crop.shape[:2]
        score += 700.0 if model_kind == "direct" else 350.0
        score += min(w, 600) / 10.0
        score += min(h, 160) / 20.0
        score += float(model_conf or 0.0) * 100.0
    if rotation == 180:
        score -= 1.0

    return {
        "rotation": rotation,
        "model_crop": model_crop,
        "model_kind": model_kind,
        "model_conf": model_conf,
        "score": score,
    }


def stage2_crop_model_from_label(label_img_path, out_path=None):
    img = read_image(label_img_path)
    if img is None:
        return None

    preds = _stage2_infer_field_preds(img, label_img_path)
    candidates = [_stage2_build_model_only_candidate(img, preds, rotation=0)]

    if candidates[0].get("model_crop") is None and _stage2_should_retry_rot180(
        {
            "model_required": True,
            "model_crop": None,
            "model_kind": "",
            "sn_crop": None,
            "part_no_ok": False,
        }
    ):
        rotated_img = rotate_image(img, 180)
        rotated_preds = _stage2_infer_field_preds(rotated_img, label_img_path, debug_suffix=" MODEL_ONLY_ROT180")
        candidates.append(_stage2_build_model_only_candidate(rotated_img, rotated_preds, rotation=180))

    selected = max(candidates, key=lambda item: item["score"])
    if selected.get("model_crop") is None:
        return None

    base = os.path.splitext(os.path.basename(label_img_path))[0]
    if out_path is None:
        out_path = os.path.join(OUT_MODEL_DIR, f"{base}__model.png")
    saved_path = save_png_required(out_path, selected["model_crop"], "delayed model crop")
    return {
        "label_id": base,
        "model_path": saved_path,
        "model_conf": selected.get("model_conf"),
        "model_crop_source": selected.get("model_kind", ""),
        "stage2_rotation": selected.get("rotation", 0),
    }


def main(input_dir=None, out_dir=None, log_level="info", clean=False):
    reset_stage1_reject_counts()
    set_log_level(log_level)
    configure_paths(input_dir=input_dir, out_dir=out_dir)
    ensure_dirs(clean=clean)
    
    # 额外创建分类文件夹
    MISS_SN_DIR = os.path.join(STAGE2_DIR, "miss_sn")
    MISS_MODEL_DIR = os.path.join(STAGE2_DIR, "miss_model")
    MISS_BOTH_DIR = os.path.join(STAGE2_DIR, "miss_both")
    os.makedirs(MISS_SN_DIR, exist_ok=True)
    os.makedirs(MISS_MODEL_DIR, exist_ok=True)
    os.makedirs(MISS_BOTH_DIR, exist_ok=True)

    imgs = list_images(INPUT_DIR)
    if not imgs:
        _log(f"警告：输入文件夹中没有可识别的图片：{INPUT_DIR}", "warn")
        with STAGE1_REJECT_COUNTS_LOCK:
            stage1_rejects = dict(STAGE1_REJECT_COUNTS)
        return {
            "input_images": 0,
            "label_count": 0,
            "manifest_rows": 0,
            "ok_any": 0,
            "ok_both": 0,
            "stage1_dir": STAGE1_DIR,
            "stage1_preview_dir": STAGE1_PREVIEW_DIR,
            "stage2_dir": STAGE2_DIR,
            "manifest_path": MANIFEST_PATH,
            "stage1_rejects": stage1_rejects,
        }

    # 清空 manifest（每次跑重新生成）
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        pass

    # Stage1：全部大图 -> 小图
    all_label_crops = []
    stage1_workers = crop_worker_count("stage1")
    stage2_workers = crop_worker_count("stage2")
    if stage1_workers > 1 or stage2_workers > 1:
        _log(
            f"裁剪并发：stage1_workers={stage1_workers}, stage2_workers={stage2_workers}",
            "info",
        )

    for label_crops in _map_ordered(imgs, stage1_crop_labels, stage1_workers):
        all_label_crops.extend(label_crops)
    if not all_label_crops:
        raise RuntimeError(f"{len(imgs)} 张输入图片没有裁剪出任何标签。")

    # Stage2：全部小图 -> model/sn
    stage1_crop_count = len(all_label_crops)
    ok_any = 0
    ok_both = 0
    model_count = 0
    part_no_count = 0
    sn_count = 0
    manifest_rows = 0
    stage2_dropped = 0
    progress_logs = crop_progress_log_enabled()

    def run_stage2_job(label_path):
        label_name = _crop_label_name(label_path)
        result = stage2_crop_fields(label_path)
        if progress_logs:
            _log(f"[2/4][字段完成] {label_name} -> {_stage2_result_summary(result)}", "info")
        return result

    stage2_results = _map_ordered(all_label_crops, run_stage2_job, stage2_workers)
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        for lp, r in zip(all_label_crops, stage2_results):
            if not r:
                if not stage1_keep_all_crops_enabled() and _remove_stage1_orphan_label(lp):
                    stage2_dropped += 1
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            manifest_rows += 1
            
            has_model = bool(r.get("model_path"))
            has_part_no = bool(r.get("part_no_path"))
            has_sn = bool(r.get("sn_path"))
            has_model_key = has_model or has_part_no
            model_count += int(has_model)
            part_no_count += int(has_part_no)
            sn_count += int(has_sn)
            
            if has_model_key or has_sn:
                ok_any += 1
            if has_model_key and has_sn:
                ok_both += 1
                
            # 分类拷贝失败样本
            if has_model_key and not has_sn:
                shutil.copy2(lp, os.path.join(MISS_SN_DIR, os.path.basename(lp)))
            elif has_sn and not has_model_key:
                shutil.copy2(lp, os.path.join(MISS_MODEL_DIR, os.path.basename(lp)))
            elif (not has_sn) and (not has_model_key):
                shutil.copy2(lp, os.path.join(MISS_BOTH_DIR, os.path.basename(lp)))

    _log(f"\n标签裁剪完成：Stage1 原始裁剪 {stage1_crop_count} 个；Stage2 有效保留 {manifest_rows} 个", "info")
    if stage2_dropped:
        _log(f"Stage2 过滤并删除 {stage2_dropped} 个无效 Stage1 标签裁剪", "info")
    _log(
        f"字段裁剪统计：至少识别到一个字段 {ok_any} 个；型号线索(型号或PartNo)和SN都识别到 {ok_both} 个；"
        f"model={model_count} part_no={part_no_count} sn={sn_count}",
        "info",
    )
    if stage1_save_preview_enabled():
        _log(f"Stage1 带框预览：{STAGE1_PREVIEW_DIR}", "info")
    _log(f"清单文件：{MANIFEST_PATH}", "info")
    _log(f"缺失分类文件夹：{MISS_SN_DIR} / {MISS_MODEL_DIR}", "info")
    with STAGE1_REJECT_COUNTS_LOCK:
        stage1_rejects = dict(STAGE1_REJECT_COUNTS)
    return {
        "input_images": len(imgs),
        "label_count": manifest_rows,
        "stage1_crop_count": stage1_crop_count,
        "stage2_retained_count": manifest_rows,
        "manifest_rows": manifest_rows,
        "ok_any": ok_any,
        "ok_both": ok_both,
        "model_count": model_count,
        "part_no_count": part_no_count,
        "sn_count": sn_count,
        "stage1_dir": STAGE1_DIR,
        "stage1_preview_dir": STAGE1_PREVIEW_DIR,
        "stage2_dir": STAGE2_DIR,
        "manifest_path": MANIFEST_PATH,
        "stage1_rejects": stage1_rejects,
    }

if __name__ == "__main__":
    main()
