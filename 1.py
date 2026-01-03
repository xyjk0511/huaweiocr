import cv2
import re
import os
import sys
import json
import base64
import warnings
import subprocess
import numpy as np
from inference_sdk import InferenceHTTPClient

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

# ==================== 环境配置区 ====================
try:
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

warnings.filterwarnings("ignore")
# [NEW] 跳过 PaddleOCR 的 hoster 检查
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ==================== 参数配置区 ====================
load_dotenv()
API_KEY = os.environ.get("API_KEY", "")
if not API_KEY:
    raise RuntimeError("API_KEY is missing. Set it in .env or environment variables.")
IMAGE_FOLDER = "test_images"
OUTPUT_FOLDER = "crops_labels"
WORKFLOW_ID = "huawei-2ha7t/7"
BARCODE_CLI_PATH = r"C:\Users\55093\Downloads\BarcodeReaderCLI.win.12.0.7889\BarcodeReaderCLI\bin\BarcodeReaderCLI.exe"

# [参数微调]
CROP_PADDING = 0.1
MIN_SIZE = 300
MIN_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.3
# ====================================================

# [NEW] 全局初始化 PaddleOCR
PADDLE_OCR = None
try:
    from paddleocr import PaddleOCR

    PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
except Exception:
    PADDLE_OCR = None

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

if not os.path.exists(BARCODE_CLI_PATH):
    print(f"❌ 错误: 找不到条码识别工具: {BARCODE_CLI_PATH}")
else:
    print(f"✅ 已定位条码识别工具")


def decode_base64_safe(data_str):
    try:
        return base64.b64decode(data_str).decode('utf-8')
    except Exception:
        return data_str


# ==================== [NEW] 解析增强：normalize + regex ====================

_KEEP_CHARS_PATTERN = re.compile(r'[^0-9A-Z-]+')


def normalize_payload(s: str) -> str:
    if not s:
        return ""
    s = decode_base64_safe(s)
    s = s.upper()
    s = _KEEP_CHARS_PATTERN.sub("", s)
    return s


SN_PATTERNS = [
    re.compile(r'(?:^|[^0-9A-Z])(215[0-9A-Z]{12,})(?:$|[^0-9A-Z])'),
    re.compile(r'(?:^|[^0-9A-Z])(4E25[0-9A-Z]{8,})(?:$|[^0-9A-Z])'),
]
SN_GS1_PREFIX = re.compile(r'(?:^|[^0-9A-Z])S(215[0-9A-Z]{12,}|4E25[0-9A-Z]{8,})(?:$|[^0-9A-Z])')
MODEL_PATTERNS = [
    re.compile(r'(AP[0-9A-Z-]{3,})'),
    re.compile(r'(S\d{2,4}[0-9A-Z-]{2,})'),
]


def parse_sn_model_from_text(text: str):
    sn_set, model_set = set(), set()
    if not text:
        return sn_set, model_set
    t = normalize_payload(text)
    for m in SN_GS1_PREFIX.finditer(t):
        sn_set.add(m.group(1))
    for pat in SN_PATTERNS:
        for m in pat.finditer(t):
            sn_set.add(m.group(1))
    for pat in MODEL_PATTERNS:
        for m in pat.finditer(t):
            model_set.add(m.group(1))
    return sn_set, model_set


def extract_info_from_label_crop(cli_output_json):
    info = {"sn_list": set(), "model_list": set(), "others": []}
    try:
        data = json.loads(cli_output_json)
        if "sessions" in data:
            for session in data["sessions"]:
                for barcode in session.get("barcodes", []):
                    raw_data = barcode.get("data")
                    if not raw_data: continue
                    text = decode_base64_safe(raw_data)
                    if not text or len(text) < 3: continue
                    sn_set, model_set = parse_sn_model_from_text(text)
                    info["sn_list"].update(sn_set)
                    info["model_list"].update(model_set)
                    if not sn_set and not model_set:
                        info["others"].append(text)
    except Exception:
        pass
    return info


# ==================== [NEW] OCR 兜底与 ROI ====================

def crop_model_roi(label_bgr):
    h, w = label_bgr.shape[:2]
    # 经验：Model行通常在左侧 0~70% 宽，中上 15%~55% 高
    x1, x2 = int(0.00 * w), int(0.70 * w)
    y1, y2 = int(0.15 * h), int(0.55 * h)
    roi = label_bgr[y1:y2, x1:x2]
    return roi


def ocr_text_tesseract(bgr_img):
    try:
        import pytesseract
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(th, lang="eng") or ""
    except Exception:
        return ""


def ocr_text_paddle(bgr_img):
    if PADDLE_OCR is None:
        return ""
    try:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        result = PADDLE_OCR.ocr(rgb, cls=True)
        lines = []
        if result:
            for block in result:
                for item in block:
                    text = item[1][0]
                    if text:
                        lines.append(text)
        return "\n".join(lines)
    except Exception:
        return ""


def run_barcode_cli(image_path):
    if not os.path.exists(BARCODE_CLI_PATH): return ""
    try:
        result = subprocess.run(
            [BARCODE_CLI_PATH, image_path],
            capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        return result.stdout
    except Exception:
        return ""


def add_white_border(image, border_ratio=0.2, min_border=40):
    """
    [NEW] 给图像加白边（静区），防止条码贴边导致解码失败
    """
    h, w = image.shape[:2]
    top = max(min_border, int(h * border_ratio))
    bottom = top
    left = max(min_border, int(w * border_ratio))
    right = left
    # 填充白色 (255, 255, 255)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def enhance_variants(image):
    """
    [UPDATED] 优化变体生成顺序：原图放大 -> 灰度 -> 轻锐化 -> 二值化
    """
    variants = []
    h, w = image.shape[:2]
    if h == 0 or w == 0: return variants

    # 0. 基础放大（所有变体都基于这个放大版）
    target_w = 2200
    scale = max(1.0, target_w / float(w))
    scale = min(scale, 4.0)
    up = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # [NEW] 1. 原始放大版（加白边）- 最纯净的版本
    up_bordered = add_white_border(up)
    variants.append(("orig_up", up_bordered))

    # [NEW] 2. 灰度版（加白边）
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray_bordered = add_white_border(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))  # 转回BGR方便统一处理
    variants.append(("gray", gray_bordered))

    # 3. 轻度锐化（加白边）
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(up, -1, kernel)
    sharp_bordered = add_white_border(sharp)
    variants.append(("mild_sharp", sharp_bordered))

    # 4. CLAHE + 二值化（加白边）- 最后的手段
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    th = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
    th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    th_bordered = add_white_border(th_bgr)
    variants.append(("th", th_bordered))

    return variants


def find_barcode_rois(image):
    rois = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return [image]
    img_area = image.shape[0] * image.shape[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < img_area * 0.02: continue
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y)
        roi = image[y1:y2, x1:x2]
        if roi.size > 0: rois.append(roi)
    if not rois: rois.append(image)
    return rois


def process_single_image(img_path):
    print(f"\n[正在处理]: {os.path.basename(img_path)}")
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return

    try:
        result = client.infer(img_path, model_id=WORKFLOW_ID)
        predictions = result.get('predictions', []) if isinstance(result, dict) else result
        if not predictions:
            print("  提示: 没有检测到任何标签")
            return
    except Exception as e:
        print(f"  模型调用出错: {e}")
        return

    boxes_for_nms, confidences, valid_predictions = [], [], []
    for pred in predictions:
        if not isinstance(pred, dict) or 'x' not in pred: continue
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        conf = pred.get('confidence', 1.0)
        if w < MIN_SIZE or h < MIN_SIZE or conf < MIN_CONFIDENCE: continue
        valid_predictions.append(pred)
        boxes_for_nms.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])
        confidences.append(float(conf))

    if not boxes_for_nms:
        print("  无有效标签区域 (已过滤)。")
        return

    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    final_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
    print(f"  发现 {len(final_indices)} 个大标签 (Label Areas)")

    failed_dir = os.path.join(OUTPUT_FOLDER, "failed")
    debug_dir = os.path.join(OUTPUT_FOLDER, "debug_cli")
    # [NEW] 专门存放所有检测到的标签原图的文件夹
    all_labels_dir = os.path.join(OUTPUT_FOLDER, "all_labels")

    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(all_labels_dir, exist_ok=True)

    for i, idx in enumerate(final_indices):
        pred = valid_predictions[idx]
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        pad_w, pad_h = w * CROP_PADDING, h * CROP_PADDING
        x1, y1 = int(x - w / 2 - pad_w), int(y - h / 2 - pad_h)
        x2, y2 = int(x + w / 2 + pad_w), int(y + h / 2 + pad_h)
        y1, y2 = max(0, y1), min(img.shape[0], y2)
        x1, x2 = max(0, x1), min(img.shape[1], x2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        # ==========================================================
        # [NEW] 只要检测到，先把这个大标签存下来
        # ==========================================================
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_raw_name = f"{base_name}_label_{i + 1}.png"
        label_raw_path = os.path.join(all_labels_dir, label_raw_name)
        cv2.imencode('.png', crop)[1].tofile(label_raw_path)
        # ==========================================================

        print(f"  > 标签 #{i + 1} (已存图) 处理中...", end="", flush=True)

        rois = find_barcode_rois(crop)
        found_info = {"sn_list": set(), "model_list": set()}
        any_success = False

        for r_idx, roi_img in enumerate(rois):
            variants = enhance_variants(roi_img)
            rotations = [("r0", None), ("r90", cv2.ROTATE_90_CLOCKWISE), ("r180", cv2.ROTATE_180),
                         ("r270", cv2.ROTATE_90_COUNTERCLOCKWISE)]
            roi_success = False
            for var_name, var_img in variants:
                if roi_success: break
                for rot_name, rot_code in rotations:
                    current_try_img = cv2.rotate(var_img, rot_code) if rot_code is not None else var_img
                    # 注意：这里改用 roi 临时命名，避免和 raw label 混淆
                    tag = f"{var_name}_{rot_name}"
                    tmp_path = os.path.join(debug_dir, f"{base_name}_label_{i + 1}_roi_{r_idx}_{tag}.png")
                    cv2.imencode(".png", current_try_img)[1].tofile(tmp_path)
                    cli_output = run_barcode_cli(tmp_path)
                    if cli_output and ("barcodes" in cli_output or "sessions" in cli_output):
                        info = extract_info_from_label_crop(cli_output)
                        found_info["sn_list"].update(info["sn_list"])
                        found_info["model_list"].update(info["model_list"])
                        if info['sn_list'] or info['model_list']:
                            print(f" [ROI {r_idx} 成功]", end="")
                            roi_success = True
                            any_success = True
                            success_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_label_{i + 1}_SUCCESS_{tag}.png")
                            os.rename(tmp_path, success_path)
                            break

        # [UPDATED] 只要 Model 没找到，就对特定区域 OCR 兜底 (不管条码有没有读到SN)
        if not found_info["model_list"]:
            roi_for_ocr = crop_model_roi(crop)
            ocr_txt = ocr_text_tesseract(roi_for_ocr)
            if not ocr_txt.strip():
                ocr_txt = ocr_text_paddle(roi_for_ocr)
            if ocr_txt.strip():
                sn_set, model_set = parse_sn_model_from_text(ocr_txt)
                if model_set:
                    found_info["model_list"].update(model_set)
                    print(" [OCR兜底成功]", end="")
                    any_success = True

        if any_success:
            sn_str = f"SN: {', '.join(sorted(found_info['sn_list']))}" if found_info['sn_list'] else ""
            model_str = f"Model: {', '.join(sorted(found_info['model_list']))}" if found_info['model_list'] else ""
            print(f" ✅ {sn_str}{' | ' if sn_str and model_str else ''}{model_str}")
        else:
            print(" ⚠️ 未在标签内读到 SN/Model（条码+OCR都失败）")
            # 失败图依然存在 failed 文件夹
            fail_save_path = os.path.join(failed_dir, f"FAILED_{label_raw_name}")
            cv2.imencode('.png', crop)[1].tofile(fail_save_path)


def main():
    if not os.path.exists(IMAGE_FOLDER): os.makedirs(IMAGE_FOLDER)
    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1].lower() in exts]
    print(f"--- 任务启动：ROI + OCR优化模式 ---")
    print(f"--- 模型: [{WORKFLOW_ID}] ---")
    print(f"--- 待处理图片：{len(images)} 张 ---")
    print(f"--- 标签原图将保存至: {os.path.join(OUTPUT_FOLDER, 'all_labels')} ---")
    for img_name in images:
        process_single_image(os.path.join(IMAGE_FOLDER, img_name))
    print("\n--- 处理完毕 ---")


if __name__ == "__main__":
    main()
