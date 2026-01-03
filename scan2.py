import os
import re
import json
import cv2
import tempfile
import numpy as np
from ocr import init_ocr, ocr_one_image
from barcode import decode_small_patch
from app_paths import ensure_models_installed

# Simple log gating for CLI usage.
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

# ===================== CONFIG =====================
MODEL_CROP_DIR = r"stage2_fields\model"
SN_CROP_DIR = r"stage2_fields\sn"
OUT_JSONL = r"model_sn_ocr.jsonl"
DEBUG_LOG_PATH = r"debug_ocr_barcode.log"

SN_TEXT_ROI_TOP_RATIO = 0.0

MAX_TARGET_W = 1200
MAX_SCALE = 4.0

SN_TARGET_WIDTHS = [1000, 1400, 1800]
SN_MAX_SCALE = 4.0

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ensure_models_installed()

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def configure_paths(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=None):
    global MODEL_CROP_DIR, SN_CROP_DIR, OUT_JSONL, DEBUG_LOG_PATH
    if out_dir:
        MODEL_CROP_DIR = os.path.join(out_dir, "stage2_fields", "model")
        SN_CROP_DIR = os.path.join(out_dir, "stage2_fields", "sn")
        OUT_JSONL = os.path.join(out_dir, "model_sn_ocr.jsonl")
        DEBUG_LOG_PATH = os.path.join(out_dir, "debug_ocr_barcode.log")
    if model_dir:
        MODEL_CROP_DIR = model_dir
    if sn_dir:
        SN_CROP_DIR = sn_dir
    if out_jsonl:
        OUT_JSONL = out_jsonl
    if debug_log:
        DEBUG_LOG_PATH = debug_log

# ===================== OCR =====================
OCR_ENGINE = None


def get_ocr_engine():
    global OCR_ENGINE
    if OCR_ENGINE is None:
        OCR_ENGINE = init_ocr()
    return OCR_ENGINE


def append_debug(line: str) -> None:
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def start_debug_run():
    append_debug("=" * 72)
    append_debug(f"[RUN] model_dir={MODEL_CROP_DIR} sn_dir={SN_CROP_DIR}")


# ===================== UTILS =====================
def _read_image(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flags)
    if img is not None:
        return img
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None

def load_and_preprocess(path, roi_bottom=False):
    img = _read_image(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    if roi_bottom and SN_TEXT_ROI_TOP_RATIO > 0:
        h, w = img.shape[:2]
        top = int(h * SN_TEXT_ROI_TOP_RATIO)
        img = img[top:, :]

    h, w = img.shape[:2]
    scale = min(MAX_TARGET_W / float(w), MAX_SCALE)
    if scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    return bin_img


def load_for_ocr_color(path):
    img = _read_image(path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = min(MAX_TARGET_W / float(w), MAX_SCALE)
    if scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def ocr_text(img):
    ocr_engine = get_ocr_engine()
    texts, concat = ocr_one_image(ocr_engine, img)
    if not texts:
        return "", concat
    text = " ".join(t["text"] for t in texts)
    return text, concat


def ocr_text_with_details(img):
    ocr_engine = get_ocr_engine()
    texts, concat = ocr_one_image(ocr_engine, img)
    if not texts:
        return "", concat, []
    text = " ".join(t["text"] for t in texts)
    return text, concat, texts


def ocr_sn_top_text(path: str):
    img = _read_image(path, cv2.IMREAD_COLOR)
    if img is None:
        return "", ""

    h, w = img.shape[:2]
    views = [int(h * 0.35), int(h * 0.45)]
    for top_h in views:
        if top_h <= 0:
            continue
        top = img[:top_h, :]

        scale = min(max(MAX_TARGET_W, 2000) / float(w), MAX_SCALE)
        if scale > 1.0:
            top = cv2.resize(top, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        text, concat = ocr_text(top)
        if text or concat:
            return text, concat

        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        text, concat = ocr_text(color)
        if text or concat:
            return text, concat

        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        text, concat = ocr_text(bin_img)
        if text or concat:
            return text, concat

    return "", ""


def read_barcodes(img_path: str) -> list[str]:
    """Use barcode.py pipeline and return decoded strings."""
    img = _read_image(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return []

    info = decode_small_patch(img)
    lines = [r.get("data", "") for r in info.get("results", []) if r.get("data")]
    return list(dict.fromkeys(lines))


# ========= MODEL RULES =========

MODEL_LINE_RE = re.compile(
    r"MODEL[:：]?\s*([A-Z0-9\-]{3,32})",
    re.I,
)
S380_S8P2T_RE = re.compile(r"S380\W*S8P2T", re.I)

BAD_MODEL_WORDS = {
    "MODEL", "DESC", "DESCRIPTION", "QTY", "REV",
    "WAN", "LAN", "BASE", "UPC", "SN", "MAC",
}


def normalize_model(code: str) -> str:
    c = re.sub(r"[^0-9A-Z\-]", "", code.upper())

    if len(c) % 2 == 0 and c[:len(c)//2] == c[len(c)//2:]:
        c = c[:len(c)//2]

    c = re.sub(r"DESC$", "", c)

    if c.startswith("S1108T"):
        c = "S110-8T"
    elif c.startswith("S1108P1T"):
        c = "S110-8P1T"
    if c == "S110-6T":
        c = "S110-5T"

    if c.startswith("S380S8P2T"):
        c = "S380-" + c[len("S380"):]
    if c == "S8P27":
        c = "S380-S8P2T"

    m = re.match(r"^([A-Z0-9\-]*[A-Z])[0-9]{5,}$", c)
    if m:
        c = m.group(1)

    return c


def extract_model_from_text(text: str) -> str:
    if not text:
        return ""

    t = text.upper()
    if S380_S8P2T_RE.search(t):
        return "S380-S8P2T"
    m = MODEL_LINE_RE.search(t)
    if m:
        raw = m.group(1)
        return normalize_model(raw)

    tokens = re.findall(r"[A-Z][A-Z0-9\-]{2,}", t)
    cand = []
    for tok in tokens:
        tok_clean = tok.strip("-")
        if tok_clean in BAD_MODEL_WORDS:
            continue
        if not re.search(r"\d", tok_clean):
            continue
        cand.append(tok_clean)

    if not cand:
        return ""

    cand.sort(key=lambda s: (-len(s), not s.startswith(("A", "S"))))
    best = cand[0]
    return normalize_model(best)


# ========= SN RULES =========

SN20_RE = re.compile(
    r"(2[0-9]{9,10}(?:ERA|ER|LDR|LDRA|SRA)[0-9]{4,7})"
)

SN12_RE = re.compile(
    r"(4E[0-9A-Z]{10})"
)


def _clean_code(s: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def extract_sn_from_text(text: str) -> str:
    s = _clean_code(text)
    while s.startswith("SN"):
        s = s[2:]

    m = SN20_RE.search(s)
    if m and len(m.group(1)) == 20:
        return m.group(1)

    m = SN12_RE.search(s)
    if m and len(m.group(0)) == 12:
        return m.group(0)

    blocks = re.findall(r"[0-9A-Z]{12,20}", s)
    best = ""
    for b in blocks:
        letters = sum(ch.isalpha() for ch in b)
        digits = sum(ch.isdigit() for ch in b)
        if letters >= 2 and digits >= 6:
            if len(b) > len(best):
                best = b
    return best


def extract_sn_from_barcode_candidate(text: str) -> str:
    return extract_sn_from_text(text)


def filter_sn_lines(lines: list[str]) -> list[str]:
    filtered = []
    for ln in lines:
        s = _clean_code(ln)
        if SN20_RE.search(s) or SN12_RE.search(s):
            filtered.append(ln)
    return filtered


def filter_model_lines(lines: list[str]) -> list[str]:
    filtered = []
    for ln in lines:
        if extract_model_from_text(ln):
            filtered.append(ln)
    return filtered


def build_sn_views(img_bgr):
    h, w = img_bgr.shape[:2]
    views = []
    views.append(("orig", img_bgr))

    pad = max(int(w * 0.08), 10)
    qzpad = cv2.copyMakeBorder(
        img_bgr, 0, 0, pad, pad, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    views.append(("qzpad", qzpad))

    if h > w * 1.3:
        rot = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        views.append(("rot90ccw", rot))

        rh, rw = rot.shape[:2]
        pad_r = max(int(rw * 0.08), 10)
        rot_qzpad = cv2.copyMakeBorder(
            rot, 0, 0, pad_r, pad_r, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        views.append(("rot90ccw_qzpad", rot_qzpad))

    return views


def decode_barcodes_with_dbr(img_bgr, debug_name=""):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, img_bgr)
        return read_barcodes(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def decode_sn_with_dbr(sn_img_bgr, sn_name=""):
    codes = []
    views = build_sn_views(sn_img_bgr)

    for _, vimg in views:
        h, w = vimg.shape[:2]

        try:
            codes = decode_barcodes_with_dbr(vimg, debug_name=f"{sn_name}[orig]")
        except Exception:
            codes = []

        if codes:
            break

        for tw in SN_TARGET_WIDTHS:
            if tw <= w:
                continue

            scale = min(tw / w, SN_MAX_SCALE)
            if scale <= 1.01:
                continue

            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            resized = cv2.resize(vimg, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            try:
                codes = decode_barcodes_with_dbr(resized, debug_name=f"{sn_name}[x{scale:.2f}]")
            except Exception:
                codes = []

            if codes:
                break

        if codes:
            break

    return codes


def try_sn_from_barcode(img_path: str) -> tuple[str, str]:
    lines = read_barcodes(img_path)
    append_debug(f"[SN][BARCODE] {os.path.basename(img_path)} | {lines}")
    if not lines:
        return "", ""

    use_lines = filter_sn_lines(lines) or lines
    for ln in use_lines:
        sn = extract_sn_from_barcode_candidate(ln)
        if sn:
            return sn, ln

    return "", "; ".join(lines)


def extract_model_from_barcode_candidate(text: str) -> str:
    return extract_model_from_text(text)


def try_model_from_barcode(img_path: str) -> tuple[str, str]:
    lines = read_barcodes(img_path)
    append_debug(f"[MODEL][BARCODE] {os.path.basename(img_path)} | {lines}")
    if not lines:
        return "", ""

    use_lines = filter_model_lines(lines) or lines
    for ln in use_lines:
        model = extract_model_from_barcode_candidate(ln)
        if model:
            return model, ln

    return "", "; ".join(lines)


def recognize_model(model_path: str, label_id: str = ""):
    tag = f"{label_id} " if label_id else ""
    append_debug(f"[MODEL] {tag}{os.path.basename(model_path)}")
    m_from_bc, bc_raw = try_model_from_barcode(model_path)
    if m_from_bc:
        return m_from_bc, f"[BARCODE] {bc_raw}", "barcode"

    color_img = load_for_ocr_color(model_path)
    if color_img is not None:
        text, concat, texts = ocr_text_with_details(color_img)
        append_debug(f"[MODEL][OCR_COLOR] {tag}{os.path.basename(model_path)} | {text!r}")
        append_debug(
            f"[MODEL][OCR_COLOR][TEXTS] {tag}{os.path.basename(model_path)} | "
            f"{json.dumps(texts, ensure_ascii=False)}"
        )
        model_code = extract_model_from_text(text)
        if model_code:
            return model_code, text, "ocr_color"

    img = load_and_preprocess(model_path, roi_bottom=False)
    text, concat, texts = ocr_text_with_details(img)
    append_debug(f"[MODEL][OCR_BIN] {tag}{os.path.basename(model_path)} | {text!r}")
    append_debug(
        f"[MODEL][OCR_BIN][TEXTS] {tag}{os.path.basename(model_path)} | "
        f"{json.dumps(texts, ensure_ascii=False)}"
    )
    model_code = extract_model_from_text(text)
    if model_code:
        return model_code, text, "ocr_bin"
    return "", text, "none"


def preprocess_sn_image(path: str):
    try:
        return load_and_preprocess(path, roi_bottom=True)
    except Exception:
        return None


def recognize_sn(sn_path: str, label_id: str = ""):
    tag = f"{label_id} " if label_id else ""
    append_debug(f"[SN] {tag}{os.path.basename(sn_path)}")
    codes = read_barcodes(sn_path)
    append_debug(f"[SN][BARCODE] {tag}{os.path.basename(sn_path)} | {codes}")

    candidate_lines = filter_sn_lines(codes) or codes
    candidate_sns = []
    for c in candidate_lines:
        sn = extract_sn_from_barcode_candidate(c)
        if sn:
            candidate_sns.append(sn)
    candidate_sns = list(dict.fromkeys(candidate_sns))

    if len(candidate_sns) == 1:
        return candidate_sns[0], f"[BARCODE] {'; '.join(codes)}", "barcode"

    color_img = load_for_ocr_color(sn_path)
    if color_img is None:
        if codes:
            return "", f"[BARCODE] {'; '.join(codes)}", "barcode_raw"
        return "", "", "none"

    text, concat, texts = ocr_text_with_details(color_img)
    append_debug(f"[SN][OCR_COLOR] {tag}{os.path.basename(sn_path)} | text={text!r} concat={concat!r}")
    append_debug(f"[SN][OCR_COLOR][TEXTS] {tag}{os.path.basename(sn_path)} | {json.dumps(texts, ensure_ascii=False)}")
    sn = extract_sn_from_text(concat or text)
    if sn:
        return sn, text, "ocr"

    img = load_and_preprocess(sn_path, roi_bottom=True)
    text, concat, texts = ocr_text_with_details(img)
    append_debug(f"[SN][OCR_BIN] {tag}{os.path.basename(sn_path)} | text={text!r} concat={concat!r}")
    append_debug(f"[SN][OCR_BIN][TEXTS] {tag}{os.path.basename(sn_path)} | {json.dumps(texts, ensure_ascii=False)}")
    sn = extract_sn_from_text(concat or text)
    if sn:
        return sn, text, "ocr_bin"

    top_text, top_concat = ocr_sn_top_text(sn_path)
    append_debug(f"[SN][OCR_TOP] {tag}{os.path.basename(sn_path)} | text={top_text!r} concat={top_concat!r}")
    sn = extract_sn_from_text(top_concat or top_text)
    if sn:
        return sn, top_text, "ocr_top"

    if codes:
        return "", f"[BARCODE] {'; '.join(codes)}", "barcode_raw"

    return "", text, "none"


def label_key(name: str) -> str:
    parts = name.split("__")
    return "__".join(parts[:2]) if len(parts) >= 2 else os.path.splitext(name)[0]


# ===================== MAIN =====================
def main(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=None, log_level="info"):
    set_log_level(log_level)
    configure_paths(
        out_dir=out_dir,
        model_dir=model_dir,
        sn_dir=sn_dir,
        out_jsonl=out_jsonl,
        debug_log=debug_log,
    )
    start_debug_run()
    records = {}
    model_files = []
    sn_files = []

    for fname in os.listdir(MODEL_CROP_DIR):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in EXTS:
            continue
        model_files.append(fname)
        key = label_key(fname)
        records.setdefault(key, {})["model_path"] = os.path.join(MODEL_CROP_DIR, fname)

    for fname in os.listdir(SN_CROP_DIR):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in EXTS:
            continue
        sn_files.append(fname)
        key = label_key(fname)
        records.setdefault(key, {})["sn_path"] = os.path.join(SN_CROP_DIR, fname)

    append_debug(f"[SCAN2] model_files={len(model_files)} sn_files={len(sn_files)} keys={len(records)}")
    append_debug(f"[SCAN2] model_names={sorted(model_files)}")
    append_debug(f"[SCAN2] sn_names={sorted(sn_files)}")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for key in sorted(records.keys()):
            item = records[key]

            model_code = model_raw = ""
            sn_code = sn_raw = ""
            model_src = "missing"
            sn_src = "missing"

            if "model_path" in item:
                model_code, model_raw, model_src = recognize_model(item["model_path"], label_id=key)

            if "sn_path" in item:
                sn_code, sn_raw, sn_src = recognize_sn(item["sn_path"], label_id=key)

            if sn_code.startswith("4E25A017") and model_code in {"", "S380-S8P"}:
                model_code = "S380-S8P2T"
                model_src = f"{model_src}+sn_hint" if model_src else "sn_hint"

            if model_code == "S380-S8P2T" and sn_code and not sn_code.startswith("4E25A017"):
                append_debug(f"[WARN] {key} model= S380-S8P2T but sn={sn_code}")

            out = {
                "label_id": key,
                "model": model_code,
                "sn": sn_code,
                "model_raw": model_raw,
                "sn_raw": sn_raw,
                "model_src": model_src,
                "sn_src": sn_src,
            }

            _log(
                f"[{key}] "
                f"MODEL={model_code} (M_SRC={model_src}) | "
                f"SN={sn_code} (SN_SRC={sn_src})",
                "info",
            )

            f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
