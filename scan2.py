import os
import re
import json
import cv2
import tempfile
import numpy as np
from ocr import init_ocr, ocr_one_image
from barcode import decode_small_patch
from app_paths import ensure_models_installed
from sn_barcode import extract_sn_from_payload, learn_sn_pattern, scan_sn_barcodes

# Simple log gating for CLI usage.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
LOG_SINK = None

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
    if LOG_LEVEL != "debug":
        return
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_mask_sensitive_text(line) + "\n")
    except Exception:
        pass


def _mask_sensitive_text(value: str) -> str:
    def repl(match):
        text = match.group(0)
        if len(text) <= 8:
            return "*" * len(text)
        return text[:4] + ("*" * (len(text) - 8)) + text[-4:]

    masked = re.sub(r"(?i)([a-z]:\\|/)[^\s|,\]\)]+", "[path]", value)
    masked = re.sub(r"(?i)[^\\/\s|,\[\]'\"]+\.(jpg|jpeg|png|bmp|webp)", "[file]", masked)
    masked = re.sub(r"(?i)(label[_-]?id=)[^\s|,]+", r"\1[masked]", masked)
    masked = re.sub(r"[0-9A-Za-z_:-]{8,}", repl, masked)
    return re.sub(r"[0-9A-Za-z]{8,}", repl, masked)


def append_sensitive_debug(line: str) -> None:
    append_debug(line)


def start_debug_run():
    append_debug("=" * 72)
    append_debug("[RUN] scan2 started")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


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
S380_S8P2T_NOISY_RE = re.compile(r"\bM[O0]8S\W*[O0]8[O0]2\b", re.I)

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
    if re.match(r"^M[O0]8S-[O0]8[O0]2", c):
        c = "S380-S8P2T"
    if c in {"S380-", "S380-S", "S380-S8P", "S380-S8P2"}:
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
    if S380_S8P2T_NOISY_RE.search(t):
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


def extract_model_from_ocr_result(text: str, concat: str) -> str:
    # Keep OCR spacing for MODEL first; concat can merge trailing description
    # text into the model token (for example "AP162E 9SC" -> "AP162E9SC").
    return extract_model_from_text(text) or extract_model_from_text(concat)


# ========= SN RULES =========

def _clean_code(s: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def extract_sn_from_text(text: str) -> str:
    raw = "" if text is None else str(text)
    if not raw.strip():
        return ""

    payload_sn = extract_sn_from_payload(raw)
    if payload_sn:
        return payload_sn

    upper = raw.upper()
    for match in re.finditer(r"(?:S\s*/\s*N|SN|SERIAL(?:\s*NO)?|SNO)\s*[:：#-]?\s*([0-9A-Z][0-9A-Z\s:/#\-._]{8,48})", upper):
        candidate = extract_sn_from_payload(match.group(1))
        if candidate:
            return candidate

    cleaned = _clean_code(raw)
    for marker in ("SERIALNO", "SERIAL", "SNO", "SN"):
        start = 0
        while True:
            idx = cleaned.find(marker, start)
            if idx < 0:
                break
            candidate = extract_sn_from_payload(cleaned[idx:])
            if candidate:
                return candidate
            start = idx + len(marker)

    for match in re.finditer(r"[0-9A-Z]{10,40}", upper):
        candidate = extract_sn_from_payload(match.group(0))
        if candidate:
            return candidate

    return _extract_unmatched_barcode_sn(raw)


def extract_sn_from_barcode_candidate(text: str) -> str:
    return extract_sn_from_payload(text)


def _strip_sn_prefixes(value: str) -> str:
    s = _clean_code(value)
    while s.startswith("SN"):
        s = s[2:]
    for prefix in ("SERIALNO", "SERIAL", "SNO"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _extract_unmatched_barcode_sn(raw_text: str) -> str:
    raw = "" if raw_text is None else str(raw_text)
    if not raw.strip():
        return ""
    raw_upper = raw.upper()

    strict_unknown_sn20 = re.compile(r"^2[0-9]{9,10}[A-Z]{2,5}[0-9]{4,7}$")
    strict_sn12 = re.compile(r"^4E[0-9A-Z]{10}$")
    non_sn_prefix = re.compile(r"^(SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART|PN|MODEL|DESC|ROUTE|WAYBILL|SNMP|IMEI)")
    non_sn_leading = re.compile(
        r"(?:^|[^A-Z0-9])(?:SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART(?:\s*NO)?|P\s*/?\s*N|PN|MODEL|DESC|ROUTE|WAYBILL|SNMP|IMEI)\s*[:：#-]?\s*$"
    )

    candidates = []
    for match in re.finditer(r"[0-9A-Z]{8,40}", raw_upper):
        token = match.group(0)
        leading = raw_upper[max(0, match.start() - 24):match.start()]
        if non_sn_leading.search(leading):
            continue
        cleaned = _strip_sn_prefixes(token)
        if len(cleaned) < 10 or non_sn_prefix.match(cleaned):
            continue
        if strict_sn12.match(cleaned):
            return cleaned
        if len(cleaned) != 20:
            continue
        if not strict_unknown_sn20.match(cleaned):
            continue
        score = (
            abs(len(cleaned) - 20),
            -len(cleaned),
        )
        candidates.append((score, cleaned))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return ""


def _try_learn_sn_pattern(sn: str, meta: dict | None = None) -> bool:
    try:
        learned = learn_sn_pattern(sn)
    except Exception as exc:
        append_debug(f"[SN][PATTERN_LEARN_FAIL] {exc.__class__.__name__}: {exc}")
        if isinstance(meta, dict):
            meta["sn_pattern_learned"] = False
            meta["sn_pattern_learn_error"] = exc.__class__.__name__
        return False
    if learned and isinstance(meta, dict):
        meta["sn_pattern_learned"] = True
    return learned


def _confirm_barcode_with_ocr(
    barcode_candidate: str,
    ocr_text: str,
    barcode_report,
    meta: dict,
    ocr_source: str,
):
    if not barcode_candidate:
        return None
    ocr_candidate = _extract_unmatched_barcode_sn(ocr_text)
    if not ocr_candidate or ocr_candidate != barcode_candidate:
        return None
    _try_learn_sn_pattern(barcode_candidate, meta)
    return (
        barcode_candidate,
        f"[BARCODE_OCR_AGREE:{ocr_source}] {barcode_report.raw_text} | OCR={ocr_text}",
        "barcode_ocr_agree",
        meta,
    )


def filter_sn_lines(lines: list[str]) -> list[str]:
    filtered = []
    for ln in lines:
        if extract_sn_from_payload(ln):
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


def _scan_barcode_sources(sources: list[tuple[str, str]], label_id: str = "", field: str = "SN") -> list[dict]:
    entries = []
    seen = set()
    tag = f"{label_id} " if label_id else ""
    for source, img_path in sources:
        if not img_path:
            continue
        lines = read_barcodes(img_path)
        append_sensitive_debug(
            f"[{field}][BARCODE][{source}] {tag}{os.path.basename(img_path)} | {lines}"
        )
        for line in lines:
            if not line:
                continue
            key = (source, line)
            if key in seen:
                continue
            seen.add(key)
            entries.append({"source": source, "data": line})
    return entries


def _format_barcode_entries(entries: list[dict]) -> str:
    parts = []
    for entry in entries:
        source = entry.get("source", "unknown")
        data = entry.get("data", "")
        if data:
            parts.append(f"{source}:{data}")
    return "; ".join(parts)


def _debug_candidate_dir() -> str:
    if not _env_flag("SN_BARCODE_DEBUG_CANDIDATES"):
        return ""
    root = os.environ.get("SN_BARCODE_DEBUG_DIR", "").strip()
    if root:
        return root
    return os.path.join(os.path.dirname(os.path.abspath(DEBUG_LOG_PATH)), "sn_barcode_candidates")


def _format_barcode_report(report) -> str:
    parts = []
    for result in report.results:
        if result.raw_text:
            parts.append(f"{result.source_region}:{result.raw_text}")
    return "; ".join(parts)


def _scan_sn_barcode_report(
    sources: list[tuple[str, str]],
    label_id: str = "",
    allow_early_exit: bool = True,
):
    return scan_sn_barcodes(
        sources,
        fallback_path_decoder=read_barcodes,
        label_id=label_id,
        debug_dir=_debug_candidate_dir(),
        early_exit=allow_early_exit,
    )


def _select_sn_from_barcode_entries(entries: list[dict]) -> tuple[str, str, str]:
    candidates = []
    seen = set()
    for entry in entries:
        data = entry.get("data", "")
        sn = extract_sn_from_barcode_candidate(data)
        if not sn:
            continue
        key = (sn, entry.get("source", "unknown"))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((sn, data, entry.get("source", "unknown")))

    unique_sns = sorted({sn for sn, _, _ in candidates})
    if len(unique_sns) == 1:
        for sn, raw, source in candidates:
            if sn == unique_sns[0]:
                return sn, raw, source

    for preferred_source in ("sn", "label"):
        preferred = [(sn, raw, source) for sn, raw, source in candidates if source == preferred_source]
        preferred_unique = sorted({sn for sn, _, _ in preferred})
        if len(preferred_unique) == 1:
            for sn, raw, source in preferred:
                if sn == preferred_unique[0]:
                    return sn, raw, source

    return "", "", ""


def try_sn_from_barcode(img_path: str) -> tuple[str, str]:
    report = _scan_sn_barcode_report([("sn", img_path)], allow_early_exit=True)
    if not report.results:
        return "", ""

    if report.status == "hit":
        return report.sn, report.raw_text

    return "", _format_barcode_report(report)


def extract_model_from_barcode_candidate(text: str) -> str:
    return extract_model_from_text(text)


def try_model_from_barcode(img_path: str) -> tuple[str, str]:
    entries = _scan_barcode_sources([("model", img_path)], field="MODEL")
    if not entries:
        return "", ""

    lines = [entry["data"] for entry in entries]
    use_lines = filter_model_lines(lines) or lines
    for ln in use_lines:
        model = extract_model_from_barcode_candidate(ln)
        if model:
            return model, ln

    return "", _format_barcode_entries(entries)


def recognize_model(model_path: str, label_id: str = "", use_barcode: bool = False):
    tag = f"{label_id} " if label_id else ""
    append_debug(f"[MODEL] {tag}{os.path.basename(model_path)}")
    if use_barcode:
        m_from_bc, bc_raw = try_model_from_barcode(model_path)
        if m_from_bc:
            return m_from_bc, f"[BARCODE] {bc_raw}", "barcode"

    text, concat, texts = ocr_text_with_details(model_path)
    append_sensitive_debug(f"[MODEL][OCR_FILE] {tag}{os.path.basename(model_path)} | {text!r}")
    append_sensitive_debug(
        f"[MODEL][OCR_FILE][TEXTS] {tag}{os.path.basename(model_path)} | "
        f"{json.dumps(texts, ensure_ascii=False)}"
    )
    model_code = extract_model_from_ocr_result(text, concat)
    if model_code:
        return model_code, text or concat, "ocr_file"

    color_img = load_for_ocr_color(model_path)
    if color_img is not None:
        text, concat, texts = ocr_text_with_details(color_img)
        append_sensitive_debug(f"[MODEL][OCR_COLOR] {tag}{os.path.basename(model_path)} | {text!r}")
        append_sensitive_debug(
            f"[MODEL][OCR_COLOR][TEXTS] {tag}{os.path.basename(model_path)} | "
            f"{json.dumps(texts, ensure_ascii=False)}"
        )
        model_code = extract_model_from_ocr_result(text, concat)
        if model_code:
            return model_code, text, "ocr_color"

    img = load_and_preprocess(model_path, roi_bottom=False)
    text, concat, texts = ocr_text_with_details(img)
    append_sensitive_debug(f"[MODEL][OCR_BIN] {tag}{os.path.basename(model_path)} | {text!r}")
    append_sensitive_debug(
        f"[MODEL][OCR_BIN][TEXTS] {tag}{os.path.basename(model_path)} | "
        f"{json.dumps(texts, ensure_ascii=False)}"
    )
    model_code = extract_model_from_ocr_result(text, concat)
    if model_code:
        return model_code, text, "ocr_bin"
    return "", text, "none"


def preprocess_sn_image(path: str):
    try:
        return load_and_preprocess(path, roi_bottom=True)
    except Exception:
        return None


def recognize_sn(
    sn_path: str = "",
    label_id: str = "",
    label_path: str = "",
    original_path: str = "",
    allow_ocr: bool = True,
):
    tag = f"{label_id} " if label_id else ""
    append_debug(
        f"[SN] {tag}sn={os.path.basename(sn_path) if sn_path else '[missing]'} "
        f"label={os.path.basename(label_path) if label_path else '[missing]'} "
        f"original={os.path.basename(original_path) if original_path else '[missing]'}"
    )
    sources = []
    if sn_path:
        sources.append(("sn", sn_path))
    if label_path and label_path != sn_path:
        sources.append(("label", label_path))
    if original_path and original_path not in {sn_path, label_path}:
        sources.append(("original", original_path))

    barcode_report = _scan_sn_barcode_report(sources, label_id=label_id, allow_early_exit=True)
    meta = barcode_report.to_meta()
    meta["ocr_text_found"] = False
    barcode_raw = _format_barcode_report(barcode_report)
    barcode_unmatched_sn = ""
    if barcode_report.status == "parse_failure":
        barcode_unmatched_sn = _extract_unmatched_barcode_sn(barcode_report.raw_text)
    append_sensitive_debug(
        f"[SN][BARCODE_REPORT] {tag}status={barcode_report.status} "
        f"attempts={barcode_report.attempts} decoded={barcode_report.decoded_count} raw={barcode_raw!r}"
    )

    if barcode_report.status == "hit":
        return (
            barcode_report.sn,
            f"[BARCODE:{barcode_report.source_region}] {barcode_report.raw_text}",
            "barcode",
            meta,
        )

    if not allow_ocr:
        if barcode_report.status == "ambiguous":
            return "", f"[BARCODE_AMBIGUOUS] {barcode_raw}", "barcode_ambiguous", meta
        if barcode_report.status == "parse_failure":
            if barcode_unmatched_sn:
                return (
                    barcode_unmatched_sn,
                    f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {barcode_report.raw_text}",
                    "barcode_unmatched",
                    meta,
                )
            return "", f"[BARCODE_PARSE_FAIL] {barcode_raw}", "barcode_parse_fail", meta
        if barcode_report.status == "quality_reject":
            return "", "[BARCODE_QUALITY_REJECT]", "barcode_quality_reject", meta
        return "", "", "barcode_decoder_miss", meta

    if not sn_path:
        if barcode_unmatched_sn:
            return (
                barcode_unmatched_sn,
                f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {barcode_report.raw_text}",
                "barcode_unmatched",
                meta,
            )
        if barcode_report.results:
            if barcode_report.status == "ambiguous":
                return "", f"[BARCODE_AMBIGUOUS] {barcode_raw}", "barcode_ambiguous", meta
            return "", f"[BARCODE] {barcode_raw}", "barcode_no_match", meta
        return "", "", "none", meta

    color_img = load_for_ocr_color(sn_path)
    if color_img is None:
        if barcode_report.results:
            return "", f"[BARCODE] {barcode_raw}", "barcode_no_match", meta
        return "", "", "none", meta

    text, concat, texts = ocr_text_with_details(color_img)
    append_sensitive_debug(f"[SN][OCR_COLOR] {tag}{os.path.basename(sn_path)} | text={text!r} concat={concat!r}")
    append_sensitive_debug(f"[SN][OCR_COLOR][TEXTS] {tag}{os.path.basename(sn_path)} | {json.dumps(texts, ensure_ascii=False)}")
    if text or concat:
        meta["ocr_text_found"] = True
    sn = extract_sn_from_text(concat or text)
    if sn:
        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
            _try_learn_sn_pattern(sn, meta)
            return (
                sn,
                f"[BARCODE_OCR_AGREE:ocr_color] {barcode_report.raw_text} | OCR={concat or text}",
                "barcode_ocr_agree",
                meta,
            )
        return sn, text, "ocr", meta
    agreed = _confirm_barcode_with_ocr(
        barcode_unmatched_sn,
        concat or text,
        barcode_report,
        meta,
        "ocr_color",
    )
    if agreed:
        return agreed

    img = load_and_preprocess(sn_path, roi_bottom=True)
    text, concat, texts = ocr_text_with_details(img)
    append_sensitive_debug(f"[SN][OCR_BIN] {tag}{os.path.basename(sn_path)} | text={text!r} concat={concat!r}")
    append_sensitive_debug(f"[SN][OCR_BIN][TEXTS] {tag}{os.path.basename(sn_path)} | {json.dumps(texts, ensure_ascii=False)}")
    if text or concat:
        meta["ocr_text_found"] = True
    sn = extract_sn_from_text(concat or text)
    if sn:
        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
            _try_learn_sn_pattern(sn, meta)
            return (
                sn,
                f"[BARCODE_OCR_AGREE:ocr_bin] {barcode_report.raw_text} | OCR={concat or text}",
                "barcode_ocr_agree",
                meta,
            )
        return sn, text, "ocr_bin", meta
    agreed = _confirm_barcode_with_ocr(
        barcode_unmatched_sn,
        concat or text,
        barcode_report,
        meta,
        "ocr_bin",
    )
    if agreed:
        return agreed

    top_text, top_concat = ocr_sn_top_text(sn_path)
    append_sensitive_debug(f"[SN][OCR_TOP] {tag}{os.path.basename(sn_path)} | text={top_text!r} concat={top_concat!r}")
    if top_text or top_concat:
        meta["ocr_text_found"] = True
    sn = extract_sn_from_text(top_concat or top_text)
    if sn:
        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
            _try_learn_sn_pattern(sn, meta)
            return (
                sn,
                f"[BARCODE_OCR_AGREE:ocr_top] {barcode_report.raw_text} | OCR={top_concat or top_text}",
                "barcode_ocr_agree",
                meta,
            )
        return sn, top_text, "ocr_top", meta
    agreed = _confirm_barcode_with_ocr(
        barcode_unmatched_sn,
        top_concat or top_text,
        barcode_report,
        meta,
        "ocr_top",
    )
    if agreed:
        return agreed

    if barcode_report.results:
        if barcode_report.status == "ambiguous":
            return "", f"[BARCODE_AMBIGUOUS] {barcode_raw}", "barcode_ambiguous", meta
        if barcode_report.status == "parse_failure":
            if barcode_unmatched_sn:
                return (
                    barcode_unmatched_sn,
                    f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {barcode_report.raw_text}",
                    "barcode_unmatched",
                    meta,
                )
            return "", f"[BARCODE_PARSE_FAIL] {barcode_raw}", "barcode_parse_fail", meta
        return "", f"[BARCODE] {barcode_raw}", "barcode_no_match", meta

    if meta["ocr_text_found"]:
        return "", top_text or text, "ocr_no_match", meta
    return "", text, "none", meta


def label_key(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    for suffix in ("__model", "__sn", "__FAILED"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _stage2_root():
    model_parent = os.path.dirname(os.path.abspath(MODEL_CROP_DIR))
    sn_parent = os.path.dirname(os.path.abspath(SN_CROP_DIR))
    if model_parent == sn_parent:
        return model_parent
    return os.path.commonpath([model_parent, sn_parent])


def _manifest_path():
    return os.path.join(_stage2_root(), "manifest.jsonl")


def _load_manifest_records():
    records = {}
    path = _manifest_path()
    if not os.path.isfile(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid manifest JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Invalid manifest row at {path}:{line_no}: expected object")
            label_id = item.get("label_id")
            if not label_id:
                raise ValueError(f"Manifest row missing label_id at {path}:{line_no}")
            record = records.setdefault(label_id, {})
            if item.get("label_crop"):
                if not os.path.isfile(item["label_crop"]):
                    raise FileNotFoundError(f"Manifest label_crop is missing at {path}:{line_no}: {item['label_crop']}")
                record["label_crop"] = item["label_crop"]
            if item.get("model_path"):
                if not os.path.isfile(item["model_path"]):
                    raise FileNotFoundError(f"Manifest model_path is missing at {path}:{line_no}: {item['model_path']}")
                record["model_path"] = item["model_path"]
            if item.get("sn_path"):
                if not os.path.isfile(item["sn_path"]):
                    raise FileNotFoundError(f"Manifest sn_path is missing at {path}:{line_no}: {item['sn_path']}")
                record["sn_path"] = item["sn_path"]
            original_path = item.get("original_image_path") or item.get("image_path")
            if original_path:
                if not os.path.isfile(original_path):
                    raise FileNotFoundError(f"Manifest original image is missing at {path}:{line_no}: {original_path}")
                record["original_image_path"] = original_path
    return records


# ===================== MAIN =====================
def main(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=None, log_level="info"):
    set_log_level(log_level)
    mask_raw = _env_flag("SCAN2_MASK_RAW") and not _env_flag("SCAN2_UNSAFE_RAW")
    model_barcode = _env_flag_default("SCAN2_MODEL_BARCODE", True)
    configure_paths(
        out_dir=out_dir,
        model_dir=model_dir,
        sn_dir=sn_dir,
        out_jsonl=out_jsonl,
        debug_log=debug_log,
    )
    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSONL)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(DEBUG_LOG_PATH)) or ".", exist_ok=True)
    start_debug_run()
    records = _load_manifest_records()
    model_files = []
    sn_files = []
    stats = {
        "records": 0,
        "model_total": 0,
        "model_success": 0,
        "model_barcode_hits": 0,
        "model_barcode_hit_rate": 0.0,
        "model_ocr_recoveries": 0,
        "sn_total": 0,
        "sn_attempted": 0,
        "sn_success": 0,
        "sn_barcode_attempts": 0,
        "sn_barcode_hits": 0,
        "sn_barcode_ocr_agree": 0,
        "sn_barcode_hit_rate": 0.0,
        "sn_ocr_recoveries": 0,
        "sn_barcode_parse_failures": 0,
        "sn_barcode_decoder_misses": 0,
        "sn_barcode_ambiguous": 0,
        "sn_barcode_quality_rejects": 0,
        "regex_fail": 0,
        "barcode_fail": 0,
        "ocr_fail": 0,
    }

    if os.path.isdir(MODEL_CROP_DIR):
        for fname in os.listdir(MODEL_CROP_DIR):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTS:
                continue
            model_files.append(fname)
            key = label_key(fname)
            records.setdefault(key, {})["model_path"] = os.path.join(MODEL_CROP_DIR, fname)

    if os.path.isdir(SN_CROP_DIR):
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
            sn_meta = {
                "barcode_found": False,
                "ocr_text_found": False,
                "barcode_status": "not_attempted",
                "barcode_attempts": 0,
                "barcode_decoded_count": 0,
            }

            if "model_path" in item:
                model_code, model_raw, model_src = recognize_model(
                    item["model_path"],
                    label_id=key,
                    use_barcode=model_barcode,
                )
                stats["model_total"] += 1
                if model_code:
                    stats["model_success"] += 1
                if model_src == "barcode":
                    stats["model_barcode_hits"] += 1
                elif model_src.startswith("ocr"):
                    stats["model_ocr_recoveries"] += 1

            sn_input_available = bool(item.get("sn_path") or item.get("label_crop"))
            if sn_input_available:
                sn_code, sn_raw, sn_src, sn_meta = recognize_sn(
                    item.get("sn_path", ""),
                    label_id=key,
                    label_path=item.get("label_crop", ""),
                    original_path=item.get("original_image_path", ""),
                )

            if sn_code.startswith("4E25A017") and model_code in {"", "S380-S8P", "S380", "S380-", "S380-S"}:
                model_code = "S380-S8P2T"
                model_src = f"{model_src}+sn_hint" if model_src else "sn_hint"

            if sn_code.startswith("4E25B0") and model_code in {"", "S380-S8P", "S380", "S380-", "S380-S"}:
                model_code = "S380-S8P2T"
                model_src = f"{model_src}+sn_hint" if model_src else "sn_hint"

            if model_code == "S380-S8P2T" and sn_code and not sn_code.startswith("4E25A017"):
                append_sensitive_debug(f"[WARN] {key} model= S380-S8P2T but sn={sn_code}")

            out = {
                "label_id": key,
                "model": model_code,
                "sn": sn_code,
                "model_raw": _mask_sensitive_text(model_raw) if mask_raw else model_raw,
                "sn_raw": _mask_sensitive_text(sn_raw) if mask_raw else sn_raw,
                "model_src": model_src,
                "sn_src": sn_src,
                "sn_barcode_status": sn_meta.get("barcode_status", "not_attempted"),
                "sn_barcode_attempts": sn_meta.get("barcode_attempts", 0),
                "sn_barcode_decoded_count": sn_meta.get("barcode_decoded_count", 0),
                "sn_barcode_sources": sn_meta.get("barcode_sources", []),
                "sn_barcode_source_regions": sn_meta.get("barcode_source_regions", []),
                "sn_barcode_decoder_names": sn_meta.get("barcode_decoder_names", []),
                "sn_barcode_ambiguous_sns": sn_meta.get("barcode_ambiguous_sns", []),
            }

            _log(
                f"[{key}] "
                f"MODEL={model_code} (M_SRC={model_src}) | "
                f"SN={sn_code} (SN_SRC={sn_src})",
                "info",
            )

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["records"] += 1

            if sn_input_available:
                stats["sn_total"] += 1
                barcode_status = sn_meta.get("barcode_status", "not_attempted")
                stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
                if sn_src in {"barcode", "barcode_unmatched"}:
                    stats["sn_barcode_hits"] += 1
                elif sn_src == "barcode_ocr_agree":
                    stats["sn_barcode_ocr_agree"] += 1
                elif sn_src.startswith("ocr") and barcode_status in {
                    "decoder_miss",
                    "parse_failure",
                    "ambiguous",
                    "quality_reject",
                }:
                    stats["sn_ocr_recoveries"] += 1
                if barcode_status == "parse_failure":
                    stats["sn_barcode_parse_failures"] += 1
                elif barcode_status == "decoder_miss":
                    stats["sn_barcode_decoder_misses"] += 1
                elif barcode_status == "ambiguous":
                    stats["sn_barcode_ambiguous"] += 1
                elif barcode_status == "quality_reject":
                    stats["sn_barcode_quality_rejects"] += 1
                if sn_meta.get("barcode_found") or sn_meta.get("ocr_text_found"):
                    stats["sn_attempted"] += 1
                if sn_code:
                    stats["sn_success"] += 1
                else:
                    if sn_meta.get("barcode_found") or sn_meta.get("ocr_text_found"):
                        stats["regex_fail"] += 1
                    if not sn_meta.get("barcode_found"):
                        stats["barcode_fail"] += 1
                    if not sn_meta.get("ocr_text_found"):
                        stats["ocr_fail"] += 1

    if stats["sn_total"]:
        stats["sn_barcode_hit_rate"] = stats["sn_barcode_hits"] / float(stats["sn_total"])
    if stats["model_total"]:
        stats["model_barcode_hit_rate"] = stats["model_barcode_hits"] / float(stats["model_total"])

    return stats


if __name__ == "__main__":
    main()
