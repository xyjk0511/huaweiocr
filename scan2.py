import os
import re
import json
import cv2
import tempfile
import concurrent.futures
import threading
import time
import numpy as np
from pathlib import Path
from barcode import decode_small_patch
from app_paths import ensure_models_installed, get_user_data_dir  # noqa: F401  (kept as scan2 module attribute for back-compat)
from envutil import env_flag_default as _env_flag_default
from sn_barcode import (
    SN12_BODY_PATTERN,  # noqa: F401  (kept as scan2 module attribute for back-compat)
    SN12_RE,
    SN20_RE,
    extract_sn_from_payload,
    scan_sn_barcodes,
)
from huaweiocr.core.extract import (
    BAD_MODEL_WORDS,
    KNOWN_MODEL_CODES,
    KNOWN_MODEL_CODES_UPPER,
    MODEL_CODE_ACCEPT_RE,
    MODEL_LINE_RE,
    PART_NO_MODEL_MAP,
    PART_NO_RE,  # noqa: F401  (kept as scan2 module attribute for back-compat)
    S380_S8P2T_NOISY_RE,
    S380_S8P2T_RE,
    SN_OCR_BOUNDED_RES,  # noqa: F401  (kept as scan2 module attribute for back-compat)
    UNKNOWN_SN_NON_PREFIX_RE,  # noqa: F401  (kept as scan2 module attribute for back-compat)
    _clean_code,
    _extract_sn_from_ocr_text_bounds,  # noqa: F401  (kept as scan2 module attribute for back-compat)
    _extract_unknown_sn_candidates,
    _normalize_part_no,
    _unknown_model_candidate_is_reasonable,
    extract_part_numbers_from_text,
    extract_sn_from_barcode_candidate,
    extract_sn_from_text,
    filter_sn_lines,  # noqa: F401  (kept as scan2 module attribute for back-compat)
    normalize_model,
)

# Simple log gating for CLI usage.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
LOG_SINK = None

def set_log_level(level: str) -> None:
    global LOG_LEVEL
    LOG_LEVEL = (level or "info").lower()

def set_log_sink(sink) -> None:
    global LOG_SINK
    LOG_SINK = sink
    if OCR_MODULE is not None and hasattr(OCR_MODULE, "set_log_sink"):
        try:
            OCR_MODULE.set_log_sink(sink)
        except Exception:
            pass

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
PART_NO_CROP_DIR = r"stage2_fields\part_no"
OUT_JSONL = r"model_sn_ocr.jsonl"
DEBUG_LOG_PATH = r"debug_ocr_barcode.log"

SN_TEXT_ROI_TOP_RATIO = 0.0

MAX_TARGET_W = 1200
MAX_SCALE = 4.0

SN_TARGET_WIDTHS = [1000, 1400, 1800]
SN_MAX_SCALE = 4.0
PART_NO_TARGET_WIDTHS = [650, 900, 1200]
PART_NO_MAX_SCALE = 4.0
PART_NO_PIXEL_REPAIR_MAX_SCORE = float(os.environ.get("SCAN2_PART_NO_PIXEL_REPAIR_MAX_SCORE", "0.17"))
PART_NO_PIXEL_REPAIR_MIN_MARGIN = float(os.environ.get("SCAN2_PART_NO_PIXEL_REPAIR_MIN_MARGIN", "0.010"))
PART_NO_STRIPE_RESCUE_ENABLED = os.environ.get("SCAN2_PART_NO_STRIPE_RESCUE", "1").strip().lower() not in {"0", "false", "no"}
SCAN_BARCODE_MAX_AUTO_WORKERS = 8
SCAN_OCR_MAX_AUTO_WORKERS = 1

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PART_NO_PAYLOAD_RE = re.compile(r"(?:^|[^0-9A-Z])((?:500|980)\d{5})(?=[^0-9A-Z]|[A-Z]{2,}|$)", re.I)

def configure_paths(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=None):
    global MODEL_CROP_DIR, SN_CROP_DIR, PART_NO_CROP_DIR, OUT_JSONL, DEBUG_LOG_PATH
    if out_dir:
        MODEL_CROP_DIR = os.path.join(out_dir, "stage2_fields", "model")
        SN_CROP_DIR = os.path.join(out_dir, "stage2_fields", "sn")
        PART_NO_CROP_DIR = os.path.join(out_dir, "stage2_fields", "part_no")
        OUT_JSONL = os.path.join(out_dir, "model_sn_ocr.jsonl")
        DEBUG_LOG_PATH = os.path.join(out_dir, "debug_ocr_barcode.log")
    if model_dir:
        MODEL_CROP_DIR = model_dir
        PART_NO_CROP_DIR = os.path.join(os.path.dirname(os.path.abspath(model_dir)), "part_no")
    if sn_dir:
        SN_CROP_DIR = sn_dir
        if not model_dir:
            PART_NO_CROP_DIR = os.path.join(os.path.dirname(os.path.abspath(sn_dir)), "part_no")
    if out_jsonl:
        OUT_JSONL = out_jsonl
    if debug_log:
        DEBUG_LOG_PATH = debug_log


def display_source(value: str) -> str:
    src = str(value or "")
    if src == "barcode":
        return "扫描条形码"
    if src == "barcode_ocr_consensus":
        return "条码+文字识别一致"
    if src == "barcode_visual":
        return "条形码视觉校验"
    if src in {"ocr_file", "ocr_color", "ocr_bin", "ocr_top"}:
        return "文字识别"
    if src == "ocr_no_match":
        return "文字识别未匹配"
    if src.startswith("ocr"):
        return "文字识别"
    if src == "barcode_ambiguous":
        return "条形码结果冲突"
    if src == "barcode_parse_fail":
        return "条形码解析失败"
    if src == "barcode_quality_reject":
        return "条形码质量不足"
    if src == "barcode_decoder_miss":
        return "未扫到条形码"
    if src == "barcode_no_match":
        return "条形码未匹配"
    if src in {"part_no_hint", "part_no_barcode", "part_no_learned"}:
        return "Part No模板"
    if src.startswith("part_no"):
        return "Part No未匹配"
    if src == "missing":
        return "缺失"
    if src == "none":
        return "未识别"
    if "+sn_hint" in src:
        return display_source(src.replace("+sn_hint", "")) + "+SN辅助判断"
    return src

# ===================== OCR =====================
OCR_MODULE = None
OCR_ENGINE = None
OCR_ENGINE_LOCK = threading.Lock()
OCR_INFERENCE_LOCK = threading.Lock()
OCR_PREWARM_LOCK = threading.Lock()
OCR_THREAD_LOCAL = threading.local()
OCR_THREAD_LOCAL_MODE = False
OCR_PREWARM_STARTED = False
OCR_PREWARM_DONE = False

OCR_WARMUP_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02"
    b"\xfeA\xe2\xdf\x16\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _load_ocr_module():
    global OCR_MODULE
    if OCR_MODULE is None:
        import ocr as ocr_module

        OCR_MODULE = ocr_module
        if hasattr(OCR_MODULE, "set_log_sink"):
            try:
                OCR_MODULE.set_log_sink(LOG_SINK)
            except Exception:
                pass
    return OCR_MODULE


def init_ocr():
    return _load_ocr_module().init_ocr()


def ocr_one_image(ocr_engine, img):
    return _load_ocr_module().ocr_one_image(ocr_engine, img)


def get_ocr_engine():
    global OCR_ENGINE
    if OCR_THREAD_LOCAL_MODE:
        engine = getattr(OCR_THREAD_LOCAL, "engine", None)
        if engine is None:
            with OCR_ENGINE_LOCK:
                engine = init_ocr()
            OCR_THREAD_LOCAL.engine = engine
        return engine

    with OCR_ENGINE_LOCK:
        if OCR_ENGINE is None:
            OCR_ENGINE = init_ocr()
        return OCR_ENGINE


def _ocr_one_image_thread_safe(ocr_engine, img):
    if OCR_THREAD_LOCAL_MODE:
        return ocr_one_image(ocr_engine, img)
    with OCR_INFERENCE_LOCK:
        return ocr_one_image(ocr_engine, img)


def _write_ocr_warmup_image(path: str) -> None:
    try:
        img = np.full((80, 260, 3), 255, dtype=np.uint8)
        cv2.putText(
            img,
            "Model: AP162E",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            getattr(cv2, "LINE_AA", 16),
        )
        if cv2.imwrite(path, img):
            return
    except Exception:
        pass

    with open(path, "wb") as f:
        f.write(OCR_WARMUP_PNG)


def _run_ocr_warmup_probe(ocr_engine) -> None:
    fd, path = tempfile.mkstemp(prefix="huaweiocr_ocr_warmup_", suffix=".png")
    os.close(fd)
    try:
        _write_ocr_warmup_image(path)
        _ocr_one_image_thread_safe(ocr_engine, path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def prewarm_ocr_engine(run_probe: bool = True, log=None) -> bool:
    global OCR_PREWARM_STARTED, OCR_PREWARM_DONE

    with OCR_PREWARM_LOCK:
        if OCR_PREWARM_DONE:
            return True
        if OCR_PREWARM_STARTED:
            return False
        OCR_PREWARM_STARTED = True

    started_at = time.perf_counter()
    try:
        ocr_engine = get_ocr_engine()
        if run_probe:
            _run_ocr_warmup_probe(ocr_engine)
    except Exception as exc:
        with OCR_PREWARM_LOCK:
            OCR_PREWARM_STARTED = False
            OCR_PREWARM_DONE = False
        append_debug(f"[OCR][PREWARM][ERROR] {exc.__class__.__name__}: {exc}")
        if log:
            log(f"OCR预热失败，将在识别时再初始化：{exc.__class__.__name__}: {exc}")
        return False

    elapsed = time.perf_counter() - started_at
    with OCR_PREWARM_LOCK:
        OCR_PREWARM_DONE = True
    append_debug(f"[OCR][PREWARM] done elapsed={elapsed:.2f}s probe={run_probe}")
    if log:
        log(f"OCR预热完成（{elapsed:.1f}s）")
    return True


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


def raw_result_fields_are_masked() -> bool:
    return not (_env_flag("SCAN2_UNSAFE_RAW") or _env_flag("HUAWEIOCR_UNSAFE_RAW"))


def _env_int(names, default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise RuntimeError(f"{name} must be a positive integer.") from exc
        if value < 1:
            raise RuntimeError(f"{name} must be a positive integer.")
        return value
    return default


def _auto_scan_workers(kind: str) -> int:
    cpu_count = os.cpu_count() or 2
    if kind == "barcode":
        return min(max(2, cpu_count), SCAN_BARCODE_MAX_AUTO_WORKERS)
    if kind == "ocr":
        return min(max(1, cpu_count // 4), SCAN_OCR_MAX_AUTO_WORKERS)
    return 1


def scan_worker_count(kind: str) -> int:
    kind = str(kind or "").strip().lower()
    if kind not in {"barcode", "ocr"}:
        raise RuntimeError("scan worker kind must be 'barcode' or 'ocr'.")
    if not _env_flag_default("SCAN2_PARALLEL", True):
        return 1
    return _env_int(
        [f"SCAN2_{kind.upper()}_WORKERS", "SCAN2_WORKERS"],
        _auto_scan_workers(kind),
    )


def scan_ocr_fallback_enabled() -> bool:
    return _env_flag_default("SCAN2_OCR_FALLBACK", True)


def part_no_ocr_fallback_enabled() -> bool:
    return _env_flag_default("SCAN2_PART_NO_OCR_FALLBACK", scan_ocr_fallback_enabled())


def scan_label_with_sn_enabled() -> bool:
    return _env_flag_default("SCAN2_SCAN_LABEL_WITH_SN", True)


def scan_label_without_sn_enabled() -> bool:
    return _env_flag_default("SCAN2_SCAN_LABEL_WITHOUT_SN", True)


def scan_part_no_first_enabled() -> bool:
    return _env_flag_default("SCAN2_PART_NO_FIRST", True)


def scan_progress_log_enabled() -> bool:
    raw = os.environ.get("SCAN2_PROGRESS_LOG")
    if raw is None or str(raw).strip() == "":
        return bool(LOG_SINK)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _scan_job_label(kind: str, key: str) -> str:
    kind = str(kind or "")
    if kind == "part_no_model":
        return f"{key} PartNo->型号"
    if kind == "model":
        return f"{key} 型号"
    if kind == "sn":
        return f"{key} SN"
    return f"{key} {kind}"


def _barcode_job_result_summary(kind: str, result) -> str:
    if kind == "part_no_model":
        model_code, _raw, src = result
        if model_code:
            return f"命中 {model_code}（{display_source(src)}）"
        return f"未命中（{display_source(src)}）"
    if kind == "model":
        model_code, _raw, src = result
        if model_code:
            return f"命中 {model_code}（{display_source(src)}）"
        return f"未命中（{display_source(src)}）"
    sn_code, _raw, src, _meta, _report = result
    if sn_code:
        return f"命中 {sn_code}（{display_source(src)}）"
    return f"未命中（{display_source(src)}）"


def _ocr_job_result_summary(kind: str, result) -> str:
    if kind == "model":
        model_code, _raw, src = result
        if model_code:
            return f"命中 {model_code}（{display_source(src)}）"
        return f"未命中（{display_source(src)}）"
    sn_code, _raw, src, _meta = result
    if sn_code:
        return f"命中 {sn_code}（{display_source(src)}）"
    return f"未命中（{display_source(src)}）"


def _progress_logger(prefix: str):
    last_done = {"value": 0}

    def _inner(done: int, total: int) -> None:
        done = int(done or 0)
        total = int(total or 0)
        if done <= last_done["value"]:
            return
        last_done["value"] = done
        _log(f"{prefix} {done}/{total}", "info")

    return _inner


def _map_ordered(items, fn, workers: int, progress=None):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        total = len(items)
        results = []
        for index, item in enumerate(items, 1):
            results.append(fn(item))
            if progress:
                progress(index, total)
        return results

    max_workers = min(workers, len(items))
    results = [None] * len(items)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(fn, item): index
            for index, item in enumerate(items)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            done += 1
            if progress:
                progress(done, len(items))
        return results


def _map_ocr_ordered(items, fn, workers: int, progress=None):
    global OCR_THREAD_LOCAL_MODE
    old_mode = OCR_THREAD_LOCAL_MODE
    OCR_THREAD_LOCAL_MODE = workers > 1
    try:
        return _map_ordered(items, fn, workers, progress=progress)
    finally:
        OCR_THREAD_LOCAL_MODE = old_mode


# ===================== UTILS =====================
def _path_has_non_ascii(path) -> bool:
    try:
        return any(ord(ch) > 127 for ch in os.fspath(path))
    except Exception:
        return False


def _read_image_unicode_safe(path, flags):
    if np is None or not hasattr(np, "fromfile") or not hasattr(cv2, "imdecode"):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data is None:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _read_image(path, flags=cv2.IMREAD_COLOR):
    if _path_has_non_ascii(path):
        img = _read_image_unicode_safe(path, flags)
        if img is not None:
            return img
    img = cv2.imread(path, flags)
    if img is not None:
        return img
    return _read_image_unicode_safe(path, flags)

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
    texts, concat = _ocr_one_image_thread_safe(ocr_engine, img)
    if not texts:
        return "", concat
    text = " ".join(t["text"] for t in texts)
    return text, concat


def ocr_text_with_details(img):
    ocr_engine = get_ocr_engine()
    texts, concat = _ocr_one_image_thread_safe(ocr_engine, img)
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


def _lines_contain_part_no(lines: list[str]) -> bool:
    return any(PART_NO_PAYLOAD_RE.search(str(line or "").upper()) for line in lines or [])


def _part_no_pixel_repair(img_path: str) -> list[str]:
    if os.environ.get("SCAN2_PART_NO_PIXEL_REPAIR", "1").strip().lower() in {"0", "false", "no"}:
        return []
    try:
        import linear_barcode_repair as repair

        results, _profiles, _strategy = repair.decode_image(
            Path(img_path),
            mode="digits",
            lengths_arg="8",
            accept_score=PART_NO_PIXEL_REPAIR_MAX_SCORE,
            max_profiles=8,
        )
    except Exception as exc:
        _log(f"PartNo像素恢复失败：{exc.__class__.__name__}: {exc}", "debug")
        return []

    if not results:
        return []
    best = results[0]
    text = str(best.text or "").strip()
    if not PART_NO_PAYLOAD_RE.fullmatch(text):
        return []
    margin = float(results[1].score - best.score) if len(results) > 1 else 999.0
    if best.score > PART_NO_PIXEL_REPAIR_MAX_SCORE:
        return []
    if margin < PART_NO_PIXEL_REPAIR_MIN_MARGIN:
        return []
    return [text]


def _part_no_stripe_rescue(img_path: str) -> list[str]:
    if not PART_NO_STRIPE_RESCUE_ENABLED:
        return []

    try:
        import part_no_barcode_rescue as rescue
    except Exception as exc:
        _log(f"PartNo条纹恢复模块加载失败：{exc.__class__.__name__}: {exc}", "debug")
        return []

    img = _read_image(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return []

    try:
        results = rescue.decode_part_no_candidates(img, lengths=(8,), charset="digits", max_profiles=8)
    except Exception as exc:
        _log(f"PartNo条纹恢复失败：{exc.__class__.__name__}: {exc}", "debug")
        return []

    valid = []
    for item in results:
        text = str(getattr(item, "text", "") or "").strip()
        if PART_NO_PAYLOAD_RE.fullmatch(text):
            valid.append(item)

    if not valid:
        return []

    best = valid[0]
    margin = float(valid[1].score - best.score) if len(valid) > 1 else 999.0
    if best.score > PART_NO_PIXEL_REPAIR_MAX_SCORE:
        return []
    if margin < PART_NO_PIXEL_REPAIR_MIN_MARGIN:
        return []
    return [str(best.text)]


def read_part_no_barcodes(img_path: str, *, allow_pixel_repair: bool = True) -> list[str]:
    lines = read_barcodes(img_path)
    if lines and not allow_pixel_repair:
        return lines
    if _lines_contain_part_no(lines):
        return lines

    img = _read_image(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return lines

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return lines

    seen = set(lines)
    out = list(lines)
    views = []

    pad_x = max(24, int(w * 0.14))
    pad_y = max(8, int(h * 0.08))
    views.append(
        (
            "qzpad",
            cv2.copyMakeBorder(
                img,
                pad_y,
                pad_y,
                pad_x,
                pad_x,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            ),
        )
    )

    for tw in PART_NO_TARGET_WIDTHS:
        if tw <= w:
            continue
        scale = min(tw / float(w), PART_NO_MAX_SCALE)
        if scale <= 1.01:
            continue
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        views.append((f"x{scale:.2f}", resized))

        r_pad_x = max(32, int(new_w * 0.10))
        r_pad_y = max(10, int(new_h * 0.05))
        views.append(
            (
                f"x{scale:.2f}_qzpad",
                cv2.copyMakeBorder(
                    resized,
                    r_pad_y,
                    r_pad_y,
                    r_pad_x,
                    r_pad_x,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                ),
            )
        )

    for tag, view in views:
        try:
            codes = decode_barcodes_with_dbr(view, debug_name=f"part_no[{tag}]")
        except Exception:
            codes = []
        for code in codes:
            if not code or code in seen:
                continue
            seen.add(code)
            out.append(code)
        if _lines_contain_part_no(out):
            break

    if allow_pixel_repair and not _lines_contain_part_no(out):
        repaired = _part_no_pixel_repair(img_path)
        for code in repaired:
            if code and code not in seen:
                seen.add(code)
                out.append(code)
    if allow_pixel_repair and not _lines_contain_part_no(out):
        rescued = _part_no_stripe_rescue(img_path)
        for code in rescued:
            if code and code not in seen:
                seen.add(code)
                out.append(code)

    return out


# ========= MODEL RULES =========

PART_NO_MODEL_MAP_LOCK = threading.Lock()
PART_NO_MODEL_MAP_CACHE_PATH = None
PART_NO_MODEL_MAP_CACHE = {}
LEARNED_MODEL_CODES_LOCK = threading.Lock()
LEARNED_MODEL_CODES_CACHE_PATH = None
LEARNED_MODEL_CODES_CACHE = set()
def part_no_model_map_path() -> str:
    override = os.environ.get("SCAN2_PART_NO_MODEL_MAP_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(get_user_data_dir(), "part_no_model_map.json")


def learned_model_codes_path() -> str:
    override = os.environ.get("SCAN2_LEARNED_MODEL_CODES_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(get_user_data_dir(), "learned_model_codes.json")


def _part_no_entry_model(value) -> str:
    if isinstance(value, dict):
        value = value.get("model", "")
    model = normalize_model(str(value or ""))
    if model and model_code_is_plausible(model):
        return model
    return ""


def _coerce_part_no_model_map(raw) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        part_no = _normalize_part_no(key)
        if not part_no:
            continue
        model = _part_no_entry_model(value)
        if model:
            out[part_no] = model
    return out


def _load_extra_part_no_model_map() -> dict[str, str]:
    global PART_NO_MODEL_MAP_CACHE_PATH, PART_NO_MODEL_MAP_CACHE
    path = part_no_model_map_path()
    with PART_NO_MODEL_MAP_LOCK:
        if PART_NO_MODEL_MAP_CACHE_PATH == path:
            return dict(PART_NO_MODEL_MAP_CACHE)

        mapping = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    mapping = _coerce_part_no_model_map(json.load(f))
            except Exception as exc:
                append_debug(f"[PART_NO_MAP][LOAD_ERROR] {path}: {exc.__class__.__name__}: {exc}")
                mapping = {}

        PART_NO_MODEL_MAP_CACHE_PATH = path
        PART_NO_MODEL_MAP_CACHE = dict(mapping)
        return mapping


def get_part_no_model_map() -> dict[str, str]:
    mapping = dict(PART_NO_MODEL_MAP)
    mapping.update(_load_extra_part_no_model_map())
    return mapping


def _coerce_learned_model_codes(raw) -> set[str]:
    values = []
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict):
        values = raw.keys()
    out = set()
    for value in values:
        model = normalize_model(str(value or ""))
        if model and _unknown_model_candidate_is_reasonable(model):
            out.add(model)
    return out


def _load_learned_model_codes() -> set[str]:
    global LEARNED_MODEL_CODES_CACHE_PATH, LEARNED_MODEL_CODES_CACHE
    path = learned_model_codes_path()
    with LEARNED_MODEL_CODES_LOCK:
        if LEARNED_MODEL_CODES_CACHE_PATH == path:
            return set(LEARNED_MODEL_CODES_CACHE)

        codes = set()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    codes = _coerce_learned_model_codes(json.load(f))
            except Exception as exc:
                append_debug(f"[LEARNED_MODEL][LOAD_ERROR] {path}: {exc.__class__.__name__}: {exc}")
                codes = set()

        LEARNED_MODEL_CODES_CACHE_PATH = path
        LEARNED_MODEL_CODES_CACHE = set(codes)
        return set(codes)


def get_learned_model_codes() -> set[str]:
    return _load_learned_model_codes()


def save_learned_model_code(model: str, label_id: str = "", source: str = "") -> bool:
    global LEARNED_MODEL_CODES_CACHE_PATH, LEARNED_MODEL_CODES_CACHE
    model = normalize_model(model)
    if not model or not _unknown_model_candidate_is_reasonable(model):
        return False
    if model in KNOWN_MODEL_CODES or model.upper() in KNOWN_MODEL_CODES_UPPER:
        return False

    path = learned_model_codes_path()
    with LEARNED_MODEL_CODES_LOCK:
        codes = set()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    codes = _coerce_learned_model_codes(json.load(f))
            except Exception:
                codes = set()
        if model in codes:
            return False
        codes.add(model)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".learned_model_codes.",
            suffix=".json",
            dir=os.path.dirname(path),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(sorted(codes), f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        LEARNED_MODEL_CODES_CACHE_PATH = None
        LEARNED_MODEL_CODES_CACHE = set()
        append_debug(
            f"[LEARNED_MODEL][SAVE] label={label_id or '-'} model={model} source={source or '-'} path={path}"
        )
        return True


def first_part_no_from_chunks(*chunks) -> str:
    for chunk in chunks:
        part_numbers = extract_part_numbers_from_text(chunk or "")
        if part_numbers:
            return part_numbers[0]
    return ""


def save_part_no_model_mapping(
    part_no: str,
    model: str,
    label_id: str = "",
    source: str = "",
) -> bool:
    global PART_NO_MODEL_MAP_CACHE_PATH, PART_NO_MODEL_MAP_CACHE
    part_no = _normalize_part_no(part_no)
    model = normalize_model(model)
    if not part_no or not model or not model_code_is_plausible(model):
        return False
    if PART_NO_MODEL_MAP.get(part_no) == model:
        return False

    path = part_no_model_map_path()
    with PART_NO_MODEL_MAP_LOCK:
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}

        existing_model = _part_no_entry_model(data.get(part_no))
        if existing_model and existing_model != model and not _env_flag_default("SCAN2_PART_NO_MAP_OVERWRITE", False):
            append_debug(
                f"[PART_NO_MAP][CONFLICT] part_no={part_no} existing={existing_model} new={model} label={label_id}"
            )
            return False

        data[part_no] = {
            "model": model,
            "source": source,
            "label_id": label_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".part_no_model_map.",
            suffix=".json",
            dir=os.path.dirname(path),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        PART_NO_MODEL_MAP_CACHE_PATH = None
        PART_NO_MODEL_MAP_CACHE = {}
        return True


def model_from_part_no_text(text: str) -> tuple[str, str]:
    mapping = get_part_no_model_map()
    for part_no in extract_part_numbers_from_text(text):
        model = mapping.get(part_no)
        if model:
            return model, part_no
    return "", ""


def model_from_part_no_hint(raw_chunks) -> tuple[str, str]:
    for chunk in raw_chunks:
        model, part_no = model_from_part_no_text(chunk or "")
        if model:
            return model, part_no
    return "", ""


def model_code_is_plausible(code: str) -> bool:
    if not code:
        return False
    c = normalize_model(code)
    cu = c.upper()
    if c in KNOWN_MODEL_CODES or cu in KNOWN_MODEL_CODES_UPPER:
        return True
    if c in get_learned_model_codes():
        return True
    if any(word in cu for word in ("DESC", "MODEL", "CHINA", "MAC")):
        return False
    if _env_flag_default("SCAN2_ALLOW_UNKNOWN_MODELS", False):
        return bool(MODEL_CODE_ACCEPT_RE.fullmatch(cu)) or _unknown_model_candidate_is_reasonable(cu)
    return False


def extract_model_candidate_from_text(text: str, *, allow_unknown: bool = False) -> str:
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
        model = normalize_model(raw)
        if model_code_is_plausible(model) or (allow_unknown and _unknown_model_candidate_is_reasonable(model)):
            return model

    tokens = re.findall(r"[A-Z][A-Z0-9\-]{1,}", t)
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
    for best in cand:
        model = normalize_model(best)
        if model_code_is_plausible(model) or (allow_unknown and _unknown_model_candidate_is_reasonable(model)):
            return model
    return ""


def extract_model_from_text(text: str) -> str:
    return extract_model_candidate_from_text(text, allow_unknown=False)


def extract_model_from_ocr_result(text: str, concat: str) -> str:
    # Keep OCR spacing for MODEL first; concat can merge trailing description
    # text into the model token (for example "AP162E 9SC" -> "AP162E9SC").
    return extract_model_from_text(text) or extract_model_from_text(concat)


def extract_unknown_model_candidate(text: str) -> str:
    return extract_model_candidate_from_text(text, allow_unknown=True)


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
        if str(field or "").upper() == "PART_NO":
            lines = read_part_no_barcodes(img_path)
        else:
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


def try_model_from_part_no_crop(
    part_no_path: str,
    label_id: str = "",
    use_ocr: bool = False,
) -> tuple[str, str, str]:
    if not part_no_path:
        return "", "", "part_no_missing"
    entries = _scan_barcode_sources([("part_no", part_no_path)], label_id=label_id, field="PART_NO")
    for entry in entries:
        model, part_no = model_from_part_no_text(entry.get("data", ""))
        if model:
            return model, f"[PART_NO_BARCODE] {part_no}", "part_no_barcode"
    raw = _format_barcode_entries(entries)
    if raw:
        return "", raw, "part_no_no_match"

    if not use_ocr:
        return "", "", "part_no_no_barcode"

    if os.path.isfile(part_no_path):
        tag = f"{label_id} " if label_id else ""
        append_debug(f"[PART_NO][OCR] {tag}{os.path.basename(part_no_path)}")
        text, concat, texts = ocr_text_with_details(part_no_path)
        append_sensitive_debug(f"[PART_NO][OCR_FILE] {tag}{os.path.basename(part_no_path)} | {text!r}")
        append_sensitive_debug(
            f"[PART_NO][OCR_FILE][TEXTS] {tag}{os.path.basename(part_no_path)} | "
            f"{json.dumps(texts, ensure_ascii=False)}"
        )
        model, part_no = model_from_part_no_hint([text, concat])
        if model:
            return model, f"[PART_NO_OCR] {part_no}", "part_no_ocr"
        model = extract_model_from_ocr_result(text, concat)
        if model:
            return model, f"[PART_NO_OCR_MODEL] {model}", "part_no_ocr_model"
        if text or concat:
            return "", text or concat, "part_no_ocr_no_match"

    return "", "", "part_no_no_barcode"


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
    s = _clean_code(text)
    if extract_sn_from_payload(s) or SN20_RE.search(s) or SN12_RE.search(s):
        return ""
    if re.fullmatch(r"[0-9]{8,}", s):
        return ""
    return extract_model_from_text(text)


def _try_model_from_fast_zxing(img_path: str) -> tuple[str, str]:
    try:
        import zxingcpp
        import barcode as barcode_module
    except Exception:
        return "", ""

    img = _read_image(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return "", ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        gray = barcode_module.auto_rotate_to_horizontal(gray)
    except Exception:
        pass

    candidates = [("model_zxing_full", gray)]
    for fn_name, source in (
        ("crop_detected_barcode_band", "model_zxing_detected"),
        ("crop_bar_band", "model_zxing_midband"),
    ):
        fn = getattr(barcode_module, fn_name, None)
        if not fn:
            continue
        try:
            band = fn(gray)
        except Exception:
            band = None
        if band is not None and getattr(band, "size", 0):
            candidates.append((source, band))

    format_names = ("Code128", "Code39", "Code93", "ITF", "EAN13")
    formats = tuple(
        getattr(zxingcpp.BarcodeFormat, name)
        for name in format_names
        if hasattr(zxingcpp.BarcodeFormat, name)
    )
    seen = set()
    for source, candidate in candidates:
        for try_rotate in (False, True):
            try:
                decoded = zxingcpp.read_barcodes(
                    candidate,
                    formats=formats or None,
                    try_rotate=try_rotate,
                    try_downscale=False,
                    try_invert=True,
                    return_errors=False,
                )
            except Exception:
                decoded = []
            for item in decoded:
                text = getattr(item, "text", "") or ""
                if not text or text in seen:
                    continue
                seen.add(text)
                model = extract_model_from_barcode_candidate(text)
                if model:
                    append_sensitive_debug(f"[MODEL][BARCODE][{source}] {os.path.basename(img_path)} | {text}")
                    return model, text

    return "", ""


def try_model_from_barcode(img_path: str) -> tuple[str, str]:
    model, raw = _try_model_from_fast_zxing(img_path)
    if model:
        return model, raw

    entries = _scan_barcode_sources([("model", img_path)], field="MODEL")
    for ln in filter_model_lines([entry["data"] for entry in entries]):
        model = extract_model_from_barcode_candidate(ln)
        if model:
            return model, ln

    fallback_entries = _scan_model_barcode_band_entries(img_path)
    for ln in filter_model_lines([entry["data"] for entry in fallback_entries]):
        model = extract_model_from_barcode_candidate(ln)
        if model:
            return model, ln

    entries.extend(fallback_entries)
    if not entries:
        return "", ""

    for entry in entries:
        ln = entry["data"]
        model = extract_model_from_barcode_candidate(ln)
        if model:
            return model, ln

    return "", _format_barcode_entries(entries)


def model_from_sn_hint(sn_code: str, current_model: str = "") -> str:
    if sn_code.startswith(("4E25A017", "4E25B0", "4E264006")) and current_model in {
        "",
        "S380-S8P",
        "S380",
        "S380-",
        "S380-S",
    }:
        return "S380-S8P2T"
    return ""


def _scan_model_barcode_band_entries(img_path: str) -> list[dict]:
    try:
        import barcode as barcode_module
    except Exception:
        return []

    img = _read_image(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        gray = barcode_module.auto_rotate_to_horizontal(gray)
    except Exception:
        pass

    bands = []
    detect_band = getattr(barcode_module, "crop_detected_barcode_band", None)
    if detect_band:
        try:
            band = detect_band(gray)
            if band is not None and getattr(band, "size", 0):
                bands.append(("model_band", band))
        except Exception:
            pass

    crop_band = getattr(barcode_module, "crop_bar_band", None)
    if crop_band:
        try:
            band = crop_band(gray)
            if band is not None and getattr(band, "size", 0):
                bands.append(("model_midband", band))
        except Exception:
            pass

    entries = []
    seen = set()
    sharp_decoder = getattr(barcode_module, "decode_cli_sharp_variants", None)

    for source, band in bands:
        results = []
        if sharp_decoder:
            try:
                results = sharp_decoder(band, source, {"limit": 24, "calls": 0})
            except Exception:
                results = []
        for item in results:
            data = item.get("data", "") if isinstance(item, dict) else ""
            if not data or data in seen:
                continue
            seen.add(data)
            entries.append({"source": source, "data": data})

    for source, band in bands:
        try:
            if len(band.shape) == 2:
                band_bgr = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
            else:
                band_bgr = band
            info = decode_small_patch(band_bgr)
            results = info.get("results", []) if isinstance(info, dict) else []
        except Exception:
            results = []
        for item in results:
            data = item.get("data", "") if isinstance(item, dict) else ""
            if not data or data in seen:
                continue
            seen.add(data)
            entries.append({"source": source, "data": data})
    return entries


def _model_barcode_visual_text_variants(model_code: str) -> list[str]:
    code = str(model_code or "").strip()
    if not code:
        return []

    variants = [code]
    if code.upper().endswith("PRO") and not code.endswith("Pro"):
        variants.append(code[:-3] + "Pro")
    return list(dict.fromkeys(variants))


def _verify_model_barcode_visual(img_path: str, model_code: str) -> dict | None:
    try:
        import barcode as barcode_module
    except Exception:
        return None

    verifier = getattr(barcode_module, "verify_code128b_text_in_image", None)
    if verifier is None:
        return None

    img = _read_image(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    for text in _model_barcode_visual_text_variants(model_code):
        try:
            result = verifier(img, text)
        except Exception:
            result = None
        if result:
            return result
    return None


def recognize_model_barcode(model_path: str, label_id: str = ""):
    tag = f"{label_id} " if label_id else ""
    append_debug(f"[MODEL][BARCODE] {tag}{os.path.basename(model_path)}")
    m_from_bc, bc_raw = try_model_from_barcode(model_path)
    if m_from_bc:
        return m_from_bc, f"[BARCODE] {bc_raw}", "barcode"
    return "", bc_raw, "barcode_no_match"


def recognize_model_ocr(model_path: str, label_id: str = "", verify_barcode_visual: bool = False):
    tag = f"{label_id} " if label_id else ""
    append_debug(f"[MODEL][OCR] {tag}{os.path.basename(model_path)}")
    text, concat, texts = ocr_text_with_details(model_path)
    append_sensitive_debug(f"[MODEL][OCR_FILE] {tag}{os.path.basename(model_path)} | {text!r}")
    append_sensitive_debug(
        f"[MODEL][OCR_FILE][TEXTS] {tag}{os.path.basename(model_path)} | "
        f"{json.dumps(texts, ensure_ascii=False)}"
    )
    model_code = extract_model_from_ocr_result(text, concat)
    if model_code:
        if verify_barcode_visual:
            visual = _verify_model_barcode_visual(model_path, model_code)
            if visual:
                score = visual.get("score", 0.0)
                matched = visual.get("text", model_code)
                return model_code, f"[BARCODE_VISUAL] {matched} score={score:.3f}", "barcode_visual"
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
            if verify_barcode_visual:
                visual = _verify_model_barcode_visual(model_path, model_code)
                if visual:
                    score = visual.get("score", 0.0)
                    matched = visual.get("text", model_code)
                    return model_code, f"[BARCODE_VISUAL] {matched} score={score:.3f}", "barcode_visual"
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
        if verify_barcode_visual:
            visual = _verify_model_barcode_visual(model_path, model_code)
            if visual:
                score = visual.get("score", 0.0)
                matched = visual.get("text", model_code)
                return model_code, f"[BARCODE_VISUAL] {matched} score={score:.3f}", "barcode_visual"
        return model_code, text, "ocr_bin"
    return "", text, "none"


def recognize_model_label_ocr(label_path: str, label_id: str = ""):
    tag = f"{label_id} " if label_id else ""
    append_debug(f"[MODEL][LABEL_OCR] {tag}{os.path.basename(label_path)}")
    text, concat, texts = ocr_text_with_details(label_path)
    append_sensitive_debug(f"[MODEL][LABEL_OCR_FILE] {tag}{os.path.basename(label_path)} | {text!r}")
    append_sensitive_debug(
        f"[MODEL][LABEL_OCR_FILE][TEXTS] {tag}{os.path.basename(label_path)} | "
        f"{json.dumps(texts, ensure_ascii=False)}"
    )
    model_code = extract_model_from_ocr_result(text, concat)
    if model_code:
        return model_code, text or concat, "ocr_label"

    color_img = load_for_ocr_color(label_path)
    if color_img is not None:
        text, concat, texts = ocr_text_with_details(color_img)
        append_sensitive_debug(f"[MODEL][LABEL_OCR_COLOR] {tag}{os.path.basename(label_path)} | {text!r}")
        append_sensitive_debug(
            f"[MODEL][LABEL_OCR_COLOR][TEXTS] {tag}{os.path.basename(label_path)} | "
            f"{json.dumps(texts, ensure_ascii=False)}"
        )
        model_code = extract_model_from_ocr_result(text, concat)
        if model_code:
            return model_code, text or concat, "ocr_label_color"

    img = load_and_preprocess(label_path, roi_bottom=False)
    text, concat, texts = ocr_text_with_details(img)
    append_sensitive_debug(f"[MODEL][LABEL_OCR_BIN] {tag}{os.path.basename(label_path)} | {text!r}")
    append_sensitive_debug(
        f"[MODEL][LABEL_OCR_BIN][TEXTS] {tag}{os.path.basename(label_path)} | "
        f"{json.dumps(texts, ensure_ascii=False)}"
    )
    model_code = extract_model_from_ocr_result(text, concat)
    if model_code:
        return model_code, text or concat, "ocr_label_bin"
    return "", text or concat, "label_none"


def recognize_model(model_path: str, label_id: str = "", use_barcode: bool = False):
    tag = f"{label_id} " if label_id else ""
    append_debug(f"[MODEL] {tag}{os.path.basename(model_path)}")
    if use_barcode:
        model_code, raw, source = recognize_model_barcode(model_path, label_id=label_id)
        if model_code:
            return model_code, raw, source
    return recognize_model_ocr(
        model_path,
        label_id=label_id,
        verify_barcode_visual=use_barcode,
    )


def preprocess_sn_image(path: str):
    try:
        return load_and_preprocess(path, roi_bottom=True)
    except Exception:
        return None


def _sn_barcode_sources(
    sn_path: str = "",
    label_id: str = "",
    label_path: str = "",
    original_path: str = "",
):
    tag = f"{label_id} " if label_id else ""
    sources = []
    if sn_path:
        sources.append(("sn", sn_path))
    if label_path and label_path != sn_path:
        sources.append(("label", label_path))
    if original_path and original_path not in {sn_path, label_path}:
        append_debug(f"[SN] {tag}ignored original image barcode source for SN recognition")
    return sources


def _sn_barcode_failure_result(barcode_report, barcode_raw: str, meta: dict, has_sn_path: bool):
    status = getattr(barcode_report, "status", "decoder_miss")
    results = getattr(barcode_report, "results", []) or []
    if status == "ambiguous":
        return "", f"[BARCODE_AMBIGUOUS] {barcode_raw}", "barcode_ambiguous", meta
    if status == "parse_failure":
        return "", f"[BARCODE_PARSE_FAIL] {barcode_raw}", "barcode_parse_fail", meta
    if status == "quality_reject":
        return "", "[BARCODE_QUALITY_REJECT]", "barcode_quality_reject", meta
    if results:
        return "", f"[BARCODE] {barcode_raw}", "barcode_no_match", meta
    if has_sn_path:
        return "", "", "barcode_decoder_miss", meta
    return "", "", "none", meta


def _unknown_sn_candidate_from_barcode_report(barcode_report) -> str:
    if barcode_report is None:
        return ""
    candidates = []
    for result in getattr(barcode_report, "results", []) or []:
        candidates.extend(_extract_unknown_sn_candidates(getattr(result, "raw_text", "")))
    for candidate in candidates:
        return candidate
    return ""


def _unknown_sn_consensus(barcode_report, *texts: str) -> str:
    barcode_candidate = _unknown_sn_candidate_from_barcode_report(barcode_report)
    if not barcode_candidate:
        return ""
    for text in texts:
        if barcode_candidate in _extract_unknown_sn_candidates(text):
            return barcode_candidate
    return ""


def _recognize_sn_barcode(
    sn_path: str = "",
    label_id: str = "",
    label_path: str = "",
    original_path: str = "",
):
    tag = f"{label_id} " if label_id else ""
    append_debug(
        f"[SN][BARCODE] {tag}sn={os.path.basename(sn_path) if sn_path else '[missing]'} "
        f"label={os.path.basename(label_path) if label_path else '[missing]'} "
        f"original={os.path.basename(original_path) if original_path else '[missing]'}"
    )
    sources = _sn_barcode_sources(
        sn_path=sn_path,
        label_id=label_id,
        label_path=label_path,
        original_path=original_path,
    )

    barcode_report = _scan_sn_barcode_report(sources, label_id=label_id, allow_early_exit=True)
    meta = barcode_report.to_meta()
    meta["ocr_text_found"] = False
    barcode_raw = _format_barcode_report(barcode_report)
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
            barcode_report,
        )

    sn, raw, source, meta = _sn_barcode_failure_result(
        barcode_report,
        barcode_raw,
        meta,
        has_sn_path=bool(sn_path),
    )
    return sn, raw, source, meta, barcode_report


def recognize_sn_barcode(
    sn_path: str = "",
    label_id: str = "",
    label_path: str = "",
    original_path: str = "",
):
    return _recognize_sn_barcode(
        sn_path=sn_path,
        label_id=label_id,
        label_path=label_path,
        original_path=original_path,
    )[:4]


def recognize_sn_ocr_after_barcode(
    sn_path: str = "",
    label_id: str = "",
    barcode_report=None,
    meta: dict | None = None,
):
    tag = f"{label_id} " if label_id else ""
    if barcode_report is None:
        barcode_report = _scan_sn_barcode_report([("sn", sn_path)] if sn_path else [], label_id=label_id)
    if meta is None:
        meta = barcode_report.to_meta()
        meta["ocr_text_found"] = False
    barcode_raw = _format_barcode_report(barcode_report)
    if not sn_path:
        return _sn_barcode_failure_result(barcode_report, barcode_raw, meta, has_sn_path=False)

    append_debug(f"[SN][OCR] {tag}{os.path.basename(sn_path)}")
    last_text = ""
    color_img = load_for_ocr_color(sn_path)
    if color_img is None:
        return _sn_barcode_failure_result(barcode_report, barcode_raw, meta, has_sn_path=True)

    text, concat, texts = ocr_text_with_details(color_img)
    last_text = text
    append_sensitive_debug(f"[SN][OCR_COLOR] {tag}{os.path.basename(sn_path)} | text={text!r} concat={concat!r}")
    append_sensitive_debug(f"[SN][OCR_COLOR][TEXTS] {tag}{os.path.basename(sn_path)} | {json.dumps(texts, ensure_ascii=False)}")
    if text or concat:
        meta["ocr_text_found"] = True
    sn = extract_sn_from_text(text) or extract_sn_from_text(concat or text)
    if sn:
        return sn, text, "ocr", meta
    sn = _unknown_sn_consensus(barcode_report, text, concat)
    if sn:
        return sn, text or concat, "barcode_ocr_consensus", meta

    img = load_and_preprocess(sn_path, roi_bottom=True)
    text, concat, texts = ocr_text_with_details(img)
    last_text = text
    append_sensitive_debug(f"[SN][OCR_BIN] {tag}{os.path.basename(sn_path)} | text={text!r} concat={concat!r}")
    append_sensitive_debug(f"[SN][OCR_BIN][TEXTS] {tag}{os.path.basename(sn_path)} | {json.dumps(texts, ensure_ascii=False)}")
    if text or concat:
        meta["ocr_text_found"] = True
    sn = extract_sn_from_text(text) or extract_sn_from_text(concat or text)
    if sn:
        return sn, text, "ocr_bin", meta
    sn = _unknown_sn_consensus(barcode_report, text, concat)
    if sn:
        return sn, text or concat, "barcode_ocr_consensus", meta

    top_text, top_concat = ocr_sn_top_text(sn_path)
    append_sensitive_debug(f"[SN][OCR_TOP] {tag}{os.path.basename(sn_path)} | text={top_text!r} concat={top_concat!r}")
    if top_text or top_concat:
        meta["ocr_text_found"] = True
    sn = extract_sn_from_text(top_text) or extract_sn_from_text(top_concat or top_text)
    if sn:
        return sn, top_text, "ocr_top", meta
    sn = _unknown_sn_consensus(barcode_report, top_text, top_concat)
    if sn:
        return sn, top_text or top_concat, "barcode_ocr_consensus", meta

    if barcode_report.results:
        return _sn_barcode_failure_result(barcode_report, barcode_raw, meta, has_sn_path=True)

    if meta["ocr_text_found"]:
        return "", top_text or text, "ocr_no_match", meta
    return "", last_text, "none", meta


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
    sn, raw, source, meta, barcode_report = _recognize_sn_barcode(
        sn_path=sn_path,
        label_id=label_id,
        label_path=label_path,
        original_path=original_path,
    )
    if sn or not allow_ocr or not sn_path:
        return sn, raw, source, meta
    return recognize_sn_ocr_after_barcode(
        sn_path=sn_path,
        label_id=label_id,
        barcode_report=barcode_report,
        meta=meta,
    )


def label_key(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    for suffix in ("__model", "__sn", "__part_no", "__FAILED"):
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
            if item.get("part_no_path"):
                if not os.path.isfile(item["part_no_path"]):
                    raise FileNotFoundError(f"Manifest part_no_path is missing at {path}:{line_no}: {item['part_no_path']}")
                record["part_no_path"] = item["part_no_path"]
            raw_part_no_values = []
            if item.get("part_no"):
                raw_part_no_values.append(item.get("part_no"))
            raw_codes = item.get("part_no_codes") or []
            if isinstance(raw_codes, str):
                raw_part_no_values.append(raw_codes)
            elif isinstance(raw_codes, list):
                raw_part_no_values.extend(raw_codes)
            part_no_codes = []
            seen_part_nos = set()
            for raw_value in raw_part_no_values:
                for part_no in extract_part_numbers_from_text(str(raw_value or "")):
                    if part_no in seen_part_nos:
                        continue
                    seen_part_nos.add(part_no)
                    part_no_codes.append(part_no)
            if part_no_codes:
                record["part_no"] = part_no_codes[0]
                record["part_no_codes"] = part_no_codes
            original_path = item.get("original_image_path") or item.get("image_path")
            if original_path:
                record["original_image_path"] = original_path
    return records


def delayed_model_crop_enabled() -> bool:
    return _env_flag_default("SCAN2_DELAYED_MODEL_CROP", True)


def _safe_label_id_for_filename(label_id: str) -> str:
    value = str(label_id or "").strip()
    if not value:
        return ""
    if "/" in value or "\\" in value or ":" in value:
        return ""
    if value in {".", ".."} or ".." in value:
        return ""
    if os.path.basename(value) != value:
        return ""
    return value


def _path_within_dir(path: str, directory: str) -> bool:
    try:
        directory_abs = os.path.abspath(directory)
        path_abs = os.path.abspath(path)
        return os.path.commonpath([directory_abs, path_abs]) == directory_abs
    except ValueError:
        return False


def delayed_model_crop_from_label(item: dict, label_id: str) -> str:
    if not delayed_model_crop_enabled():
        return ""
    if item.get("model_path"):
        return item["model_path"]
    label_path = item.get("label_crop", "")
    if not label_path or not os.path.isfile(label_path):
        return ""

    safe_label_id = _safe_label_id_for_filename(label_id)
    if not safe_label_id:
        append_debug(f"[MODEL][DELAYED_CROP][REJECT] unsafe label_id={label_id!r}")
        return ""

    out_path = os.path.join(MODEL_CROP_DIR, f"{safe_label_id}__model.png")
    if not _path_within_dir(out_path, MODEL_CROP_DIR):
        append_debug(f"[MODEL][DELAYED_CROP][REJECT] escaped output path for label_id={label_id!r}")
        return ""
    if os.path.isfile(out_path):
        item["model_path"] = out_path
        return out_path

    try:
        import crop as crop_module

        os.makedirs(MODEL_CROP_DIR, exist_ok=True)
        result = crop_module.stage2_crop_model_from_label(label_path, out_path=out_path)
    except Exception as exc:
        append_debug(f"[MODEL][DELAYED_CROP][ERROR] {label_id}: {exc.__class__.__name__}: {exc}")
        return ""

    if isinstance(result, dict) and result.get("model_path") and os.path.isfile(result["model_path"]):
        item["model_path"] = result["model_path"]
        item["model_conf"] = result.get("model_conf")
        item["model_crop_source"] = result.get("model_crop_source", "delayed")
        append_debug(f"[MODEL][DELAYED_CROP] {label_id} -> {os.path.basename(result['model_path'])}")
        return result["model_path"]
    return ""


def maybe_save_learned_part_no_model(
    part_no_by_key: dict,
    part_no_map_updates: set,
    label_id: str,
    model_code: str,
    source: str,
) -> bool:
    part_no = part_no_by_key.get(label_id, "")
    if not part_no or not model_code:
        return False
    if label_id in part_no_map_updates:
        return False
    if save_part_no_model_mapping(part_no, model_code, label_id=label_id, source=source):
        part_no_map_updates.add(label_id)
        append_debug(f"[PART_NO_MAP][LEARNED] {label_id} {part_no}->{normalize_model(model_code)} source={source}")
        return True
    return False


def _extract_model_candidate_from_barcode_raw(raw: str) -> str:
    for part in str(raw or "").split(";"):
        candidate = part.split(":", 1)[-1].strip() if ":" in part else part.strip()
        model = extract_unknown_model_candidate(candidate)
        if model:
            return model
    return extract_unknown_model_candidate(raw)


def _accept_model_barcode_ocr_consensus(
    key: str,
    barcode_raw: str,
    ocr_raw: str,
    part_no_by_key: dict,
    part_no_map_updates: set,
    stats: dict,
) -> tuple[str, str, str]:
    barcode_model = _extract_model_candidate_from_barcode_raw(barcode_raw)
    ocr_model = extract_unknown_model_candidate(ocr_raw)
    if not barcode_model or not ocr_model or normalize_model(barcode_model) != normalize_model(ocr_model):
        return "", "", ""

    learned_model = normalize_model(barcode_model)
    learned = save_learned_model_code(learned_model, label_id=key, source="barcode_ocr_consensus")
    if learned:
        stats["model_consensus_learned"] = int(stats.get("model_consensus_learned", 0)) + 1
    if maybe_save_learned_part_no_model(
        part_no_by_key,
        part_no_map_updates,
        key,
        learned_model,
        "barcode_ocr_consensus",
    ):
        stats["model_part_no_learned"] += 1
    return learned_model, f"[BARCODE_OCR_CONSENSUS] {learned_model}", "barcode_ocr_consensus"


def assign_part_no_model_result(
    model_results: dict,
    part_no_key_groups: dict,
    part_no: str,
    model_code: str,
    raw: str,
    source: str,
):
    if not part_no or not model_code:
        return
    result = (normalize_model(model_code), raw, source)
    for key in part_no_key_groups.get(part_no, []):
        if key not in model_results:
            model_results[key] = result


# ===================== MAIN =====================
def main(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=None, log_level="info"):
    set_log_level(log_level)
    mask_raw = raw_result_fields_are_masked()
    model_barcode = _env_flag_default("SCAN2_MODEL_BARCODE", True)
    ocr_fallback = scan_ocr_fallback_enabled()
    part_no_ocr_fallback = part_no_ocr_fallback_enabled()
    model_ocr_allowed = ocr_fallback or not model_barcode
    scan_label_with_sn = scan_label_with_sn_enabled()
    scan_label_without_sn = scan_label_without_sn_enabled()
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
    part_no_files = []
    stats = {
        "records": 0,
        "model_total": 0,
        "model_success": 0,
        "model_barcode_hits": 0,
        "model_part_no_hits": 0,
        "model_part_no_learned": 0,
        "model_consensus_learned": 0,
        "model_deferred_crops": 0,
        "model_barcode_hit_rate": 0.0,
        "model_ocr_recoveries": 0,
        "part_no_total": 0,
        "part_no_decoded": 0,
        "sn_total": 0,
        "sn_attempted": 0,
        "sn_success": 0,
        "sn_barcode_attempts": 0,
        "sn_barcode_hits": 0,
        "sn_barcode_hit_rate": 0.0,
        "sn_ocr_recoveries": 0,
        "sn_barcode_parse_failures": 0,
        "sn_barcode_decoder_misses": 0,
        "sn_barcode_ambiguous": 0,
        "sn_barcode_quality_rejects": 0,
        "sn_problem": 0,
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

    if os.path.isdir(PART_NO_CROP_DIR):
        for fname in os.listdir(PART_NO_CROP_DIR):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTS:
                continue
            part_no_files.append(fname)
            key = label_key(fname)
            records.setdefault(key, {})["part_no_path"] = os.path.join(PART_NO_CROP_DIR, fname)

    if os.path.isdir(SN_CROP_DIR):
        for fname in os.listdir(SN_CROP_DIR):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTS:
                continue
            sn_files.append(fname)
            key = label_key(fname)
            records.setdefault(key, {})["sn_path"] = os.path.join(SN_CROP_DIR, fname)

    append_debug(
        f"[SCAN2] model_files={len(model_files)} part_no_files={len(part_no_files)} "
        f"sn_files={len(sn_files)} keys={len(records)}"
    )
    append_debug(f"[SCAN2] model_names={sorted(model_files)}")
    append_debug(f"[SCAN2] part_no_names={sorted(part_no_files)}")
    append_debug(f"[SCAN2] sn_names={sorted(sn_files)}")

    keys = sorted(records.keys())
    barcode_workers = scan_worker_count("barcode")
    ocr_workers = scan_worker_count("ocr")
    if barcode_workers > 1 or ocr_workers > 1:
        _log(
            f"识别并发：barcode_workers={barcode_workers}, ocr_workers={ocr_workers}",
            "info",
        )

    first_barcode_jobs = []
    model_barcode_jobs = []
    ocr_jobs = []
    model_results = {}
    model_barcode_misses = {}
    part_no_by_key = {}
    part_no_src_by_key = {}
    part_no_map_updates = set()
    part_no_scan_miss_keys = set()
    sn_results = {}
    sn_reports = {}
    part_no_first = scan_part_no_first_enabled()
    progress_logs = scan_progress_log_enabled()

    for key in keys:
        item = records[key]
        manifest_part_no = item.get("part_no", "")
        if part_no_first and manifest_part_no:
            part_no_by_key[key] = manifest_part_no
            part_no_src_by_key[key] = "part_no_barcode"
            model_code, part_no = model_from_part_no_text(manifest_part_no)
            if model_code:
                model_results[key] = (
                    model_code,
                    f"[PART_NO_BARCODE] {part_no}",
                    "part_no_barcode",
                )
        if part_no_first and item.get("part_no_path") and key not in part_no_by_key:
            first_barcode_jobs.append(("part_no_model", key, item))
        if item.get("sn_path") or (scan_label_without_sn and item.get("label_crop")):
            first_barcode_jobs.append(("sn", key, item))

    def run_barcode_job(job):
        kind, key, item = job
        job_label = _scan_job_label(kind, key)
        if kind == "part_no_model":
            result = (
                kind,
                key,
                try_model_from_part_no_crop(
                    item.get("part_no_path", ""),
                    label_id=key,
                    use_ocr=False,
                ),
            )
            if progress_logs:
                _log(f"[条码完成] {job_label} -> {_barcode_job_result_summary(kind, result[2])}", "info")
            return result
        if kind == "model":
            result = (
                kind,
                key,
                recognize_model_barcode(item["model_path"], label_id=key),
            )
            if progress_logs:
                _log(f"[条码完成] {job_label} -> {_barcode_job_result_summary(kind, result[2])}", "info")
            return result
        if item.get("original_image_path"):
            append_debug(f"[SN] {key} original image barcode fallback disabled")
        label_path = ""
        if item.get("label_crop"):
            if item.get("sn_path"):
                if scan_label_with_sn:
                    label_path = item.get("label_crop", "")
            elif scan_label_without_sn:
                label_path = item.get("label_crop", "")
        result = (
            kind,
            key,
            _recognize_sn_barcode(
                item.get("sn_path", ""),
                label_id=key,
                label_path=label_path,
                original_path="",
            ),
        )
        if progress_logs:
            _log(f"[条码完成] {job_label} -> {_barcode_job_result_summary(kind, result[2])}", "info")
        return result

    for kind, key, result in _map_ordered(
        first_barcode_jobs,
        run_barcode_job,
        barcode_workers,
        progress=None,
    ):
        item = records[key]
        if kind == "part_no_model":
            model_code, model_raw, model_src = result
            part_no = first_part_no_from_chunks(model_raw)
            if part_no:
                part_no_by_key[key] = part_no
                part_no_src_by_key[key] = model_src
            if model_code:
                model_results[key] = (model_code, model_raw, model_src)
            else:
                model_barcode_misses[key] = (model_raw, model_src)
                if not part_no:
                    part_no_scan_miss_keys.add(key)
            continue

        if kind == "model":
            model_code, model_raw, model_src = result
            if model_code:
                model_results[key] = (model_code, model_raw, model_src)
            else:
                model_barcode_misses[key] = (model_raw, model_src)
                ocr_jobs.append(("model", key, item, None))
            continue

        sn_code, sn_raw, sn_src, sn_meta, barcode_report = result
        sn_reports[key] = barcode_report
        sn_results[key] = (sn_code, sn_raw, sn_src, sn_meta)
        if ocr_fallback and not sn_code and (item.get("sn_path") or item.get("label_crop")):
            ocr_jobs.append(("sn", key, item, barcode_report))

    if part_no_ocr_fallback:
        for key in keys:
            if key in model_results or key in part_no_by_key:
                continue
            item = records[key]
            if not item.get("part_no_path"):
                continue
            miss_raw, miss_src = model_barcode_misses.get(key, ("", ""))
            if miss_raw or (miss_src and miss_src != "part_no_no_barcode"):
                continue
            model_code, model_raw, model_src = try_model_from_part_no_crop(
                item.get("part_no_path", ""),
                label_id=key,
                use_ocr=True,
            )
            part_no = first_part_no_from_chunks(model_raw)
            if part_no:
                part_no_by_key[key] = part_no
                part_no_src_by_key[key] = model_src
            if model_code:
                model_results[key] = (model_code, model_raw, model_src)
            else:
                model_barcode_misses[key] = (model_raw, model_src)
                if not part_no:
                    part_no_scan_miss_keys.add(key)

    part_no_key_groups = {}
    for key, part_no in part_no_by_key.items():
        if key not in model_results and part_no:
            part_no_key_groups.setdefault(part_no, []).append(key)

    for part_no, group_keys in part_no_key_groups.items():
        cached_model = get_part_no_model_map().get(part_no, "")
        if cached_model:
            assign_part_no_model_result(
                model_results,
                part_no_key_groups,
                part_no,
                cached_model,
                f"[PART_NO_CACHE] {part_no}",
                "part_no_learned",
            )
            continue

        if not delayed_model_crop_enabled():
            continue

        for learn_key in group_keys:
            item = records[learn_key]
            had_model_path = bool(item.get("model_path"))
            if not item.get("model_path"):
                if delayed_model_crop_from_label(item, learn_key) and not had_model_path:
                    stats["model_deferred_crops"] += 1
            if not item.get("model_path"):
                continue

            learned_model, learned_raw, learned_src = "", "", "missing"
            if model_barcode:
                learned_model, learned_raw, learned_src = recognize_model_barcode(
                    item["model_path"],
                    label_id=learn_key,
                )
            if not learned_model and model_ocr_allowed:
                learned_model, learned_raw, learned_src = recognize_model_ocr(
                    item["model_path"],
                    label_id=learn_key,
                    verify_barcode_visual=model_barcode,
                )
            if not learned_model or not model_code_is_plausible(learned_model):
                if learned_raw or learned_src:
                    model_barcode_misses[learn_key] = (learned_raw, learned_src)
                continue

            learned_model = normalize_model(learned_model)
            if maybe_save_learned_part_no_model(
                part_no_by_key,
                part_no_map_updates,
                learn_key,
                learned_model,
                learned_src,
            ):
                stats["model_part_no_learned"] += 1
            assign_part_no_model_result(
                model_results,
                part_no_key_groups,
                part_no,
                learned_model,
                f"[PART_NO_LEARNED] {part_no}",
                "part_no_learned",
            )
            break

    for key in keys:
        if key in model_results:
            continue
        item = records[key]
        if (
            not item.get("model_path")
            and (
                part_no_by_key.get(key)
                or key in part_no_scan_miss_keys
                or (part_no_first and not item.get("part_no_path"))
            )
            and delayed_model_crop_enabled()
        ):
            had_model_path = bool(item.get("model_path"))
            if delayed_model_crop_from_label(item, key) and not had_model_path:
                stats["model_deferred_crops"] += 1
        if item.get("model_path"):
            if model_barcode:
                model_barcode_jobs.append(("model", key, item))
            else:
                ocr_jobs.append(("model", key, item, None))
        elif item.get("part_no_path") and (ocr_fallback or part_no_ocr_fallback):
            ocr_jobs.append(("model", key, item, None))

    for kind, key, result in _map_ordered(
        model_barcode_jobs,
        run_barcode_job,
        barcode_workers,
        progress=None,
    ):
        item = records[key]
        if kind != "model":
            continue
        model_code, model_raw, model_src = result
        if model_code:
            model_results[key] = (model_code, model_raw, model_src)
            if maybe_save_learned_part_no_model(
                part_no_by_key,
                part_no_map_updates,
                key,
                model_code,
                model_src,
            ):
                stats["model_part_no_learned"] += 1
        else:
            model_barcode_misses[key] = (model_raw, model_src)
            ocr_jobs.append(("model", key, item, None))

    if ocr_jobs:
        pending_ocr_jobs = []
        for job in ocr_jobs:
            kind, key, item, barcode_report = job
            if kind == "model":
                sn_code = sn_results.get(key, ("", "", "", {}))[0]
                hinted_model = model_from_sn_hint(sn_code, "")
                if hinted_model:
                    model_results[key] = (hinted_model, f"[SN_HINT] {sn_code}", "sn_hint")
                    continue
                model_miss_raw = model_barcode_misses.get(key, ("", ""))[0]
                sn_raw = sn_results.get(key, ("", "", "", {}))[1]
                report_raw = _format_barcode_report(sn_reports[key]) if key in sn_reports else ""
                part_model, part_no = model_from_part_no_hint([model_miss_raw, sn_raw, report_raw])
                if part_model:
                    model_results[key] = (
                        part_model,
                        f"[PART_NO_HINT] {part_no}",
                        "part_no_hint",
                    )
                    continue
                part_model, part_raw, part_source = try_model_from_part_no_crop(
                    item.get("part_no_path", ""),
                    label_id=key,
                    use_ocr=part_no_ocr_fallback,
                )
                if part_model:
                    model_results[key] = (part_model, part_raw, part_source)
                    continue
                if (
                    not model_ocr_allowed
                    or (not item.get("model_path") and not item.get("label_crop"))
                ):
                    continue
            pending_ocr_jobs.append(job)
        ocr_jobs = pending_ocr_jobs

    def run_ocr_job(job):
        kind, key, item, barcode_report = job
        job_label = _scan_job_label(kind, key)
        if kind == "model":
            model_result = ("", "", "missing")
            if item.get("model_path"):
                model_result = recognize_model_ocr(
                    item["model_path"],
                    label_id=key,
                    verify_barcode_visual=model_barcode,
                )
            if (
                (not model_result[0] or not model_code_is_plausible(model_result[0]))
                and item.get("label_crop")
            ):
                label_result = recognize_model_label_ocr(
                    item["label_crop"],
                    label_id=key,
                )
                if label_result[0] and model_code_is_plausible(label_result[0]):
                    model_result = label_result
                elif not model_result[0] and label_result[1]:
                    model_result = label_result
            result = (kind, key, model_result)
            if progress_logs:
                _log(f"[OCR完成] {job_label} -> {_ocr_job_result_summary(kind, result[2])}", "info")
            return result
        current = sn_results.get(key, ("", "", "missing", {}))
        sn_meta = current[3] if len(current) > 3 else {}
        sn_ocr_path = item.get("sn_path") or item.get("label_crop", "")
        result = (
            kind,
            key,
            recognize_sn_ocr_after_barcode(
                sn_ocr_path,
                label_id=key,
                barcode_report=barcode_report or sn_reports.get(key),
                meta=sn_meta,
            ),
        )
        if progress_logs:
            _log(f"[OCR完成] {job_label} -> {_ocr_job_result_summary(kind, result[2])}", "info")
        return result

    for kind, key, result in _map_ocr_ordered(
        ocr_jobs,
        run_ocr_job,
        ocr_workers,
        progress=None,
    ):
        if kind == "model":
            model_code, model_raw, model_src = result
            if not model_code:
                consensus_model, consensus_raw, consensus_src = _accept_model_barcode_ocr_consensus(
                    key,
                    model_barcode_misses.get(key, ("", ""))[0],
                    model_raw,
                    part_no_by_key,
                    part_no_map_updates,
                    stats,
                )
                if consensus_model:
                    result = (consensus_model, consensus_raw, consensus_src)
                    model_code, model_raw, model_src = result
            if (
                not model_code_is_plausible(model_code)
                and records.get(key, {}).get("part_no_path")
                and part_no_ocr_fallback
            ):
                part_model, part_raw, part_source = try_model_from_part_no_crop(
                    records[key].get("part_no_path", ""),
                    label_id=key,
                    use_ocr=True,
                )
                if part_model:
                    result = (part_model, part_raw, part_source)
                    model_code, model_raw, model_src = result
            model_results[key] = result
            if model_code and maybe_save_learned_part_no_model(
                part_no_by_key,
                part_no_map_updates,
                key,
                model_code,
                model_src,
            ):
                stats["model_part_no_learned"] += 1
        else:
            sn_results[key] = result

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for key in keys:
            item = records[key]

            model_code, model_raw, model_src = model_results.get(key, ("", "", "missing"))
            sn_code, sn_raw, sn_src, sn_meta = sn_results.get(
                key,
                (
                    "",
                    "",
                    "missing",
                    {
                        "barcode_found": False,
                        "ocr_text_found": False,
                        "barcode_status": "not_attempted",
                        "barcode_attempts": 0,
                        "barcode_decoded_count": 0,
                    },
                ),
            )

            part_no = (
                part_no_by_key.get(key)
                or item.get("part_no", "")
                or first_part_no_from_chunks(model_raw, sn_raw)
            )
            part_no_src = part_no_src_by_key.get(key, "")
            if item.get("part_no_path"):
                stats["part_no_total"] += 1
                if part_no:
                    stats["part_no_decoded"] += 1

            sn_input_available = bool(item.get("sn_path") or item.get("label_crop"))
            sn_problem = bool(sn_input_available and not sn_code)
            sn_problem_reason = ""
            if sn_input_available:
                stats["sn_total"] += 1
                barcode_status = sn_meta.get("barcode_status", "not_attempted")
                stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
                if sn_src == "barcode":
                    stats["sn_barcode_hits"] += 1
                elif sn_code and sn_src.startswith("ocr") and barcode_status in {
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
                    stats["sn_problem"] += 1
                    sn_problem_reason = sn_src or barcode_status or "missing"
                    if sn_meta.get("barcode_found") or sn_meta.get("ocr_text_found"):
                        stats["regex_fail"] += 1
                    if not sn_meta.get("barcode_found"):
                        stats["barcode_fail"] += 1
                    if not sn_meta.get("ocr_text_found"):
                        stats["ocr_fail"] += 1

            hinted_model = model_from_sn_hint(sn_code, model_code)
            if hinted_model:
                model_code = hinted_model
                model_src = f"{model_src}+sn_hint" if model_src else "sn_hint"
            elif not model_code:
                part_model, hint_part_no = model_from_part_no_hint([model_raw, sn_raw])
                if part_model:
                    model_code = part_model
                    model_raw = f"[PART_NO_HINT] {hint_part_no}"
                    model_src = "part_no_hint"

            if model_code:
                normalized_model = normalize_model(model_code)
                if model_code_is_plausible(normalized_model):
                    model_code = normalized_model
                else:
                    append_sensitive_debug(
                        f"[MODEL][REJECT] {key} model={model_code!r} src={model_src!r} raw={model_raw!r}"
                    )
                    model_code = ""
                    model_src = f"invalid_{model_src}" if model_src else "invalid"

            if item.get("model_path") or item.get("part_no_path"):
                stats["model_total"] += 1
                if model_code:
                    stats["model_success"] += 1
                if model_src.startswith("barcode"):
                    stats["model_barcode_hits"] += 1
                elif model_src.startswith("part_no"):
                    stats["model_part_no_hits"] += 1
                elif model_src.startswith("ocr"):
                    stats["model_ocr_recoveries"] += 1

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
                "part_no": part_no,
                "part_no_src": part_no_src,
                "part_no_model_map_updated": key in part_no_map_updates,
                "sn_barcode_status": sn_meta.get("barcode_status", "not_attempted"),
                "sn_barcode_attempts": sn_meta.get("barcode_attempts", 0),
                "sn_barcode_decoded_count": sn_meta.get("barcode_decoded_count", 0),
                "sn_barcode_sources": sn_meta.get("barcode_sources", []),
                "sn_barcode_source_regions": sn_meta.get("barcode_source_regions", []),
                "sn_barcode_decoder_names": sn_meta.get("barcode_decoder_names", []),
                "sn_barcode_ambiguous_sns": sn_meta.get("barcode_ambiguous_sns", []),
                "sn_problem": sn_problem,
                "sn_problem_reason": sn_problem_reason,
            }

            _log(
                f"[{key}] "
                f"型号={model_code}（来源={display_source(model_src)}） | "
                f"SN={sn_code}（来源={display_source(sn_src)}）",
                "info",
            )

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["records"] += 1

    if stats["sn_total"]:
        stats["sn_barcode_hit_rate"] = stats["sn_barcode_hits"] / float(stats["sn_total"])
    if stats["model_total"]:
        stats["model_barcode_hit_rate"] = stats["model_barcode_hits"] / float(stats["model_total"])

    return stats


if __name__ == "__main__":
    main()
