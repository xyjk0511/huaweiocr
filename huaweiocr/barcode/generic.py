#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
涓撻棬閽堝浣犺繖绫烩€滃皬鏉″舰鐮?patch鈥濈殑澧炲己 + 瑙ｇ爜鑴氭湰

渚濊禆:
    pip install opencv-python numpy pyzbar
"""

import os
import subprocess
import tempfile
from typing import List, Dict, Tuple

import cv2
import numpy as np

from win_subprocess import hide_subprocess_windows

hide_subprocess_windows()

try:
    from app_paths import get_barcode_cli_path
except Exception:
    get_barcode_cli_path = None


# ===================== 閰嶇疆鍖?=====================

# 杩欎簺鍙傛暟鏄寜浣犻偅鍑犲紶 130x700 宸﹀彸鐨勫皬鍥捐皟鐨?
UPSCALE_TARGET_W = 2200   # 鏀惧ぇ鍚庣殑鐩爣瀹藉害
MAX_SCALE = 5.0           # 鏈€澶ф斁澶у€嶆暟

CROP_TOP_RATIO = 0.10     # 涓婇潰瑁佹帀涓€閮ㄥ垎鏂囧瓧鍖哄煙
CROP_BOTTOM_RATIO = 0.98  # 搴曢儴淇濈暀鍒?95% 楂樺害锛堢暀涓€鐐?margin锛?

ADAPTIVE_BLOCK_SIZE = 21  # 鑷€傚簲闃堝€奸偦鍩燂紙蹇呴』濂囨暟锛?
ADAPTIVE_C = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_DIRS = [
    r"stage2_fields\sn",
]

BARCODE_CLI_PATH = get_barcode_cli_path() if get_barcode_cli_path else ""
CLI_SCALE_FACTORS = [1.0, 1.5, 2.0, 2.5]
CLI_MAX_PIXELS = 50_000_000
CLI_MAX_CALLS_PER_PATCH = int(os.environ.get("BARCODE_CLI_MAX_CALLS_PER_PATCH", "4"))
CLI_TIMEOUT_SECONDS = float(os.environ.get("BARCODE_CLI_TIMEOUT_SECONDS", "2"))
_CLI_UNAVAILABLE = False
_CLI_UNAVAILABLE_REASON = ""
_PYZBAR_MODULE = None
_PYZBAR_IMPORT_ERROR = None

PAD_X = 30
PAD_Y = 16

CODE128_VISUAL_MIN_SCORE = float(os.environ.get("CODE128_VISUAL_MIN_SCORE", "0.75"))
CODE128_VISUAL_MIN_CORR = float(os.environ.get("CODE128_VISUAL_MIN_CORR", "0.53"))
CODE128_VISUAL_MIN_SEP = float(os.environ.get("CODE128_VISUAL_MIN_SEP", "0.20"))
CODE128_VISUAL_MIN_SYMBOL_CORR = float(os.environ.get("CODE128_VISUAL_MIN_SYMBOL_CORR", "0.0"))
CODE128_VISUAL_MIN_SYMBOL_SEP = float(os.environ.get("CODE128_VISUAL_MIN_SYMBOL_SEP", "0.0"))

CODE128_PATTERNS = (
    "212222", "222122", "222221", "121223", "121322", "131222", "122213", "122312",
    "132212", "221213", "221312", "231212", "112232", "122132", "122231", "113222",
    "123122", "123221", "223211", "221132", "221231", "213212", "223112", "312131",
    "311222", "321122", "321221", "312212", "322112", "322211", "212123", "212321",
    "232121", "111323", "131123", "131321", "112313", "132113", "132311", "211313",
    "231113", "231311", "112133", "112331", "132131", "113123", "113321", "133121",
    "313121", "211331", "231131", "213113", "213311", "213131", "311123", "311321",
    "331121", "312113", "312311", "332111", "314111", "221411", "431111", "111224",
    "111422", "121124", "121421", "141122", "141221", "112214", "112412", "122114",
    "122411", "142112", "142211", "241211", "221114", "413111", "241112", "134111",
    "111242", "121142", "121241", "114212", "124112", "124211", "411212", "421112",
    "421211", "212141", "214121", "412121", "111143", "111341", "131141", "114113",
    "114311", "411113", "411311", "113141", "114131", "311141", "411131", "211412",
    "211214", "211232", "2331112",
)

def _run_cli(cmd, **kwargs):
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs.setdefault("startupinfo", startupinfo)
        hidden_flags = subprocess.CREATE_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | hidden_flags
    return subprocess.run(cmd, **kwargs)

class _StderrSilencer:
    def __enter__(self):
        self._orig_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._orig_fd, 2)
        os.close(self._devnull)
        os.close(self._orig_fd)
        return False


def _short_error(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return f"{exc.__class__.__name__}:{text}"
    return exc.__class__.__name__


def _append_decoder_error(decoder_errors, message: str) -> None:
    if decoder_errors is None or not message:
        return
    if message not in decoder_errors:
        decoder_errors.append(message)


def _get_pyzbar(decoder_errors=None):
    global _PYZBAR_MODULE, _PYZBAR_IMPORT_ERROR
    if _PYZBAR_MODULE is not None:
        return _PYZBAR_MODULE
    if _PYZBAR_IMPORT_ERROR is not None:
        _append_decoder_error(
            decoder_errors,
            f"decoder_unavailable:pyzbar:{_short_error(_PYZBAR_IMPORT_ERROR)}",
        )
        return None

    try:
        from pyzbar import pyzbar as pyzbar_module
    except Exception as exc:
        _PYZBAR_IMPORT_ERROR = exc
        _append_decoder_error(
            decoder_errors,
            f"decoder_unavailable:pyzbar:{_short_error(exc)}",
        )
        return None

    _PYZBAR_MODULE = pyzbar_module
    return _PYZBAR_MODULE


def _one_line(value) -> str:
    return " ".join(str(value or "").split())


def _format_cli_process_error(proc) -> str:
    parts = [f"BarcodeReaderCLI_error:returncode={proc.returncode}"]
    stderr = _one_line(getattr(proc, "stderr", ""))
    stdout = _one_line(getattr(proc, "stdout", ""))
    if stderr:
        parts.append(f"stderr={stderr}")
    if stdout:
        parts.append(f"stdout={stdout}")
    return " ".join(parts)



# ===================== 鍩虹宸ュ叿 =====================

def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def auto_rotate_to_horizontal(gray: np.ndarray) -> np.ndarray:
    """
    鑷姩鍒ゆ柇鏄惁闇€瑕佹棆杞垚妯浘銆?
    瀵逛綘閭ｅ紶 525x99 杩欑, h >> w, 灏辨棆杞?90 搴︺€?
    """
    h, w = gray.shape[:2]
    if h > w * 1.8:
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    return gray


def crop_bar_band(gray: np.ndarray) -> np.ndarray:
    """
    鍙繚鐣欎腑闂粹€滄潯鐮佸甫鈥濓紝鎶婂ぇ閮ㄥ垎 S/N 鏂囨湰瑁佹帀銆?
    鎸変綘杩欎簺鍥撅紝鏉＄爜鍩烘湰鍦ㄤ笅鍗婇儴鍒嗭紝鎵€浠?
        淇濈暀 [0.25h, 0.95h]
    """
    h, w = gray.shape[:2]
    y1 = int(h * CROP_TOP_RATIO)
    y2 = int(h * CROP_BOTTOM_RATIO)
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    band = gray[y1:y2, :]
    return band


def crop_detected_barcode_band(gray: np.ndarray) -> np.ndarray | None:
    if gray is None or not hasattr(gray, "shape") or not hasattr(gray, "size") or gray.size == 0:
        return None
    h, w = gray.shape[:2]
    if h < 20 or w < 90:
        return None

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    row_dark = dark.mean(axis=1)
    row_trans = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1) / max(1, w - 1)
    candidate_rows = (row_dark > 0.08) & (row_dark < 0.75) & (row_trans > 0.07)

    min_run_h = max(16, int(h * 0.18))
    min_span_w = max(95, int(w * 0.30))
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
                active = col_dark > 0.30
                active_idx = np.where(active)[0]
                if active_idx.size:
                    span_w = int(active_idx[-1] - active_idx[0] + 1)
                    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
                    aspect = span_w / float(max(run_h, 1))
                    if span_w >= min_span_w and transitions >= 14 and aspect >= 2.8:
                        score = span_w * run_h * max(0.1, float(row_trans[start:end].mean()))
                        if score > best_score:
                            pad_x = max(20, int(span_w * 0.22))
                            pad_y = max(6, int(run_h * 0.28))
                            best = (
                                max(0, int(active_idx[0]) - pad_x),
                                max(0, start - pad_y),
                                min(w, int(active_idx[-1]) + 1 + pad_x),
                                min(h, end + pad_y),
                            )
                            best_score = score
            start = None

    if best is None:
        return None
    x1, y1, x2, y2 = best
    if x2 <= x1 or y2 <= y1:
        return None
    return gray[y1:y2, x1:x2]


def upscale(gray: np.ndarray,
            target_w: int = UPSCALE_TARGET_W,
            max_scale: float = MAX_SCALE) -> np.ndarray:
    """
    妯悜鏀惧ぇ鍒?target_w, 闃叉杩囧ぇ涓嶈秴杩?max_scale銆?
    """
    h, w = gray.shape[:2]
    if w >= target_w:
        return gray
    scale = min(target_w / float(w), max_scale)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def enhance_band(gray_band: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    瀵硅鍓嚭鐨勬潯鐮佸甫鍋氬寮?
      1) 鏀惧ぇ
      2) CLAHE 灞€閮ㄥ姣斿害澧炲己
      3) Gaussian + unsharp 閿愬寲
      4) 鑷€傚簲闃堝€?+ 闂繍绠?
    杩斿洖:
      enh_gray: 澧炲己鍚庣殑鐏板害
      bin_img : 浜屽€煎浘
    """
    # 1) 鏀惧ぇ
    g = upscale(gray_band, UPSCALE_TARGET_W, MAX_SCALE)

    # 2) CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    # 3) 杞诲害骞虫粦 + unsharp
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    sharp = cv2.addWeighted(g, 1.6, blur, -0.6, 0)

    # 4) 鑷€傚簲闃堝€?
    bin_img = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )

    # 5) 褰㈡€佸闂繍绠?(灏忔牳锛岃繛鎺ユ柇瑁傜粏鏉?
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return sharp, bin_img


def pad_quiet_zone(img: np.ndarray, pad_x: int = PAD_X, pad_y: int = PAD_Y) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        value = 255
    else:
        value = (255, 255, 255)
    return cv2.copyMakeBorder(
        img,
        pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=value,
    )


def _code128b_values(text: str) -> List[int] | None:
    values = [104]
    for ch in text:
        code = ord(ch)
        if code < 32 or code > 127:
            return None
        values.append(code - 32)

    checksum = values[0]
    for idx, value in enumerate(values[1:], 1):
        checksum += idx * value
    values.append(checksum % 103)
    values.append(106)
    return values


def _code128b_module_bits(text: str) -> np.ndarray | None:
    values = _code128b_values(text)
    if not values:
        return None

    bits = []
    for value in values:
        color = 1
        for width in map(int, CODE128_PATTERNS[value]):
            bits.extend([color] * width)
            color = 1 - color
    return np.asarray(bits, dtype=np.float32)


def _code128_value_bits(value: int) -> np.ndarray:
    bits = []
    color = 1
    for width in map(int, CODE128_PATTERNS[value]):
        bits.extend([color] * width)
        color = 1 - color
    return np.asarray(bits, dtype=np.float32)


def _code128b_symbol_quality(segment: np.ndarray, values: List[int]) -> Tuple[float, float]:
    total_modules = sum(sum(map(int, CODE128_PATTERNS[value])) for value in values)
    module_idx = ((np.arange(total_modules) + 0.5) * len(segment) / total_modules).astype(int)
    modules = segment[np.clip(module_idx, 0, len(segment) - 1)]

    min_corr = 1.0
    min_sep = 1.0
    offset = 0
    for value in values:
        ideal = _code128_value_bits(value)
        observed = modules[offset:offset + len(ideal)]
        offset += len(ideal)
        if observed.size != ideal.size or observed.std() < 1e-6:
            return 0.0, 0.0
        corr = float(np.corrcoef(observed, ideal)[0, 1])
        black = float(observed[ideal > 0.5].mean())
        white = float(observed[ideal < 0.5].mean())
        min_corr = min(min_corr, corr)
        min_sep = min(min_sep, black - white)
    return min_corr, min_sep


def _score_code128b_projection(gray: np.ndarray, text: str) -> Dict | None:
    bits = _code128b_module_bits(text)
    values = _code128b_values(text)
    if bits is None or values is None or gray is None or not hasattr(gray, "shape") or gray.size == 0:
        return None

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = auto_rotate_to_horizontal(gray)
    h, w = gray.shape[:2]
    if h < 20 or w < 90:
        return None

    row_ranges = []
    for top_ratio, bottom_ratio in ((0.25, 0.72), (0.30, 0.68), (0.20, 0.78), (0.0, 1.0)):
        y1 = int(h * top_ratio)
        y2 = int(h * bottom_ratio)
        if y2 - y1 >= 12:
            row_ranges.append((y1, y2))

    best = None
    for y1, y2 in row_ranges:
        roi = gray[y1:y2, :]
        smooth = cv2.GaussianBlur(roi.astype(np.float32), (3, 3), 0)
        dark = 1.0 - (smooth - smooth.min()) / max(1.0, smooth.max() - smooth.min())
        profile = dark.mean(axis=0)

        min_w = max(40, int(len(bits) * 1.05))
        max_w = min(w, int(len(bits) * 2.50))
        if max_w < min_w:
            continue

        for x1 in range(0, max(1, w - min_w + 1), 4):
            for width in range(min_w, max_w + 1, 4):
                x2 = x1 + width
                if x2 > w:
                    break
                segment = profile[x1:x2]
                if segment.std() < 1e-6:
                    continue

                sample_idx = ((np.arange(len(segment)) + 0.5) * len(bits) / len(segment)).astype(int)
                ideal = bits[np.clip(sample_idx, 0, len(bits) - 1)]
                corr = float(np.corrcoef(segment, ideal)[0, 1])
                black = float(segment[ideal > 0.5].mean())
                white = float(segment[ideal < 0.5].mean())
                separation = black - white
                score = corr + separation

                if best is None or score > best["score"]:
                    min_symbol_corr, min_symbol_sep = _code128b_symbol_quality(segment, values)
                    best = {
                        "score": score,
                        "corr": corr,
                        "separation": separation,
                        "min_symbol_corr": min_symbol_corr,
                        "min_symbol_sep": min_symbol_sep,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    }

    return best


def verify_code128b_text_in_image(gray_or_bgr: np.ndarray, text: str) -> Dict | None:
    if not text:
        return None

    result = _score_code128b_projection(gray_or_bgr, text)
    if not result:
        return None

    if (
        result["score"] >= CODE128_VISUAL_MIN_SCORE
        and result["corr"] >= CODE128_VISUAL_MIN_CORR
        and result["separation"] >= CODE128_VISUAL_MIN_SEP
        and result["min_symbol_corr"] >= CODE128_VISUAL_MIN_SYMBOL_CORR
        and result["min_symbol_sep"] >= CODE128_VISUAL_MIN_SYMBOL_SEP
    ):
        result["text"] = text
        return result
    return None


# ===================== 澶氳搴?+ 鍙嶈壊 瑙ｇ爜 =====================

def _rotate90(img: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return img
    return np.rot90(img, k).copy()


def decode_with_transforms(gray_or_bin: np.ndarray,
                           tag: str,
                           decoder_errors=None) -> List[Dict]:
    """
    瀵瑰崟閫氶亾鍥惧儚鍋?
      - 0/90/180/270掳 鏃嬭浆
      - 姝ｅ父 / 鍙嶈壊
    鍐嶇敤 pyzbar 瑙ｇ爜銆?
    """
    results: List[Dict] = []
    seen = set()
    pyzbar = _get_pyzbar(decoder_errors)
    if pyzbar is None:
        return results

    base = pad_quiet_zone(gray_or_bin)

    for k in range(4):
        rot = _rotate90(base, k)

        for inverted in [False, True]:
            if inverted:
                img = cv2.bitwise_not(rot)
            else:
                img = rot

            try:
                decoded = pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.CODE128])
            except Exception as exc:
                _append_decoder_error(decoder_errors, f"pyzbar_error:{_short_error(exc)}")
                decoded = []
            for d in decoded:
                try:
                    data_str = d.data.decode("utf-8", errors="ignore")
                except Exception:
                    data_str = repr(d.data)

                key = (data_str, d.type)
                if key in seen:
                    continue
                seen.add(key)

                x, y, w, h = d.rect
                results.append(
                    {
                        "type": d.type,
                        "data": data_str,
                        "rect": (int(x), int(y), int(w), int(h)),
                        "rotation_k90": k,
                        "inverted": inverted,
                        "source": tag,
                    }
                )

    return results


# ===================== BarcodeReaderCLI =====================

def read_barcodes_cli(img_path: str, decoder_errors=None) -> List[str]:
    global _CLI_UNAVAILABLE, _CLI_UNAVAILABLE_REASON
    if _CLI_UNAVAILABLE:
        _append_decoder_error(
            decoder_errors,
            _CLI_UNAVAILABLE_REASON or "decoder_unavailable:BarcodeReaderCLI:cached",
        )
        return []
    if not BARCODE_CLI_PATH or not os.path.exists(BARCODE_CLI_PATH):
        _CLI_UNAVAILABLE = True
        _CLI_UNAVAILABLE_REASON = (
            "decoder_unavailable:BarcodeReaderCLI:path_missing:"
            f"{BARCODE_CLI_PATH or '<empty>'}"
        )
        _append_decoder_error(decoder_errors, _CLI_UNAVAILABLE_REASON)
        return []

    cmd = [
        BARCODE_CLI_PATH,
        "-silent",
        "-type=code128,ucc128,code39,code93",
        "-max-bc=8",
        "-format=text",
        "-output-text={text}",
        img_path,
    ]

    try:
        proc = _run_cli(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=CLI_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        _CLI_UNAVAILABLE = True
        _CLI_UNAVAILABLE_REASON = f"BarcodeReaderCLI_file_not_found:{_short_error(exc)}"
        _append_decoder_error(decoder_errors, _CLI_UNAVAILABLE_REASON)
        return []
    except subprocess.TimeoutExpired as exc:
        timeout = getattr(exc, "timeout", CLI_TIMEOUT_SECONDS)
        _append_decoder_error(decoder_errors, f"BarcodeReaderCLI_timeout:{timeout}s")
        return []
    except Exception as exc:
        _append_decoder_error(decoder_errors, f"BarcodeReaderCLI_error:{_short_error(exc)}")
        return []

    if proc.returncode != 0:
        _append_decoder_error(decoder_errors, _format_cli_process_error(proc))
        return []

    # Strip a leading UTF-8 BOM (﻿ survives str.strip()) so the first SN
    # payload is not silently prefixed with an invisible character.
    return [
        stripped
        for ln in proc.stdout.splitlines()
        if (stripped := ln.strip().lstrip("﻿").strip())
    ]


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Write ``img`` to a possibly non-ASCII ``path``.

    cv2.imwrite is not Unicode-path safe on Windows: for a path under a CJK
    directory (e.g. a Chinese-username ``%TEMP%``) it returns True while writing
    nothing. Encode in memory and write the bytes ourselves so CJK temp dirs work.
    """
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return os.path.exists(path) and os.path.getsize(path) > 0


def decode_with_cli(img_bgr: np.ndarray, tag: str, decoder_errors=None) -> List[Dict]:
    img_bgr = pad_quiet_zone(img_bgr)
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not _imwrite_unicode(tmp_path, img_bgr):
            _append_decoder_error(decoder_errors, "BarcodeReaderCLI_error:temp_write_failed")
            lines = []
        else:
            lines = read_barcodes_cli(tmp_path, decoder_errors=decoder_errors)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    results: List[Dict] = []
    for ln in lines:
        results.append(
            {
                "type": "CLI",
                "data": ln,
                "rect": None,
                "rotation_k90": 0,
                "inverted": False,
                "source": tag,
            }
        )
    return results


def decode_cli_multi(
    img: np.ndarray,
    tag: str,
    budget: Dict | None = None,
    decoder_errors=None,
) -> List[Dict]:
    base = img
    rotations = [0, 1, 2, 3]
    if budget is None:
        budget = {"limit": CLI_MAX_CALLS_PER_PATCH, "calls": 0}

    results: List[Dict] = []
    seen = set()

    for k in rotations:
        rot = _rotate90(base, k)
        for scale in CLI_SCALE_FACTORS:
            if scale > MAX_SCALE:
                continue
            if scale <= 1.01:
                candidates = [rot]
            else:
                new_w = int(rot.shape[1] * scale)
                new_h = int(rot.shape[0] * scale)
                if new_w * new_h > CLI_MAX_PIXELS:
                    continue
                resized = cv2.resize(rot, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                candidates = [resized]

            for cand in candidates:
                if budget["calls"] >= budget["limit"]:
                    return results
                budget["calls"] += 1
                for r in decode_with_cli(cand, tag, decoder_errors=decoder_errors):
                    key = (r.get('data'), r.get('type'))
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(r)
                if results:
                    return results

    return results


def decode_cli_sharp_variants(
    gray_band: np.ndarray,
    tag: str,
    budget: Dict | None = None,
    decoder_errors=None,
) -> List[Dict]:
    if gray_band is None or gray_band.size == 0:
        return []
    h, w = gray_band.shape[:2]
    results: List[Dict] = []
    seen = set()
    for scale in (2.0, 3.0):
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w * new_h > CLI_MAX_PIXELS:
            continue
        big = cv2.resize(gray_band, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(big)
        blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
        sharp = cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)
        _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for name, candidate in (("sharp", sharp), ("otsu", otsu)):
            for r in decode_cli_multi(
                candidate,
                f"{tag}_{name}{scale:g}",
                budget,
                decoder_errors=decoder_errors,
            ):
                key = (r.get("data"), r.get("type"))
                if key in seen:
                    continue
                seen.add(key)
                results.append(r)
            if results:
                return results
    return results


# ===================== 涓绘祦绋? 鍗曞紶灏忔潯鐮佸浘 =====================

def decode_small_patch(img_bgr: np.ndarray) -> Dict:
    """
    ?????? patch ??????
      - ????
      - ????
      - ??
      - ??? + ????
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = auto_rotate_to_horizontal(gray)
    cli_budget = {"limit": CLI_MAX_CALLS_PER_PATCH, "calls": 0}
    decoder_errors: List[str] = []

    detected_band = crop_detected_barcode_band(gray)
    if detected_band is not None:
        results: List[Dict] = decode_with_transforms(
            detected_band,
            "detected_band_raw",
            decoder_errors=decoder_errors,
        )
        results += decode_cli_multi(
            detected_band,
            "detected_band_cli",
            cli_budget,
            decoder_errors=decoder_errors,
        )
        if results:
            return {"results": results, "decoder_errors": decoder_errors}

        enh_gray, bin_img = enhance_band(detected_band)
        results += decode_with_transforms(
            enh_gray,
            "detected_band_enh_gray",
            decoder_errors=decoder_errors,
        )
        results += decode_cli_multi(
            enh_gray,
            "detected_band_cli_enh_gray",
            cli_budget,
            decoder_errors=decoder_errors,
        )
        if results:
            return {"results": results, "decoder_errors": decoder_errors}

        results += decode_with_transforms(
            bin_img,
            "detected_band_enh_bin",
            decoder_errors=decoder_errors,
        )
        results += decode_cli_multi(
            bin_img,
            "detected_band_cli_enh_bin",
            cli_budget,
            decoder_errors=decoder_errors,
        )
        if results:
            return {"results": results, "decoder_errors": decoder_errors}

        results += decode_cli_sharp_variants(
            detected_band,
            "detected_band_cli",
            cli_budget,
            decoder_errors=decoder_errors,
        )
        if results:
            return {"results": results, "decoder_errors": decoder_errors}

    # ?????
    band = crop_bar_band(gray)

    # ?? band ?????????????????
    results: List[Dict] = decode_with_transforms(
        band,
        "band_raw",
        decoder_errors=decoder_errors,
    )

    # BarcodeReaderCLI multi-pass on band (if available)
    results += decode_cli_multi(
        band,
        "band_cli",
        cli_budget,
        decoder_errors=decoder_errors,
    )
    if results:
        return {"results": results, "decoder_errors": decoder_errors}

    # ????
    enh_gray, bin_img = enhance_band(band)

    # ?????????
    results += decode_with_transforms(
        enh_gray,
        "band_enh_gray",
        decoder_errors=decoder_errors,
    )
    results += decode_cli_multi(
        enh_gray,
        "band_cli_enh_gray",
        cli_budget,
        decoder_errors=decoder_errors,
    )
    if results:
        return {"results": results, "decoder_errors": decoder_errors}

    # ?????????
    results += decode_with_transforms(
        bin_img,
        "band_enh_bin",
        decoder_errors=decoder_errors,
    )
    results += decode_cli_multi(
        bin_img,
        "band_cli_enh_bin",
        cli_budget,
        decoder_errors=decoder_errors,
    )
    if results:
        return {"results": results, "decoder_errors": decoder_errors}

    results += decode_cli_multi(
        gray,
        "full_cli",
        cli_budget,
        decoder_errors=decoder_errors,
    )
    return {"results": results, "decoder_errors": decoder_errors}

def process_path(path: str) -> None:
    if os.path.isfile(path):
        paths = [path]
    elif os.path.isdir(path):
        paths = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if is_image_file(f)
        ]
        paths.sort()
    else:
        raise FileNotFoundError(path)

    for p in paths:
        print("=" * 80)
        print(f"[FILE] {p}")
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print("  鈿狅笍 鏃犳硶璇诲彇鍥剧墖")
            continue

        info = decode_small_patch(img)
        res = info["results"]

        if not res:
            print("  no barcode decoded")
            continue

        print(f"  decoded {len(res)} candidate barcodes")
        for i, r in enumerate(res, 1):
            print(
                f"    #{i}: type={r['type']}, data={r['data']!r}, "
                f"src={r['source']}, rot=90*{r['rotation_k90']}, "
                f"inverted={r['inverted']}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="灏忔潯褰㈢爜 patch 澧炲己璇嗗埆"
    )
    parser.add_argument(
        "path",
        nargs="*",
        default=None,
        help="image path or dir; default: stage2_fields\\sn and stage2_fields\\model",
    )
    args = parser.parse_args()

    paths = args.path or DEFAULT_DIRS
    for p in paths:
        process_path(p)
