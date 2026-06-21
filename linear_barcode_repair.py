#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
universal_linear_barcode_repair_v3.py

通用一维条形码像素级修复框架。

核心原则：
1. 不 OCR 条码旁边的人眼可读文字。
2. 不内置某批图片的 SN 表、不按文件名映射结果。
3. 先尝试通用扫码器 + 多尺度增强；失败后进入条纹像素级恢复。
4. 低清/压缩严重的 Code128 使用软采样 + checksum + 可配置字符集/长度/regex 约束。
5. 不把“业务约束”写死进代码；全部通过 CLI 参数传入。

适用范围：
- 主要：Code128 / GS1-128 / Code128-B 风格序列号、资产码、物流码。
- 直接增强解码：zbar 支持的 EAN-13、EAN-8、UPC-A、UPC-E、Code39、ITF 等一维码。
- 不适用：二维码、DataMatrix、PDF417、MaxiCode 等二维码；条纹被大面积遮挡或静区完全丢失的样本。

依赖：
    pip install opencv-python pillow numpy pyzbar
    # Linux 若 pyzbar 找不到 zbar:
    sudo apt-get install libzbar0

典型命令：
    # 尽量通用：大写字母数字，长度 6~32，自动扫描/修复
    python universal_linear_barcode_repair_v3.py --input ./imgs --out ./out --charset alnum --lengths 6-32 --clean

    # 对某类业务码提高确定性：传入正则，不写死在程序里
    python universal_linear_barcode_repair_v3.py --input ./imgs --out ./out --charset alnum --lengths 12,20 --regex "^(4E26[0-9]{8}|[0-9]{11}ES[0-9]{7})$" --clean

输出：
    out/direct_zbar_hits/                 直接增强扫码成功的记录
    out/code128_rebuilt_candidates/       从条纹恢复后重建的 Code128 候选
    out/patched_images_scannable/         替换回原图后的可扫图片
    out/diagnostics/                      候选、分数、profile 诊断
    out/barcode_repair_verification.csv   汇总
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from pyzbar.pyzbar import decode as zbar_decode
    from pyzbar.pyzbar import ZBarSymbol
except Exception:  # pragma: no cover
    zbar_decode = None
    ZBarSymbol = None

# -----------------------------------------------------------------------------
# Code128 table. 0..105 are ordinary codewords; 106 is STOP.
# -----------------------------------------------------------------------------
CODE128_PATTERNS: List[str] = [
    "212222", "222122", "222221", "121223", "121322", "131222", "122213", "122312", "132212", "221213",
    "221312", "231212", "112232", "122132", "122231", "113222", "123122", "123221", "223211", "221132",
    "221231", "213212", "223112", "312131", "311222", "321122", "321221", "312212", "322112", "322211",
    "212123", "212321", "232121", "111323", "131123", "131321", "112313", "132113", "132311", "211313",
    "231113", "231311", "112133", "112331", "132131", "113123", "113321", "133121", "313121", "211331",
    "231131", "213113", "213311", "213131", "311123", "311321", "331121", "312113", "312311", "332111",
    "314111", "221411", "431111", "111224", "111422", "121124", "121421", "141122", "141221", "112214",
    "112412", "122114", "122411", "142112", "142211", "241211", "221114", "413111", "241112", "134111",
    "111242", "121142", "121241", "114212", "124112", "124211", "411212", "421112", "421211", "212141",
    "214121", "412121", "111143", "111341", "131141", "114113", "114311", "411113", "411311", "113141",
    "114131", "311141", "411131", "211412", "211214", "211232", "2331112",
]
START_B = 104
STOP = 106


def pattern_to_bits(pattern: str) -> np.ndarray:
    bits: List[int] = []
    black = 1
    for ch in pattern:
        bits.extend([black] * int(ch))
        black = 1 - black
    return np.asarray(bits, dtype=np.float32)


BITS: List[np.ndarray] = [pattern_to_bits(p) for p in CODE128_PATTERNS]
BITS11 = np.stack(BITS[:106]).astype(np.float32)       # 106 x 11
STOP_BITS = BITS[STOP].astype(np.float32)              # 13
DIGIT_CODES = [ord(str(i)) - 32 for i in range(10)]
UPPER_CODES = [ord(chr(ord("A") + i)) - 32 for i in range(26)]
ALNUM_CODES = DIGIT_CODES + UPPER_CODES
PRINTABLE_CODES = list(range(96))


@dataclass(frozen=True)
class Profile:
    name: str
    y0: int
    y1: int
    x0: int
    x1: int
    row_y: int
    blackness: np.ndarray


@dataclass(frozen=True)
class GeometryLite:
    start_stop_score: float
    profile_name: str
    length: int
    x0: float
    module_px: float
    n_modules: int
    start_err: float
    stop_err: float


@dataclass
class DecodeCandidate:
    file: str
    text: str
    score: float
    avg_error: float
    total_error: float
    method: str
    length: int
    row_y: int
    x0: float
    module_px: float
    n_modules: int
    start_err: float
    checksum_err: float
    stop_err: float
    profile_name: str
    profile_y0: int
    profile_y1: int
    codewords: List[int]


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def read_color(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"图片编码失败: {path}")
    buf.tofile(str(path))


# -----------------------------------------------------------------------------
# Generic configurable constraints. These are not OCR results; they only define
# the permitted barcode payload class for ambiguity control.
# -----------------------------------------------------------------------------

import functools


def allowed_codes_for_position(length: int, mode: str) -> List[List[int]]:
    """Return allowed Code128-B character codewords by character class.

    This is deliberately generic. Product-specific formats belong in --regex,
    not inside this function.
    """
    if CURRENT_ALLOWED_OVERRIDE is not None and len(CURRENT_ALLOWED_OVERRIDE) == length:
        return CURRENT_ALLOWED_OVERRIDE
    if mode == "digits":
        return [DIGIT_CODES for _ in range(length)]
    if mode == "alnum":
        return [ALNUM_CODES for _ in range(length)]
    if mode == "upper":
        return [UPPER_CODES for _ in range(length)]
    if mode == "printable":
        return [PRINTABLE_CODES for _ in range(length)]
    raise ValueError(f"未知 charset/mode: {mode}")


def text_matches_mode(text: str, mode: str, regex: str = "") -> bool:
    if mode == "digits" and not re.fullmatch(r"[0-9]+", text):
        return False
    if mode == "upper" and not re.fullmatch(r"[A-Z]+", text):
        return False
    if mode == "alnum" and not re.fullmatch(r"[0-9A-Z]+", text):
        return False
    if mode == "printable" and not all(32 <= ord(ch) <= 126 for ch in text):
        return False
    if regex:
        return re.fullmatch(regex, text) is not None
    return True


def compile_regex_or_empty(pattern: str) -> str:
    pattern = pattern.strip()
    if not pattern:
        return ""
    re.compile(pattern)
    return pattern


# Optional per-position constraints. They are set by --template at runtime.
# This keeps the solver generic: the program knows only pattern syntax, not any vendor/SN rule.
CURRENT_ALLOWED_OVERRIDE: Optional[List[List[int]]] = None


def template_to_allowed(template: str) -> List[List[int]]:
    """Convert a generic template to Code128-B allowed codewords.

    Syntax:
      #  digit 0-9
      @  uppercase A-Z
      *  uppercase alphanumeric 0-9A-Z
      ?  printable ASCII 32-127
      other printable characters are fixed literals, e.g. 4E26########
    """
    allowed: List[List[int]] = []
    for ch in template:
        if ch == "#":
            allowed.append(DIGIT_CODES)
        elif ch == "@":
            allowed.append(UPPER_CODES)
        elif ch == "*":
            allowed.append(ALNUM_CODES)
        elif ch == "?":
            allowed.append(PRINTABLE_CODES)
        else:
            code = ord(ch) - 32
            if not (0 <= code <= 95):
                raise ValueError(f"模板字符 Code128-B 不支持: {ch!r}")
            allowed.append([code])
    return allowed


def parse_templates(spec: str) -> List[List[List[int]]]:
    spec = spec.strip()
    if not spec:
        return []
    out: List[List[List[int]]] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(template_to_allowed(part))
    return out


def text_matches_template(text: str, allowed: List[List[int]]) -> bool:
    if len(text) != len(allowed):
        return False
    for ch, codes in zip(text, allowed):
        if ord(ch) - 32 not in codes:
            return False
    return True

# -----------------------------------------------------------------------------
# Image -> candidate scanline profiles.
# -----------------------------------------------------------------------------

def normalize_blackness(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.float32)
    lo, hi = np.percentile(raw, [2, 98])
    if hi <= lo + 1e-3:
        lo, hi = float(raw.min()), float(raw.max() + 1e-3)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def profile_from_band(gray: np.ndarray, y0: int, y1: int, x0: int, x1: int, mode: str) -> Optional[np.ndarray]:
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return None
    if mode == "mean":
        white = crop.mean(axis=0)
    elif mode == "median":
        white = np.percentile(crop, 50, axis=0)
    elif mode.startswith("p"):
        white = np.percentile(crop, int(mode[1:]), axis=0)
    else:
        raise ValueError(mode)
    # white low means black bar. Convert to blackness in [0, 1].
    return normalize_blackness(255.0 - white)


def find_row_bands(gray: np.ndarray, max_bands: int) -> List[Tuple[int, int]]:
    """Find horizontal bands rich in vertical strokes. No OCR/text recognition is involved."""
    h, _ = gray.shape
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    row_score = np.mean(np.abs(sx), axis=1)
    k = max(5, (h // 80) * 2 + 1)
    smooth = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, k), 0).ravel()

    bands: List[Tuple[int, int, float]] = []
    min_sep = max(25, h // 12)
    for y in np.argsort(smooth)[::-1]:
        y = int(y)
        if any(abs(y - (a + b) // 2) < min_sep for a, b, _ in bands):
            continue
        threshold = max(float(np.percentile(smooth, 65)), float(smooth[y] * 0.35))
        a, b = y, y + 1
        while a > 0 and smooth[a - 1] >= threshold:
            a -= 1
        while b < h and smooth[b] >= threshold:
            b += 1

        min_h = max(20, h // 18)
        if b - a < min_h:
            pad = (min_h - (b - a)) // 2 + 1
            a, b = max(0, y - pad), min(h, y + pad)
        max_h = max(60, h // 3)
        if b - a > max_h:
            a, b = max(0, y - max_h // 2), min(h, y + max_h // 2)
        bands.append((a, b, float(smooth[y])))
        if len(bands) >= max_bands:
            break
    bands.sort(key=lambda t: -t[2])
    return [(a, b) for a, b, _ in bands]


def make_profiles(gray: np.ndarray, max_profiles: int = 6) -> List[Profile]:
    h, w = gray.shape
    max_bands = max(3, max_profiles // 2)
    bands = find_row_bands(gray, max_bands=max_bands)
    profiles: List[Profile] = []

    seen: set[Tuple[int, int, str]] = set()
    for bi, (y0, y1) in enumerate(bands):
        # A band-level profile is robust to tilted/blurred bars.
        for mode in ("p35", "mean"):
            key = (y0, y1, mode)
            if key not in seen:
                p = profile_from_band(gray, y0, y1, 0, w, mode)
                if p is not None:
                    profiles.append(Profile(f"band{bi}_{mode}", y0, y1, 0, w, (y0 + y1) // 2, p))
                seen.add(key)
            if len(profiles) >= max_profiles:
                return profiles

        # A thin row-window profile is sharper when the full band is smeared.
        sx = cv2.Sobel(gray[y0:y1, :], cv2.CV_32F, 1, 0, ksize=3)
        local_score = np.mean(np.abs(sx), axis=1)
        for yy in np.argsort(local_score)[::-1][:2]:
            cy = y0 + int(yy)
            half = max(8, (y1 - y0) // 6)
            a, b = max(0, cy - half), min(h, cy + half + 1)
            key = (a, b, "p35")
            if key in seen:
                continue
            p = profile_from_band(gray, a, b, 0, w, "p35")
            if p is not None:
                profiles.append(Profile(f"row{cy}_p35", a, b, 0, w, cy, p))
            seen.add(key)
            if len(profiles) >= max_profiles:
                return profiles
    return profiles[:max_profiles]


# -----------------------------------------------------------------------------
# 1D barcode geometry and soft Code128-B decode.
# -----------------------------------------------------------------------------

def extent_candidates(profile: np.ndarray, max_extents: int = 60) -> List[Tuple[float, float]]:
    """Candidate [x0, x1] ranges for first-to-last black bars."""
    p = profile.astype(np.float32)
    w = len(p)
    out: List[Tuple[float, float]] = []
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    thresholds.extend(float(np.percentile(p, q)) for q in (55, 60, 65, 70, 75, 80, 85))

    for t in sorted(set(round(float(x), 3) for x in thresholds)):
        mask = (p > t).astype(np.uint8)
        changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, w]
        vals = mask[starts]
        run_idx = np.where(vals == 1)[0]
        if len(run_idx) < 5:
            continue
        # Allow many skips because label borders/text/noise can appear before the true bars.
        max_skip = min(12, len(run_idx) - 1)
        for skip_left in range(max_skip + 1):
            for skip_right in range(max_skip + 1):
                if len(run_idx) <= skip_left + skip_right:
                    continue
                x0 = float(starts[run_idx[skip_left]])
                x1 = float(ends[run_idx[-1 - skip_right]])
                if x1 - x0 > max(120, 0.25 * w):
                    out.append((x0, x1))

    # Gradient-based fallback.
    grad = np.abs(np.gradient(p))
    for q in (70, 75, 80, 85, 90, 93):
        xs = np.where(grad > np.percentile(grad, q))[0]
        if len(xs) >= 10:
            x0 = float(np.percentile(xs, 1))
            x1 = float(np.percentile(xs, 99))
            if x1 - x0 > max(120, 0.25 * w):
                out.append((x0, x1))

    # Deduplicate, then keep a balanced set: wide ranges plus mid-width ranges.
    uniq: List[Tuple[float, float]] = []
    for x0, x1 in sorted(out, key=lambda e: e[1] - e[0], reverse=True):
        if all(abs(x0 - a) > 4 or abs(x1 - b) > 4 for a, b in uniq):
            uniq.append((x0, x1))
        if len(uniq) >= max_extents * 3:
            break

    # Sort by width but keep enough smaller candidates; true barcode may not be the widest dark region.
    return uniq[:max_extents]


def sample_modules(profile: np.ndarray, x0: float, module_px: float, n_modules: int) -> np.ndarray:
    xs = x0 + (np.arange(n_modules, dtype=np.float32) + 0.5) * module_px
    xs = np.clip(xs, 0, len(profile) - 1)
    return np.interp(xs, np.arange(len(profile), dtype=np.float32), profile).astype(np.float32)


def symbol_costs(seg: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    # seg is blackness: 1=black, 0=white. Pattern matrix uses same convention.
    return ((patterns - seg[None, :]) ** 2).mean(axis=1)


def start_stop_score(profile: np.ndarray, x0: float, module_px: float, n_modules: int) -> Tuple[float, float, float]:
    start_seg = sample_modules(profile, x0, module_px, 11)
    stop_x = x0 + (n_modules - 13) * module_px
    stop_seg = sample_modules(profile, stop_x, module_px, 13)
    start_err = float(((BITS11[START_B] - start_seg) ** 2).mean())
    stop_err = float(((STOP_BITS - stop_seg) ** 2).mean())
    return start_err + stop_err, start_err, stop_err


def quick_geometry_score(sampled: np.ndarray, length: int, mode: str) -> Tuple[float, float, float, float]:
    k = length + 2
    segs = sampled[: 11 * k].reshape(k, 11)
    stop_seg = sampled[11 * k:]
    start_err = float(symbol_costs(segs[0], BITS11[[START_B]])[0])
    stop_err = float(((STOP_BITS - stop_seg) ** 2).mean())
    allowed = allowed_codes_for_position(length, mode)
    best_data_err: List[float] = []
    for pos, codes in enumerate(allowed):
        d = symbol_costs(segs[1 + pos], BITS11[np.asarray(codes, dtype=np.int16)])
        best_data_err.append(float(np.min(d)))
    data_err = float(np.mean(best_data_err)) if best_data_err else 99.0
    return start_err + stop_err + data_err, start_err, stop_err, data_err


def decode_direct(sampled: np.ndarray, length: int, mode: str) -> Optional[Tuple[str, float, float, float, float, float, List[int]]]:
    k = length + 2
    segs = sampled[: 11 * k].reshape(k, 11)
    stop_seg = sampled[11 * k:]
    start_err = float(symbol_costs(segs[0], BITS11[[START_B]])[0])
    stop_err = float(((STOP_BITS - stop_seg) ** 2).mean())
    if start_err > 0.75 or stop_err > 0.85:
        return None

    allowed = allowed_codes_for_position(length, mode)
    data_codes: List[int] = []
    data_errs: List[float] = []
    for pos, codes in enumerate(allowed):
        arr = np.asarray(codes, dtype=np.int16)
        d = symbol_costs(segs[1 + pos], BITS11[arr])
        j = int(np.argmin(d))
        data_codes.append(int(arr[j]))
        data_errs.append(float(d[j]))

    checksum = (START_B + sum((i + 1) * code for i, code in enumerate(data_codes))) % 103
    checksum_err = float(symbol_costs(segs[-1], BITS11[[checksum]])[0])
    text = "".join(chr(c + 32) for c in data_codes)
    total = start_err + sum(data_errs) + checksum_err + stop_err
    avg = total / (length + 3)
    return text, float(total), float(avg), start_err, checksum_err, stop_err, [START_B] + data_codes + [checksum, STOP]


def decode_dp(sampled: np.ndarray, length: int, mode: str) -> Optional[Tuple[str, float, float, float, float, float, List[int]]]:
    """Exact dynamic programming over allowed characters, enforcing Code128 checksum."""
    k = length + 2
    segs = sampled[: 11 * k].reshape(k, 11)
    stop_seg = sampled[11 * k:]
    start_err = float(symbol_costs(segs[0], BITS11[[START_B]])[0])
    stop_err = float(((STOP_BITS - stop_seg) ** 2).mean())
    if start_err > 1.20 or stop_err > 1.20:
        return None

    allowed = allowed_codes_for_position(length, mode)
    per_pos: List[Tuple[np.ndarray, np.ndarray]] = []
    for pos, codes in enumerate(allowed):
        arr = np.asarray(codes, dtype=np.int16)
        costs = symbol_costs(segs[1 + pos], BITS11[arr]).astype(np.float32)
        per_pos.append((arr, costs))

    inf = np.float32(1e9)
    dp = np.full(103, inf, dtype=np.float32)
    dp[START_B % 103] = np.float32(start_err)
    prev_mod: List[np.ndarray] = []
    prev_code: List[np.ndarray] = []

    for pos, (codes, costs) in enumerate(per_pos, start=1):
        ndp = np.full(103, inf, dtype=np.float32)
        pm = np.full(103, -1, dtype=np.int16)
        pc = np.full(103, -1, dtype=np.int16)
        active_mods = np.where(dp < inf / 2)[0]
        for mod in active_mods:
            next_mods = (mod + pos * codes) % 103
            vals = dp[mod] + costs
            for code, next_mod, val in zip(codes, next_mods, vals):
                if val < ndp[next_mod]:
                    ndp[next_mod] = val
                    pm[next_mod] = mod
                    pc[next_mod] = code
        dp = ndp
        prev_mod.append(pm)
        prev_code.append(pc)

    checksum_costs = symbol_costs(segs[-1], BITS11[:103]).astype(np.float32)
    totals = dp + checksum_costs + np.float32(stop_err)
    checksum = int(np.argmin(totals))
    total = float(totals[checksum])
    checksum_err = float(checksum_costs[checksum])

    data_codes: List[int] = []
    cur = checksum
    for pos in range(length - 1, -1, -1):
        code = int(prev_code[pos][cur])
        mod = int(prev_mod[pos][cur])
        if code < 0 or mod < 0:
            return None
        data_codes.append(code)
        cur = mod
    data_codes.reverse()

    text = "".join(chr(c + 32) for c in data_codes)
    avg = total / (length + 3)
    return text, total, float(avg), start_err, checksum_err, stop_err, [START_B] + data_codes + [checksum, STOP]


def decode_for_lengths(
    path: Path,
    profiles: Sequence[Profile],
    lengths: Sequence[int],
    mode: str,
    regex: str = "",
    max_extents: int = 60,
    stage1_keep: int = 300,
    stage2_keep: int = 80,
    decode_keep: int = 60,
) -> List[DecodeCandidate]:
    phase_grid = (-1.4, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.4)
    scales = (0.975, 0.988, 1.000, 1.012, 1.025)
    profile_map = {p.name: p for p in profiles}

    stage1: List[GeometryLite] = []
    for profile in profiles:
        extents = extent_candidates(profile.blackness, max_extents=max_extents)
        width = len(profile.blackness)
        for x0e, x1e in extents:
            extent_width = x1e - x0e
            for length in lengths:
                n_modules = 11 * (length + 2) + 13
                base_module = extent_width / n_modules
                if not (1.0 <= base_module <= 16.0):
                    continue
                for dx_factor in phase_grid:
                    x0 = x0e + dx_factor * base_module
                    if x0 < 0:
                        continue
                    for scale in scales:
                        module_px = base_module * scale
                        if x0 + n_modules * module_px >= width - 1:
                            continue
                        ss, start_err, stop_err = start_stop_score(profile.blackness, x0, module_px, n_modules)
                        if start_err < 0.90 and stop_err < 0.90:
                            stage1.append(GeometryLite(ss, profile.name, length, x0, module_px, n_modules, start_err, stop_err))

    stage1.sort(key=lambda g: g.start_stop_score)
    deduped: List[GeometryLite] = []
    seen: set[Tuple[str, int, float, float]] = set()
    for g in stage1:
        key = (g.profile_name, g.length, round(g.x0, 1), round(g.module_px, 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(g)
        if len(deduped) >= stage1_keep:
            break

    stage2: List[Tuple[float, GeometryLite, np.ndarray]] = []
    for g in deduped:
        profile = profile_map[g.profile_name]
        sampled = sample_modules(profile.blackness, g.x0, g.module_px, g.n_modules)
        quick_score, start_err, stop_err, _ = quick_geometry_score(sampled, g.length, mode)
        if start_err < 0.80 and stop_err < 0.80:
            stage2.append((quick_score, g, sampled))
    stage2.sort(key=lambda t: t[0])
    stage2 = stage2[:stage2_keep]

    results: Dict[str, DecodeCandidate] = {}
    for quick_score, g, sampled in stage2[:decode_keep]:
        profile = profile_map[g.profile_name]
        decoded = decode_direct(sampled, g.length, mode)
        if decoded is None or decoded[2] > 0.20:
            dp_decoded = decode_dp(sampled, g.length, mode)
            if dp_decoded is not None:
                decoded = dp_decoded
        if decoded is None:
            continue
        text, total, avg, start_err, checksum_err, stop_err, codewords = decoded
        if not text_matches_mode(text, mode, regex):
            continue

        # Lower is better. avg_error is the main signal; quick_score is a mild geometry prior.
        score = float(avg + 0.02 * quick_score)
        cand = DecodeCandidate(
            file=path.name,
            text=text,
            score=score,
            avg_error=float(avg),
            total_error=float(total),
            method="soft_code128b_dp_checksum",
            length=g.length,
            row_y=profile.row_y,
            x0=float(g.x0),
            module_px=float(g.module_px),
            n_modules=g.n_modules,
            start_err=float(start_err),
            checksum_err=float(checksum_err),
            stop_err=float(stop_err),
            profile_name=profile.name,
            profile_y0=profile.y0,
            profile_y1=profile.y1,
            codewords=codewords,
        )
        if text not in results or cand.score < results[text].score:
            results[text] = cand

    return sorted(results.values(), key=lambda c: (c.score, c.avg_error, -c.length))


def decode_image(
    path: Path,
    mode: str,
    lengths_arg: str,
    regex: str = "",
    templates: Optional[List[List[List[int]]]] = None,
    accept_score: float = 0.20,
    max_profiles: int = 6,
) -> Tuple[List[DecodeCandidate], List[Profile], str]:
    gray = read_gray(path)
    profiles = make_profiles(gray, max_profiles=max_profiles)
    if not profiles:
        return [], [], "no_profiles"

    global CURRENT_ALLOWED_OVERRIDE
    templates = templates or []
    if templates:
        merged: Dict[str, DecodeCandidate] = {}
        old_override = CURRENT_ALLOWED_OVERRIDE
        try:
            for allowed in templates:
                CURRENT_ALLOWED_OVERRIDE = allowed
                partial = decode_for_lengths(
                    path, profiles, [len(allowed)], mode, regex="",
                    max_extents=60, stage1_keep=350, stage2_keep=110, decode_keep=90,
                )
                for cand in partial:
                    if not text_matches_template(cand.text, allowed):
                        continue
                    if regex and not text_matches_mode(cand.text, mode, regex):
                        continue
                    if cand.text not in merged or cand.score < merged[cand.text].score:
                        merged[cand.text] = cand
        finally:
            CURRENT_ALLOWED_OVERRIDE = old_override
        return sorted(merged.values(), key=lambda c: (c.score, c.avg_error, -c.length)), profiles, "template_constrained_code128b_pixel_rescue"

    lengths = parse_lengths(lengths_arg)
    results = decode_for_lengths(
        path,
        profiles,
        lengths,
        mode,
        regex=regex,
        max_extents=60,
        stage1_keep=350,
        stage2_keep=110,
        decode_keep=90,
    )
    return results, profiles, "generic_code128b_pixel_rescue"

# -----------------------------------------------------------------------------
# Rendering, patching, and zbar verification.
# -----------------------------------------------------------------------------

def code128b_values(text: str) -> List[int]:
    data = []
    for ch in text:
        code = ord(ch) - 32
        if not (0 <= code <= 95):
            raise ValueError(f"Code128-B 不支持字符: {ch!r}")
        data.append(code)
    checksum = (START_B + sum((i + 1) * code for i, code in enumerate(data))) % 103
    return [START_B] + data + [checksum, STOP]


def render_code128b(text: str, module_px: int = 4, height: int = 120, quiet_modules: int = 12) -> Image.Image:
    values = code128b_values(text)
    total_modules = 2 * quiet_modules + sum(sum(int(c) for c in CODE128_PATTERNS[v]) for v in values)
    img = Image.new("L", (total_modules * module_px, height), 255)
    draw = ImageDraw.Draw(img)
    x = quiet_modules * module_px
    for value in values:
        black = True
        for ch in CODE128_PATTERNS[value]:
            w = int(ch) * module_px
            if black:
                draw.rectangle([x, 0, x + w - 1, height - 1], fill=0)
            x += w
            black = not black
    return img


def render_code128b_to_fit(text: str, target_w: int, target_h: int, quiet_modules: int = 12) -> Image.Image:
    values = code128b_values(text)
    symbol_modules = sum(sum(int(c) for c in CODE128_PATTERNS[v]) for v in values)
    total_modules = symbol_modules + 2 * quiet_modules
    module_px = max(1, target_w // total_modules)
    height = max(55, target_h)
    barcode = render_code128b(text, module_px=module_px, height=height, quiet_modules=quiet_modules)
    canvas = Image.new("L", (target_w, target_h), 255)
    px = max(0, (target_w - barcode.width) // 2)
    py = max(0, (target_h - barcode.height) // 2)
    canvas.paste(barcode.crop((0, 0, min(barcode.width, target_w), min(barcode.height, target_h))), (px, py))
    return canvas


def verify_code128_with_zbar(image: Image.Image) -> List[str]:
    if zbar_decode is None:
        return []
    try:
        symbols = [ZBarSymbol.CODE128] if ZBarSymbol is not None else None
        decoded = zbar_decode(image, symbols=symbols)
    except Exception:
        return []
    values: List[str] = []
    for d in decoded:
        try:
            values.append(d.data.decode("utf-8"))
        except Exception:
            values.append(d.data.decode(errors="replace"))
    return values



@dataclass
class ZBarHit:
    data: str
    symbology: str
    quality: int
    rect: Tuple[int, int, int, int]
    variant: str


def pil_from_gray_or_color(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def zbar_decode_any_pil(image: Image.Image) -> List[ZBarHit]:
    if zbar_decode is None:
        return []
    hits: List[ZBarHit] = []
    try:
        decoded = zbar_decode(image)
    except Exception:
        return []
    for d in decoded:
        try:
            data = d.data.decode("utf-8")
        except Exception:
            data = d.data.decode(errors="replace")
        rect = getattr(d, "rect", None)
        if rect is not None:
            r = (int(rect.left), int(rect.top), int(rect.width), int(rect.height))
        else:
            r = (0, 0, image.width, image.height)
        quality = int(getattr(d, "quality", 0) or 0)
        sym = getattr(d, "type", "UNKNOWN")
        hits.append(ZBarHit(data=data, symbology=str(sym), quality=quality, rect=r, variant=""))
    return hits


def enhanced_variants_for_zbar(path: Path, max_side: int = 2400) -> List[Tuple[str, Image.Image]]:
    gray = read_gray(path)
    variants: List[Tuple[str, np.ndarray]] = [("gray", gray)]
    # Contrast-limited adaptive histogram equalization.
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants.append(("clahe", clahe))
    except Exception:
        pass
    # Sharpen and threshold variants.
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    variants.append(("sharp", sharp))
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))
    adap = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    variants.append(("adaptive", adap))

    out: List[Tuple[str, Image.Image]] = []
    for name, arr in variants:
        h, w = arr.shape[:2]
        for scale in (1.0, 1.5, 2.0, 3.0, 4.0):
            if max(h, w) * scale > max_side:
                continue
            if scale == 1.0:
                up = arr
            else:
                up = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            out.append((f"{name}_x{scale:g}", pil_from_gray_or_color(up)))
            # Some cameras store labels upside-down; 1D barcodes are often decodable after 180 rotation.
            out.append((f"{name}_x{scale:g}_rot180", pil_from_gray_or_color(cv2.rotate(up, cv2.ROTATE_180))))
    return out


def zbar_rescue_decode(path: Path) -> List[ZBarHit]:
    all_hits: Dict[Tuple[str, str], ZBarHit] = {}
    for variant_name, img in enhanced_variants_for_zbar(path):
        for hit in zbar_decode_any_pil(img):
            key = (hit.symbology, hit.data)
            hit.variant = variant_name
            if key not in all_hits or hit.quality > all_hits[key].quality:
                all_hits[key] = hit
    return sorted(all_hits.values(), key=lambda h: (-h.quality, h.symbology, h.data))

def estimate_patch_box(gray: np.ndarray, cand: DecodeCandidate, quiet_modules: int = 12) -> Tuple[int, int, int, int]:
    h, w = gray.shape
    x_first = cand.x0
    x_last = cand.x0 + cand.n_modules * cand.module_px
    x0 = int(max(0, np.floor(x_first - quiet_modules * cand.module_px - 5)))
    x1 = int(min(w, np.ceil(x_last + quiet_modules * cand.module_px + 5)))
    if x1 <= x0:
        x0, x1 = 0, w

    # Estimate vertical band near the decoded row. Dark density + vertical edges are both useful.
    roi = gray[:, x0:x1].astype(np.float32)
    if roi.size == 0:
        return 0, h, x0, x1
    inv = normalize_blackness(255.0 - roi.mean(axis=1))
    sx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    edge = normalize_blackness(np.mean(np.abs(sx), axis=1))
    row_score = 0.55 * inv + 0.45 * edge
    row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 11), 0).ravel()

    cy = int(np.clip(cand.row_y, 0, h - 1))
    local_level = float(row_score[cy])
    threshold = max(float(np.percentile(row_score, 65)), local_level * 0.35)
    y0, y1 = cy, cy + 1
    while y0 > 0 and row_score[y0 - 1] >= threshold:
        y0 -= 1
    while y1 < h and row_score[y1] >= threshold:
        y1 += 1

    # Merge with the actual profile band that produced the decode.
    y0 = min(y0, cand.profile_y0)
    y1 = max(y1, cand.profile_y1)

    # Ensure practical barcode height for scanning; avoid replacing unrelated text if possible.
    min_h = 70 if h < 500 else 100
    if y1 - y0 < min_h:
        center = (max(y0, 0) + min(y1, h)) // 2
        y0 = center - min_h // 2
        y1 = y0 + min_h
    y0 = int(max(0, y0 - 5))
    y1 = int(min(h, y1 + 5))
    return y0, y1, x0, x1


def patch_image(original_path: Path, cand: DecodeCandidate) -> Tuple[Image.Image, Tuple[int, int, int, int], List[str]]:
    gray = read_gray(original_path)
    box = estimate_patch_box(gray, cand)
    y0, y1, x0, x1 = box
    base = Image.open(original_path).convert("RGB")
    target_w = max(1, x1 - x0)
    target_h = max(1, y1 - y0)
    barcode = render_code128b_to_fit(cand.text, target_w, target_h, quiet_modules=12).convert("RGB")
    base.paste(barcode, (x0, y0))
    zbar_values = verify_code128_with_zbar(base)
    return base, box, zbar_values


# -----------------------------------------------------------------------------
# CLI and file I/O.
# -----------------------------------------------------------------------------

def parse_lengths(spec: str) -> List[int]:
    s = spec.strip().lower()
    if s == "auto":
        return [12, 20]
    lengths: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lengths.extend(range(int(a), int(b) + 1))
        else:
            lengths.append(int(part))
    return sorted(set(lengths))


def list_input_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(path.glob(pat))
    out: List[Path] = []
    seen: set[Path] = set()
    for f in sorted(files):
        if f in seen:
            continue
        seen.add(f)
        name = f.name.lower()
        if "patched" in name or "code128" in name or "contact_sheet" in name or "roi_only" in name:
            continue
        out.append(f)
    return out


def extract_zip_if_needed(input_path: Path, work_dir: Path) -> Path:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = work_dir / "_unzipped_input"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir
    return input_path


def make_contact_sheet(items: List[Tuple[str, str, Image.Image]], out_path: Path) -> None:
    if not items:
        return
    cell_w = max(max(img.width for _, _, img in items) + 20, 900)
    cell_h = 220
    sheet = Image.new("RGB", (cell_w, cell_h * len(items)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, text, img) in enumerate(items):
        y = i * cell_h
        draw.text((10, y + 10), f"{name} -> {text}", fill=(0, 0, 0))
        shown = img.convert("RGB")
        if shown.width > cell_w - 20:
            ratio = (cell_w - 20) / shown.width
            shown = shown.resize((int(shown.width * ratio), int(shown.height * ratio)), Image.Resampling.LANCZOS)
        sheet.paste(shown, (10, y + 40))
    sheet.save(out_path)


def run_batch(args: argparse.Namespace) -> Path:
    input_path = Path(args.input)
    out_dir = Path(args.out)
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_input = extract_zip_if_needed(input_path, out_dir)
    files = list_input_images(actual_input)
    if not files:
        raise SystemExit(f"没有找到输入图片: {actual_input}")

    standalone_dir = out_dir / "standalone_rebuilt_code128"
    patched_dir = out_dir / "patched_images_scannable"
    roi_dir = out_dir / "roi_debug_profiles"
    diag_dir = out_dir / "diagnostics"
    for d in (standalone_dir, patched_dir, roi_dir, diag_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    sheet_items: List[Tuple[str, str, Image.Image]] = []

    for image_path in files:
        t0 = time.time()
        print(f"processing {image_path.name} ...", flush=True)
        zbar_hits = zbar_rescue_decode(image_path) if args.try_zbar else []
        try:
            results, profiles, strategy = decode_image(image_path, mode=args.mode, lengths_arg=args.lengths, regex=compile_regex_or_empty(args.regex), templates=parse_templates(args.template), accept_score=args.accept_score, max_profiles=args.max_profiles)
        except Exception as exc:
            rows.append({"file": image_path.name, "status": "FAIL", "error": str(exc), "seconds": round(time.time() - t0, 3)})
            print(f"  FAIL: {exc}", flush=True)
            continue

        # Save profile diagnostics; these are 1D bar-blackness projections, not OCR text.
        with (diag_dir / f"{image_path.stem}__top_candidates.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in results[:20]], f, ensure_ascii=False, indent=2)

        if not results and zbar_hits:
            top_hit = zbar_hits[0]
            rows.append({
                "file": image_path.name,
                "status": "PASS_DIRECT_ZBAR",
                "decoded_from_bar_pixels": top_hit.data,
                "symbology": top_hit.symbology,
                "strategy": "zbar_multiscale_enhancement",
                "direct_zbar_hits": "|".join(f"{h.symbology}:{h.data}" for h in zbar_hits[:10]),
                "seconds": round(time.time() - t0, 3),
                "note": "通用扫码器在增强/放大版本上直接成功，未进入重建替换",
            })
            print(f"  PASS_DIRECT_ZBAR: {top_hit.symbology} {top_hit.data}", flush=True)
            continue

        if not results:
            rows.append({
                "file": image_path.name,
                "status": "FAIL",
                "decoded_from_bar_pixels": "",
                "strategy": strategy,
                "seconds": round(time.time() - t0, 3),
                "note": "没有找到满足 Code128 checksum 和格式约束的候选",
                "direct_zbar_hits": "|".join(f"{h.symbology}:{h.data}" for h in zbar_hits[:10]),
            })
            print("  FAIL: no checksum-valid candidate", flush=True)
            continue

        best = results[0]
        standalone = render_code128b(best.text, module_px=args.module_px, height=args.bar_height, quiet_modules=12)
        standalone_values = verify_code128_with_zbar(standalone)
        standalone_path = standalone_dir / f"{image_path.stem}__decoded_from_bars__{best.text}__CODE128B.png"
        standalone.save(standalone_path)
        sheet_items.append((image_path.name, best.text, standalone))

        patched_values: List[str] = []
        patch_box: Optional[Tuple[int, int, int, int]] = None
        patched_path: Optional[Path] = None
        if not args.no_patch:
            patched, patch_box, patched_values = patch_image(image_path, best)
            patched_path = patched_dir / f"{image_path.stem}__patched_scannable__{best.text}.png"
            patched.save(patched_path)

        margin = float(results[1].score - best.score) if len(results) > 1 else 999.0
        status = "PASS" if best.text in standalone_values and (args.no_patch or best.text in patched_values) else "DECODED"
        if margin < args.min_margin:
            status = "AMBIGUOUS"
        rows.append({
            "file": image_path.name,
            "status": status,
            "decoded_from_bar_pixels": best.text,
            "strategy": strategy,
            "mode": args.mode,
            "length": best.length,
            "score": round(best.score, 8),
            "ambiguity_margin": round(margin, 8),
            "second_candidate": results[1].text if len(results) > 1 else "",
            "second_score": round(results[1].score, 8) if len(results) > 1 else "",
            "avg_error": round(best.avg_error, 8),
            "start_err": round(best.start_err, 8),
            "checksum_err": round(best.checksum_err, 8),
            "stop_err": round(best.stop_err, 8),
            "profile": best.profile_name,
            "row_y": best.row_y,
            "x0_est": round(best.x0, 3),
            "module_px_est": round(best.module_px, 6),
            "patch_box_y0_y1_x0_x1": str(patch_box) if patch_box is not None else "",
            "standalone_zbar": "|".join(standalone_values),
            "patched_zbar": "|".join(patched_values),
            "direct_zbar_hits": "|".join(f"{h.symbology}:{h.data}" for h in zbar_hits[:10]),
            "standalone_file": str(standalone_path.relative_to(out_dir)),
            "patched_file": str(patched_path.relative_to(out_dir)) if patched_path else "",
            "seconds": round(time.time() - t0, 3),
        })
        print(f"  {status}: {best.text}  score={best.score:.4f}  sec={time.time() - t0:.2f}", flush=True)

    csv_path = out_dir / "barcode_repair_verification.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    make_contact_sheet(sheet_items, out_dir / "contact_sheet_rebuilt_code128.png")

    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out_dir.parent))

    print(f"CSV: {csv_path}", flush=True)
    print(f"ZIP: {zip_path}", flush=True)
    return zip_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal 1D barcode repair from stripe pixels; generic Code128 pixel rescue + zbar enhancement.")
    parser.add_argument("--input", required=True, help="输入图片、图片目录或 zip")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--charset", dest="mode", default="alnum", choices=["digits", "upper", "alnum", "printable"], help="Code128-B 像素恢复阶段的字符集约束；不是 OCR")
    parser.add_argument("--lengths", default="6-32", help="Code128-B payload 长度范围，例如 12、12,20、6-32")
    parser.add_argument("--accept-score", type=float, default=0.20, help="自动接受阈值，越小越严格；泛化场景建议同时看 ambiguity_margin")
    parser.add_argument("--max-profiles", type=int, default=6, help="每张图提取的扫描线/带状 profile 数量")
    parser.add_argument("--module-px", type=int, default=4, help="独立高清条码的模块宽度")
    parser.add_argument("--bar-height", type=int, default=120, help="独立高清条码高度")
    parser.add_argument("--no-patch", action="store_true", help="只输出独立条码，不替换回原图")
    parser.add_argument("--regex", default="", help="可选：payload 正则约束，用来过滤候选；例如 ^[0-9A-Z]{12}$")
    parser.add_argument("--template", default="", help="可选：逐位模板，用来消除低清 Code128 多解；#=数字 @=大写字母 *=大写字母数字 ?=可打印字符，多个模板逗号分隔，例如 4E26########,###########ES#######")
    parser.add_argument("--try-zbar", action="store_true", default=True, help="先尝试 zbar 多尺度增强直接扫码")
    parser.add_argument("--min-margin", type=float, default=0.015, help="第一候选和第二候选的最小分数差；低于此值标为 AMBIGUOUS")
    parser.add_argument("--clean", action="store_true", help="运行前清空输出目录")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
