from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


CODE128_PATTERNS: list[str] = [
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


def _pattern_to_bits(pattern: str) -> np.ndarray:
    bits = []
    black = 1
    for ch in pattern:
        bits.extend([black] * int(ch))
        black = 1 - black
    return np.asarray(bits, dtype=np.float32)


_BITS = [_pattern_to_bits(pattern) for pattern in CODE128_PATTERNS]
_BITS11 = np.stack(_BITS[:106]).astype(np.float32)
_STOP_BITS = _BITS[STOP].astype(np.float32)


@dataclass(frozen=True)
class RescueCandidate:
    text: str
    score: float
    avg_error: float
    checksum_err: float
    stop_err: float
    row_y: int
    x0_est: float
    module_px_est: float
    n_modules: int
    data_units: int
    method: str


def _normalize01(raw: np.ndarray, lo_q: float = 2, hi_q: float = 98) -> np.ndarray:
    raw = raw.astype(np.float32)
    if raw.size == 0:
        return raw
    lo, hi = np.percentile(raw, [lo_q, hi_q])
    if hi <= lo + 1e-4:
        lo, hi = float(raw.min()), float(raw.max() + 1e-4)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _profile_from_band(gray: np.ndarray, y0: int, y1: int, mode: str) -> np.ndarray:
    crop = gray[y0:y1, :].astype(np.float32)
    if mode == "mean":
        white = crop.mean(axis=0)
    elif mode == "median":
        white = np.percentile(crop, 50, axis=0)
    elif mode.startswith("p"):
        white = np.percentile(crop, int(mode[1:]), axis=0)
    else:
        raise ValueError(mode)
    return _normalize01(255.0 - white)


def _make_profiles(gray: np.ndarray, max_profiles: int = 8) -> list[tuple[str, int, np.ndarray]]:
    h, _w = gray.shape
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    row_score = np.mean(np.abs(sx), axis=1)
    k = max(5, (h // 80) * 2 + 1)
    smooth = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, k), 0).ravel()
    bands: list[tuple[int, int]] = []
    min_sep = max(8, h // 8)
    for y in np.argsort(smooth)[::-1]:
        y = int(y)
        if any(abs(y - (a + b) // 2) < min_sep for a, b in bands):
            continue
        thresh = max(float(np.percentile(smooth, 60)), float(smooth[y]) * 0.35)
        a, b = y, y + 1
        while a > 0 and smooth[a - 1] >= thresh:
            a -= 1
        while b < h and smooth[b] >= thresh:
            b += 1
        min_h = max(8, h // 8)
        if b - a < min_h:
            pad = (min_h - (b - a)) // 2 + 1
            a, b = max(0, y - pad), min(h, y + pad)
        max_h = max(20, h // 2)
        if b - a > max_h:
            a, b = max(0, y - max_h // 2), min(h, y + max_h // 2)
        bands.append((a, b))
        if len(bands) >= max(3, max_profiles // 2):
            break

    profiles: list[tuple[str, int, np.ndarray]] = []
    seen = set()
    for band_idx, (a, b) in enumerate(bands):
        for mode in ("p35", "mean", "median"):
            key = (a, b, mode)
            if key in seen:
                continue
            seen.add(key)
            profiles.append((f"band{band_idx}_{mode}", (a + b) // 2, _profile_from_band(gray, a, b, mode)))
            if len(profiles) >= max_profiles:
                return profiles
    if not profiles:
        profiles.append(("full_mean", h // 2, _profile_from_band(gray, 0, h, "mean")))
    return profiles[:max_profiles]


def _extent_candidates(profile: np.ndarray, max_extents: int = 45) -> list[tuple[float, float]]:
    p = profile.astype(np.float32)
    w = len(p)
    out: list[tuple[float, float]] = []
    thresholds = [0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64]
    thresholds.extend(float(np.percentile(p, q)) for q in (55, 62, 70, 78, 84, 90))
    for t in sorted(set(round(float(value), 4) for value in thresholds)):
        mask = (p > t).astype(np.uint8)
        changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, w]
        vals = mask[starts]
        runs = np.where(vals == 1)[0]
        if len(runs) < 3:
            continue
        max_skip = min(10, len(runs) - 1)
        for skip_left in range(max_skip + 1):
            for skip_right in range(max_skip + 1):
                if len(runs) <= skip_left + skip_right:
                    continue
                x0 = float(starts[runs[skip_left]])
                x1 = float(ends[runs[-1 - skip_right]])
                if x1 - x0 >= max(35, 0.18 * w):
                    out.append((x0, x1))

    grad = np.abs(np.gradient(p))
    for q in (70, 78, 85, 90, 94):
        xs = np.where(grad > np.percentile(grad, q))[0]
        if len(xs) < 8:
            continue
        x0 = float(np.percentile(xs, 1))
        x1 = float(np.percentile(xs, 99))
        if x1 - x0 >= max(35, 0.18 * w):
            out.append((x0, x1))

    uniq: list[tuple[float, float]] = []
    for x0, x1 in sorted(out, key=lambda item: item[1] - item[0], reverse=True):
        if all(abs(x0 - prev_x0) > 3 or abs(x1 - prev_x1) > 3 for prev_x0, prev_x1 in uniq):
            uniq.append((x0, x1))
        if len(uniq) >= max_extents:
            break
    return uniq


def _sample_modules(profile: np.ndarray, x0: float, module_px: float, n_modules: int) -> np.ndarray:
    xs = x0 + (np.arange(n_modules, dtype=np.float32) + 0.5) * module_px
    xs = np.clip(xs, 0, len(profile) - 1)
    return np.interp(xs, np.arange(len(profile), dtype=np.float32), profile).astype(np.float32)


def _symbol_costs(seg: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    return ((patterns - seg[None, :]) ** 2).mean(axis=1)


def _allowed_code128b_codes(charset: str) -> list[int]:
    if charset == "digits":
        return [ord(ch) - 32 for ch in "0123456789"]
    if charset == "alnum":
        return [ord(ch) - 32 for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    if charset == "printable":
        return list(range(95))
    return list(range(95))


def _decode_code128b_fixed(
    sampled: np.ndarray,
    text_len: int,
    charset: str,
) -> tuple[str, float, float, float, float, float] | None:
    codeword_count = text_len + 2
    segs = sampled[: 11 * codeword_count].reshape(codeword_count, 11)
    stop_seg = sampled[11 * codeword_count:]
    start_err = float(_symbol_costs(segs[0], _BITS11[[START_B]])[0])
    stop_err = float(((_STOP_BITS - stop_seg) ** 2).mean())
    if start_err > 1.10 or stop_err > 1.10:
        return None

    allowed = np.asarray(_allowed_code128b_codes(charset), dtype=np.int16)
    per_pos_costs = []
    for pos in range(text_len):
        per_pos_costs.append(_symbol_costs(segs[1 + pos], _BITS11[allowed]).astype(np.float32))

    inf = np.float32(1e9)
    dp = np.full(103, inf, dtype=np.float32)
    dp[START_B % 103] = np.float32(start_err)
    prev_mod = []
    prev_idx = []
    for pos, costs in enumerate(per_pos_costs, start=1):
        ndp = np.full(103, inf, dtype=np.float32)
        pm = np.full(103, -1, dtype=np.int16)
        pi = np.full(103, -1, dtype=np.int16)
        for mod in np.where(dp < inf / 2)[0]:
            next_mods = (mod + pos * allowed) % 103
            vals = dp[mod] + costs
            for idx, (next_mod, val) in enumerate(zip(next_mods, vals)):
                if val < ndp[next_mod]:
                    ndp[next_mod] = val
                    pm[next_mod] = mod
                    pi[next_mod] = idx
        dp = ndp
        prev_mod.append(pm)
        prev_idx.append(pi)

    checksum_costs = _symbol_costs(segs[-1], _BITS11[:103]).astype(np.float32)
    totals = dp + checksum_costs + np.float32(stop_err)
    checksum = int(np.argmin(totals))
    total = float(totals[checksum])
    checksum_err = float(checksum_costs[checksum])
    codes = []
    cur = checksum
    for pos in range(text_len - 1, -1, -1):
        idx = int(prev_idx[pos][cur])
        pm = int(prev_mod[pos][cur])
        if idx < 0 or pm < 0:
            return None
        codes.append(int(allowed[idx]))
        cur = pm
    codes.reverse()
    text = "".join(chr(code + 32) for code in codes)
    avg = total / (text_len + 3)
    return text, total, avg, start_err, checksum_err, stop_err


def _geometry_candidates_code128(
    profile: np.ndarray,
    unit_counts: Sequence[int],
    max_extents: int = 32,
) -> list[tuple[float, int, float, int, float]]:
    width = len(profile)
    phase_grid = (-0.8, -0.4, 0.0, 0.4, 0.8)
    scales = (0.985, 1.0, 1.015)
    out: list[tuple[float, int, float, int, float]] = []
    for x0e, x1e in _extent_candidates(profile, max_extents=max_extents):
        extent_w = x1e - x0e
        for units in unit_counts:
            n_modules = 11 * (units + 2) + 13
            base = extent_w / n_modules
            if not (0.85 <= base <= 18.0):
                continue
            for ph in phase_grid:
                x0 = x0e + ph * base
                if x0 < -2:
                    continue
                for scale in scales:
                    module_px = base * scale
                    if x0 + n_modules * module_px >= width + 2:
                        continue
                    start = _sample_modules(profile, x0, module_px, 11)
                    stop = _sample_modules(profile, x0 + (n_modules - 13) * module_px, module_px, 13)
                    start_err = float(_symbol_costs(start, _BITS11[[START_B]]).min())
                    stop_err = float(((_STOP_BITS - stop) ** 2).mean())
                    start_stop = start_err + stop_err
                    if start_err < 0.80 and stop_err < 0.90:
                        out.append((start_stop, units, x0, n_modules, module_px))
    out.sort(key=lambda item: item[0])
    dedup = []
    seen = set()
    for geometry in out:
        key = (geometry[1], round(geometry[2], 1), round(geometry[4], 3))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(geometry)
        if len(dedup) >= 180:
            break
    return dedup


def decode_part_no_candidates(
    gray_or_bgr: np.ndarray,
    *,
    lengths: Sequence[int] = (8,),
    charset: str = "digits",
    max_profiles: int = 8,
) -> list[RescueCandidate]:
    if gray_or_bgr is None:
        return []
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr.copy()
    if gray.size == 0:
        return []

    results: dict[str, RescueCandidate] = {}
    for _profile_name, row_y, profile in _make_profiles(gray, max_profiles=max_profiles):
        geometries = _geometry_candidates_code128(profile, lengths, max_extents=28)
        for start_stop, units, x0, n_modules, module_px in geometries[:80]:
            sampled = _sample_modules(profile, x0, module_px, n_modules)
            decoded = _decode_code128b_fixed(sampled, units, charset=charset)
            if decoded is None:
                continue
            text, _total, avg, _start_err, checksum_err, stop_err = decoded
            if not text or not all("0" <= ch <= "9" for ch in text):
                continue
            score = float(avg + 0.015 * start_stop)
            candidate = RescueCandidate(
                text=text,
                score=score,
                avg_error=float(avg),
                checksum_err=float(checksum_err),
                stop_err=float(stop_err),
                row_y=int(row_y),
                x0_est=float(x0),
                module_px_est=float(module_px),
                n_modules=int(n_modules),
                data_units=int(units),
                method="soft_code128b_checksum:digits",
            )
            existing = results.get(text)
            if existing is None or candidate.score < existing.score:
                results[text] = candidate
    return sorted(results.values(), key=lambda item: (item.score, item.avg_error, -len(item.text)))
