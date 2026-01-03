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
from pyzbar import pyzbar

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
CLI_SCALE_FACTORS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
CLI_MAX_PIXELS = 12_000_000_000

PAD_X = 30
PAD_Y = 16

def _run_cli(cmd, **kwargs):
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs.setdefault("startupinfo", startupinfo)
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
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


# ===================== 澶氳搴?+ 鍙嶈壊 瑙ｇ爜 =====================

def _rotate90(img: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return img
    return np.rot90(img, k).copy()


def decode_with_transforms(gray_or_bin: np.ndarray,
                           tag: str) -> List[Dict]:
    """
    瀵瑰崟閫氶亾鍥惧儚鍋?
      - 0/90/180/270掳 鏃嬭浆
      - 姝ｅ父 / 鍙嶈壊
    鍐嶇敤 pyzbar 瑙ｇ爜銆?
    """
    results: List[Dict] = []
    seen = set()

    base = pad_quiet_zone(gray_or_bin)

    for k in range(4):
        rot = _rotate90(base, k)

        for inverted in [False, True]:
            if inverted:
                img = cv2.bitwise_not(rot)
            else:
                img = rot

            with _StderrSilencer():
                decoded = pyzbar.decode(img, symbols=[pyzbar.ZBarSymbol.CODE128])
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

def read_barcodes_cli(img_path: str) -> List[str]:
    if not BARCODE_CLI_PATH or not os.path.exists(BARCODE_CLI_PATH):
        return []

    cmd = [
        BARCODE_CLI_PATH,
        "-silent",
        "-type=code128",
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
            timeout=5,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def decode_with_cli(img_bgr: np.ndarray, tag: str) -> List[Dict]:
    img_bgr = pad_quiet_zone(img_bgr)
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, img_bgr)
        lines = read_barcodes_cli(tmp_path)
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


def decode_cli_multi(img: np.ndarray, tag: str) -> List[Dict]:
    base = img
    rotations = [0, 1, 2, 3]

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
                for r in decode_with_cli(cand, tag):
                    key = (r.get('data'), r.get('type'))
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(r)

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

    # ?????
    band = crop_bar_band(gray)

    # ?? band ?????????????????
    results: List[Dict] = decode_with_transforms(band, "band_raw")

    # BarcodeReaderCLI multi-pass on band (if available)
    results += decode_cli_multi(band, "band_cli")
    if results:
        return {"results": results}

    # ????
    enh_gray, bin_img = enhance_band(band)

    # ?????????
    results += decode_with_transforms(enh_gray, "band_enh_gray")
    results += decode_cli_multi(enh_gray, "band_cli_enh_gray")
    if results:
        return {"results": results}

    # ?????????
    results += decode_with_transforms(bin_img, "band_enh_bin")
    results += decode_cli_multi(bin_img, "band_cli_enh_bin")
    if results:
        return {"results": results}

    results += decode_cli_multi(gray, "full_cli")
    return {"results": results}

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
