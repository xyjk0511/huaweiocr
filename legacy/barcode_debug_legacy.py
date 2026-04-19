import os
import cv2
import json
import subprocess
import numpy as np
import shutil
import uuid

# ========= 按你自己的路径改 =========
BARCODE_CLI_PATH = r"C:\Users\55093\Downloads\BarcodeReaderCLI.win.12.0.7889\BarcodeReaderCLI\bin\BarcodeReaderCLI.exe"
SN_DIR = r"stage2_fields\sn"      # 你现在 sn 小图所在的目录
TMP_DIR = r"_tmp_barcode_debug"   # 临时目录（随便）

# ===================== 你只改这里 =====================
MODEL_CROP_DIR = r"stage2_fields\model"
SN_CROP_DIR    = r"stage2_fields\sn"

OUT_JSONL = r"scan_results_ocr.jsonl"
OUT_FAILED_DIR = r"scan_failed_ocr"

UPSCALE_TARGET_W = 2200   # 大图 / 其他用
MAX_SCALE        = 4.0

# —— SN 小图专用放大参数 ——
SN_TARGET_WIDTHS = [1000, 1400, 1800]   # 多档宽度轮流试
SN_MAX_SCALE     = 4.0                  # 防止放得太夸张

SN_TEXT_ROI_TOP_RATIO = 0.60
USE_ROTATIONS = True
# ======================================================

os.makedirs(TMP_DIR, exist_ok=True)

def _run_cli(cmd, **kwargs):
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs.setdefault("startupinfo", startupinfo)
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    return subprocess.run(cmd, **kwargs)


def call_barcode_cli(img_path, tag="pass"):
    """
    调一次 BarcodeReaderCLI，打印完整 stdout/stderr 和解析到的 code 列表。
    返回 codes（list[str]）
    """
    cmd = [BARCODE_CLI_PATH, img_path, "--format", "json"]

    try:
        p = _run_cli(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15
        )
    except Exception as e:
        print(f"  [{tag}] RUN ERR : {repr(e)}")
        return []

    out = p.stdout.decode("utf-8", errors="ignore")
    
    codes = []
    try:
        j = json.loads(out)
        def collect_from_obj(obj):
            nonlocal codes
            if isinstance(obj, dict):
                v = (
                    obj.get("text")
                    or obj.get("value")
                    or obj.get("barcodeText")
                    or obj.get("BarcodeText")
                )
                if v:
                    codes.append(v)
                if "sessions" in obj:
                    for s in obj["sessions"]:
                        collect_from_obj(s)
                if "barcodes" in obj:
                    for bc in obj["barcodes"]:
                        collect_from_obj(bc)
                if "results" in obj:
                    for r in obj["results"]:
                        collect_from_obj(r)
            elif isinstance(obj, list):
                for x in obj:
                    collect_from_obj(x)
        collect_from_obj(j)
    except Exception:
        pass
    return codes


def pad_quiet_zone(img: np.ndarray,
                   pad_y: int = 20,
                   pad_x: int = 40) -> np.ndarray:
    """
    在 SN 小图外面加一圈 padding，模拟条码的 quiet zone。
    颜色用四个角的中位数，避免把背景搞错。
    """
    h, w = img.shape[:2]

    if img.ndim == 2:
        corners = np.concatenate([
            img[0:5, 0:5].ravel(),
            img[0:5, -5:].ravel(),
            img[-5:, 0:5].ravel(),
            img[-5:, -5:].ravel(),
        ])
        value = int(np.median(corners))
    else:
        corners = np.concatenate([
            img[0:5, 0:5].reshape(-1, 3),
            img[0:5, -5:].reshape(-1, 3),
            img[-5:, 0:5].reshape(-1, 3),
            img[-5:, -5:].reshape(-1, 3),
        ], axis=0)
        value = [int(x) for x in np.median(corners, axis=0)]

    padded = cv2.copyMakeBorder(
        img,
        pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=value,
    )
    return padded


def resize_to_width(img_bgr, target_w):
    h, w = img_bgr.shape[:2]
    if w == target_w:
        return img_bgr, 1.0

    scale = target_w / float(w)
    # 限制最大放大倍数
    if scale > SN_MAX_SCALE:
        scale = SN_MAX_SCALE
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def extract_sn_bar_band(img):
    """
    假设 SN 小图中，上面是文字，下面是条码。
    这里简单粗暴取下半部分。
    """
    h, w = img.shape[:2]
    start_y = int(h * 0.5) 
    return img[start_y:, :]


def enhance_bar_band(img):
    """
    转灰度 + Otsu 二值化增强
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if img.ndim == 3:
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    return bin_img


def gen_small_rotations(img):
    """
    生成微小旋转版本：0度, -2度, +2度
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    yield "rot0", img
    
    for angle in [-2, 2]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        yield f"rot{angle}", rot


def decode_barcode_img(img):
    """
    保存临时图片并调用 CLI
    """
    tmp_name = f"tmp_{uuid.uuid4().hex}.png"
    tmp_path = os.path.join(TMP_DIR, tmp_name)
    cv2.imwrite(tmp_path, img)
    
    codes = call_barcode_cli(tmp_path, tag="mem")
    
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return codes


def decode_sn_with_dbr(sn_path: str) -> list[str]:
    print("\n================ SN DEBUG ==================")
    print(f"[FILE] {sn_path}")

    # 支持中文路径读取
    try:
        data = np.fromfile(sn_path, np.uint8)
        img0 = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img0 = None
    
    if img0 is None:
        img0 = cv2.imread(sn_path)

    if img0 is None:
        print("  [SN] 读取失败")
        print("[FINAL] codes=[]")
        print("===========================================")
        return []

    h0, w0 = img0.shape[:2]

    # ---------- 1) 基础候选：原图 + (竖图)旋转 + QuietZone ----------
    base_views = []

    # 原图
    base_views.append(("orig", img0))

    # 如果是很瘦高的图，额外加一个 90° 旋转的版本
    if h0 > w0 * 1.3:
        rot90 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        base_views.append(("rot90ccw", rot90))
        
    # 也可以加一个带 padding 的版本
    padded = pad_quiet_zone(img0)
    base_views.append(("qzpad", padded))

    # ---------- 2) 展开：每个 base 生成 band / band_bin / small rotations ----------
    tried_any = False

    for base_tag, base_img in base_views:
        hh, ww = base_img.shape[:2]
        print(f"  [SN] 基础视图 {base_tag}: {ww}x{hh}")

        # 如果是横向 SN 图，就裁出条码带子 & 增强
        candidates = [(f"{base_tag}_full", base_img)]

        if ww >= hh:  # 横向
            band = extract_sn_bar_band(base_img)
            band_bin = enhance_bar_band(band)
            candidates.append((f"{base_tag}_band", band))
            candidates.append((f"{base_tag}_band_bin", band_bin))

        # ---------- 3) 对每个 candidate 做小角度旋转 + 多尺寸解码 ----------
        for cand_tag, cand_img in candidates:
            ch, cw = cand_img.shape[:2]
            # print(f"    [SN] 候选 {cand_tag}: {cw}x{ch}")

            for rot_tag, rot_img in gen_small_rotations(cand_img):
                rh, rw = rot_img.shape[:2]
                # print(f"      [SN] {cand_tag}/{rot_tag} 原尺寸 {rw}x{rh}")
                tried_any = True

                for tw in SN_TARGET_WIDTHS:
                    # 小图才放大，大图可以不放那么狠
                    resized, scale = resize_to_width(rot_img, tw)
                    r_h, r_w = resized.shape[:2]

                    # 如果缩放前后差不多，就少打一条 log
                    if abs(scale - 1.0) < 1e-3:
                        print(f"        [SN] 尝试 {cand_tag}/{rot_tag} 宽度 {tw} (无需缩放)")
                    else:
                        print(f"        [SN] 尝试 {cand_tag}/{rot_tag} 放大{scale:.2f}倍 -> {r_w}x{r_h}")

                    # === 这里调用你原来的条码解码函数 ===
                    codes = decode_barcode_img(resized)

                    if codes:
                        print(f"        [SN] 成功 {cand_tag}/{rot_tag} (宽度 {tw}) -> {codes}")
                        print(f"[FINAL] codes={codes}")
                        print("===========================================")
                        return codes

    if not tried_any:
        print("  [SN] 没有生成任何候选？检查一下逻辑")

    print("  [SN] 所有候选都未读出条码")
    print("[FINAL] codes=[]")
    
    # 失败时保存到 failed 目录
    os.makedirs(OUT_FAILED_DIR, exist_ok=True)
    dst = os.path.join(OUT_FAILED_DIR, os.path.basename(sn_path))
    try:
        shutil.copy(sn_path, dst)
    except Exception:
        pass
        
    print("===========================================")
    return []


def debug_one_sn(sn_path):
    decode_sn_with_dbr(sn_path)


def main():
    # 指定要测试的文件列表
    TARGET_FILES = [
        "10__label_2__sn.png",
        "2__label_1__sn.png",
        "3__label_1__sn.png",
        "3__label_4__sn.png",
        "3__label_7__sn.png",
        "3__label_8__sn.png",
        "4__label_1__sn.png",
        "4__label_3__sn.png",
        "5__label_3__sn.png",
        "5__label_4__sn.png",
        "5__label_5__sn.png",
        "5__label_6__sn.png"
    ]

    if not os.path.exists(SN_DIR):
        print(f"Directory not found: {SN_DIR}")
        return

    cnt = 0
    # 遍历目录下的所有文件，如果文件名在 TARGET_FILES 中则处理
    # 这样可以避免硬编码路径问题，同时支持模糊匹配（如果需要）
    # 这里采用精确匹配文件名
    
    # 为了确保找到文件，我们先列出目录下的所有文件
    all_files = os.listdir(SN_DIR)
    
    # 过滤出存在于 TARGET_FILES 中的文件
    files_to_process = []
    for target in TARGET_FILES:
        if target in all_files:
            files_to_process.append(target)
        else:
            print(f"Warning: File {target} not found in {SN_DIR}")
    
    for name in files_to_process:
        sn_path = os.path.join(SN_DIR, name)
        debug_one_sn(sn_path)
        cnt += 1

    print(f"\n共调试 {cnt} 张 SN 小图。")


if __name__ == "__main__":
    main()
