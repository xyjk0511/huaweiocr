"""Image exports shared by the current and HEAD stage1 review tools."""
import os
from pathlib import Path

import cv2
import numpy as np


def save_image(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if path.suffix.lower() in {".jpg", ".jpeg"} else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(buf.tobytes())


def safe_rel_name(rel_path: str) -> str:
    return os.path.splitext(rel_path)[0].replace("/", "__").replace("\\", "__")


def map_box_back_from_rotation(box, original_shape, rotation):
    if box is None or rotation not in {90, 180, 270}:
        return box
    h, w = original_shape[:2]
    x1, y1, x2, y2 = box

    def inverse_point(xr, yr):
        if rotation == 90:
            return yr, h - xr
        if rotation == 180:
            return w - xr, h - yr
        return w - yr, xr

    mapped = [inverse_point(x, y) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
    xs, ys = zip(*mapped)
    ox1 = max(0, min(w, int(round(min(xs)))))
    oy1 = max(0, min(h, int(round(min(ys)))))
    ox2 = max(0, min(w, int(round(max(xs)))))
    oy2 = max(0, min(h, int(round(max(ys)))))
    return (ox1, oy1, ox2, oy2) if ox2 > ox1 and oy2 > oy1 else None


def draw_overlay(img, entries):
    overlay = img.copy()
    for idx, entry in enumerate(entries, start=1):
        box = entry.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(
            overlay, f"det{idx}:{float(entry.get('confidence', 0.0)):.3f}",
            (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 0, 255), 2, cv2.LINE_AA,
        )
    return overlay


def build_contact_sheet(image_paths, out_path: Path, read_image, thumb_w=420, thumb_h=300, cols=3):
    thumbs = []
    for image_path in image_paths:
        img = read_image(str(image_path))
        if img is None:
            continue
        canvas = cv2.copyMakeBorder(img, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h, w = canvas.shape[:2]
        scale = min(thumb_w / max(1, w), thumb_h / max(1, h))
        resized = cv2.resize(canvas, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        thumb = np.full((thumb_h, thumb_w, 3), 255, dtype="uint8")
        y0 = (thumb_h - resized.shape[0]) // 2
        x0 = (thumb_w - resized.shape[1]) // 2
        thumb[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        cv2.putText(thumb, Path(image_path).stem[:50], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
        thumbs.append(thumb)

    if not thumbs:
        return
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype="uint8")
    for idx, thumb in enumerate(thumbs):
        y0, x0 = idx // cols * thumb_h, idx % cols * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
    save_image(out_path, sheet)
