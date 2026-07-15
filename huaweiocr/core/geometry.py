import numpy as np


SLANT_GUARD_ANGLE_DEG = 6.0


def pred_class(p):
    return p.get("class") or p.get("class_name") or ""

def to_xywh_topleft(p):
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    return x1, y1, int(w), int(h)


def slant_guard_px(width, max_px, angle_deg=SLANT_GUARD_ANGLE_DEG, min_px=2):
    if max_px <= 0 or width <= 0:
        return 0
    guard = int(np.ceil(float(width) * np.tan(np.deg2rad(angle_deg)) * 0.5))
    return max(0, min(int(max_px), max(int(min_px), guard)))

def box_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter) / float(max(1, union))


def box_overlap_ratio(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(max(1, min(area_a, area_b)))


def crop_from_box(img, box):
    if box is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    # Clamp to image bounds. A negative coordinate would otherwise be treated by
    # numpy as an index "from the end", silently returning a shifted or empty
    # crop instead of the intended region (positive over-bounds numpy already
    # clips, so behaviour is unchanged there).
    x1 = max(0, min(int(x1), W))
    y1 = max(0, min(int(y1), H))
    x2 = max(0, min(int(x2), W))
    y2 = max(0, min(int(y2), H))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size else None


def expand_box_pixels(img, box, pad_x=0, pad_y=0):
    if box is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(W, int(x2 + pad_x)),
        min(H, int(y2 + pad_y)),
    )

def union_boxes(*boxes):
    boxes = [b for b in boxes if b is not None and b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
