"""Text-line geometry from connected components, independent of pipeline state."""

import cv2

def text_component_line_box_near_y(
    img,
    x1,
    x2,
    y1,
    y2,
    target_y,
    min_span_w,
    pad_top_min=5,
    pad_bottom_min=5,
    pad_y_ratio=0.22,
):
    H, W = img.shape[:2]
    x1 = max(0, min(W, int(x1)))
    x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H, int(y1)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    components = cv2.connectedComponentsWithStats(bw, 8)
    if len(components) < 4:
        return None
    _, _, stats, _ = components

    comps = []
    roi_h, roi_w = bw.shape[:2]
    for i in range(1, len(stats)):
        cx, cy, cw, ch, area = stats[i]
        if area < 6 or cw < 2 or ch < 4:
            continue
        if ch > max(48, int(roi_h * 0.75)):
            continue
        if cw > int(roi_w * 0.80):
            continue
        fill = float(area) / float(max(1, cw * ch))
        if fill < 0.08:
            continue
        comps.append((cx, cy, cw, ch, area))
    if not comps:
        return None

    comps.sort(key=lambda c: c[1] + c[3] / 2.0)
    clusters = []
    for comp in comps:
        cy = comp[1] + comp[3] / 2.0
        if not clusters:
            clusters.append({"items": [comp], "sum_cy": cy, "count": 1})
            continue
        cluster = clusters[-1]
        mean_cy = cluster["sum_cy"] / cluster["count"]
        if abs(cy - mean_cy) <= max(9, comp[3] * 0.85):
            cluster["items"].append(comp)
            cluster["sum_cy"] += cy
            cluster["count"] += 1
        else:
            clusters.append({"items": [comp], "sum_cy": cy, "count": 1})

    best = None
    best_score = None
    for cluster in clusters:
        items = cluster["items"]
        gx1 = min(c[0] for c in items)
        gy1 = min(c[1] for c in items)
        gx2 = max(c[0] + c[2] for c in items)
        gy2 = max(c[1] + c[3] for c in items)
        span_w = gx2 - gx1
        line_h = gy2 - gy1
        avg_w = sum(c[2] for c in items) / float(len(items))
        if span_w < min_span_w or line_h < 8:
            continue
        if len(items) < 3 and avg_w < 8:
            continue
        if len(items) >= 5 and avg_w <= 3.5:
            continue
        center_y = y1 + (gy1 + gy2) / 2.0
        barcode_like_penalty = 14 if len(items) >= 8 and avg_w < 7.0 else 0
        score = abs(center_y - target_y) + barcode_like_penalty
        if best_score is None or score < best_score:
            best_score = score
            best = (x1, y1 + gy1, x2, y1 + gy2)
    if best is None:
        return None

    line_h = best[3] - best[1]
    pad_top = max(pad_top_min, int(line_h * pad_y_ratio))
    pad_bottom = max(pad_bottom_min, int(line_h * pad_y_ratio))
    return (
        best[0],
        max(0, best[1] - pad_top),
        best[2],
        min(H, best[3] + pad_bottom),
    )
