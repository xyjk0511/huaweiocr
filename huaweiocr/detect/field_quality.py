"""Field-crop validation, trimming and padding using only label-local pixels."""

import cv2
import numpy as np

from huaweiocr.detect.barcode_regions import (
    _part_no_barcode_like_boxes_in_region,
    barcode_like_box_in_region,
)
from huaweiocr.detect.text_regions import text_component_line_box_near_y

def trim_model_top_fragment(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 40 or w < 80:
        return crop_img

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_dark = (bw > 0).mean(axis=1)
    candidate_rows = row_dark > 0.03
    runs = []
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            runs.append((start, end))
            start = None

    if len(runs) < 2:
        return crop_img
    first_start, first_end = runs[0]
    second_start, second_end = runs[1]
    first_h = first_end - first_start
    second_h = second_end - second_start
    if first_start == 0 and first_h <= max(14, int(h * 0.10)) and second_h >= 20 and second_start - first_end <= 4:
        trim_y = max(0, second_start - 3)
        if trim_y > 0:
            return crop_img[trim_y:h, :]
    return crop_img


def trim_model_text_only_crop(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 36 or w < 80:
        return crop_img

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_dark = (bw > 0).mean(axis=1)
    candidate_rows = row_dark > 0.03
    runs = []
    start = None
    for i, value in enumerate(candidate_rows):
        if value and start is None:
            start = i
        at_end = i == len(candidate_rows) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            runs.append((start, end))
            start = None

    if len(runs) < 2:
        return crop_img
    first_start, first_end = runs[0]
    second_start, second_end = runs[1]
    first_h = first_end - first_start
    second_h = second_end - second_start
    gap = second_start - first_end
    if first_h >= 20 and gap >= 3 and second_h <= max(14, int(h * 0.25)):
        bottom = min(h, first_end + max(3, int(first_h * 0.12)))
        return crop_img[:bottom, :]
    return crop_img


def crop_contains_1d_barcode(
    crop_img,
    min_span_ratio=0.28,
    row_trans_threshold=0.18,
    active_threshold=0.35,
):
    return (
        crop_1d_barcode_box(
            crop_img,
            min_span_ratio=min_span_ratio,
            row_trans_threshold=row_trans_threshold,
            active_threshold=active_threshold,
        )
        is not None
    )


def crop_1d_barcode_box(
    crop_img,
    min_span_ratio=0.28,
    row_trans_threshold=0.18,
    active_threshold=0.35,
):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 18 or w < 80:
        return None
    return barcode_like_box_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(12, int(h * 0.12)),
        min_span_w=max(70, int(w * min_span_ratio)),
        row_trans_threshold=row_trans_threshold,
        active_threshold=active_threshold,
    )


def crop_has_complete_1d_barcode(
    crop_img,
    min_span_ratio=0.28,
    edge_guard_px=2,
    row_trans_threshold=0.18,
    active_threshold=0.35,
):
    box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=min_span_ratio,
        row_trans_threshold=row_trans_threshold,
        active_threshold=active_threshold,
    )
    if box is None:
        return False
    h, w = crop_img.shape[:2]
    x1, y1, x2, y2 = box
    edge_guard_x = max(2, int(edge_guard_px))
    return x1 > edge_guard_x and x2 < w - edge_guard_x and y1 > 1 and y2 < h - 1


def model_crop_barcode_box(crop_img):
    box = crop_1d_barcode_box(crop_img)
    if box is not None:
        return box
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    return barcode_like_box_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(12, int(h * 0.16)),
        min_span_w=max(70, int(w * 0.18)),
        row_trans_threshold=0.04,
        active_threshold=0.25,
    )


def model_crop_has_complete_1d_barcode(crop_img):
    box = model_crop_barcode_box(crop_img)
    if box is None:
        return False
    h, w = crop_img.shape[:2]
    x1, y1, x2, y2 = box
    return x1 > 2 and x2 < w - 2 and y1 > 1 and y2 < h - 1


def model_crop_has_text_above_barcode(crop_img):
    box = model_crop_barcode_box(crop_img)
    if box is None:
        return False
    h, w = crop_img.shape[:2]
    _x1, y1, _x2, _y2 = box
    text_y2 = max(0, int(y1) - 2)
    if text_y2 < max(12, int(h * 0.18)):
        return False
    text_box = text_component_line_box_near_y(
        crop_img,
        0,
        w,
        0,
        text_y2,
        target_y=text_y2 / 2.0,
        min_span_w=max(55, int(w * 0.10)),
        pad_top_min=1,
        pad_bottom_min=1,
        pad_y_ratio=0.0,
    )
    return text_box is not None


def part_no_crop_contains_1d_barcode(crop_img, min_span_ratio=0.16):
    return crop_contains_1d_barcode(
        crop_img,
        min_span_ratio=min_span_ratio,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )


def part_no_crop_has_complete_1d_barcode(crop_img, min_span_ratio=0.18):
    return crop_has_complete_1d_barcode(
        crop_img,
        min_span_ratio=min_span_ratio,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )


def part_no_crop_has_partial_1d_barcode(crop_img, min_span_ratio=0.20):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return False

    boxes = _part_no_barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(6, int(h * 0.03)),
        min_span_w=max(40, int(w * min_span_ratio)),
        row_trans_threshold=0.08,
        active_threshold=0.20,
    )
    if not boxes:
        return False

    box = boxes[0]["box"]
    x1, y1, x2, y2 = box
    span_w = x2 - x1
    run_h = y2 - y1
    if span_w < max(70, int(w * min_span_ratio)):
        return False
    if run_h < max(8, int(h * 0.03)):
        return False
    if y1 > h * 0.72:
        return False
    if y2 < h * 0.12:
        return False
    return True


def part_no_crop_has_edge_clipped_barcode(crop_img, min_span_ratio=0.16, edge_guard_px=2):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return False

    box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=min_span_ratio,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if box is None:
        bands = _part_no_barcode_like_boxes_in_region(
            crop_img,
            0,
            0,
            w,
            h,
            min_run_h=max(6, int(h * 0.03)),
            min_span_w=max(40, int(w * min_span_ratio)),
            row_trans_threshold=0.08,
            active_threshold=0.20,
        )
        if not bands:
            return False
        box = bands[0]["box"]

    x1, _y1, x2, _y2 = box
    edge_guard_x = max(2, int(edge_guard_px))
    return x1 <= edge_guard_x or x2 >= w - edge_guard_x


def part_no_crop_is_structurally_valid(crop_img, allow_partial=False):
    if crop_img is None:
        return False
    if not part_no_crop_has_text_above_barcode(crop_img):
        return False
    if part_no_crop_has_lower_neighbor_content(crop_img):
        return False
    if part_no_crop_has_edge_clipped_barcode(crop_img):
        return False
    if part_no_crop_has_complete_1d_barcode(crop_img, min_span_ratio=0.18):
        return True
    if allow_partial and part_no_crop_has_partial_1d_barcode(crop_img, min_span_ratio=0.20):
        return True
    return part_no_crop_contains_1d_barcode(crop_img, min_span_ratio=0.16)


def pad_part_no_crop_quiet_zone(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return crop_img

    barcode_box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.18,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if barcode_box is None:
        return crop_img

    x1, _y1, x2, y2 = barcode_box
    barcode_w = max(1, x2 - x1)
    barcode_h = max(1, y2 - _y1)
    required_side = max(16, int(barcode_w * 0.08))
    required_bottom = max(16, int(barcode_h * 0.35))
    current_left = x1
    current_right = w - x2
    current_bottom = h - y2
    pad_left = max(0, required_side - current_left)
    pad_right = max(0, required_side - current_right)
    pad_bottom = max(0, required_bottom - current_bottom)
    if pad_left == 0 and pad_right == 0 and pad_bottom == 0:
        return crop_img

    return cv2.copyMakeBorder(
        crop_img,
        0,
        int(pad_bottom),
        int(pad_left),
        int(pad_right),
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _part_no_selected_barcode_overlaps_band(crop_img, band_box):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return False

    barcode_box = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.18,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if barcode_box is None:
        return False

    overlap = max(0, min(barcode_box[3], band_box[3]) - max(barcode_box[1], band_box[1]))
    band_h = max(1, band_box[3] - band_box[1])
    barcode_h = max(1, barcode_box[3] - barcode_box[1])
    return overlap >= max(4, int(min(band_h, barcode_h) * 0.45))


def _part_no_two_band_has_lower_neighbor(crop_img, first, second):
    h = crop_img.shape[0]
    gap = second[1] - first[3]
    if gap < max(8, int(h * 0.035)):
        return False
    if first[1] < max(18, int(h * 0.15)):
        return False
    if not _part_no_has_text_above_band(crop_img, first):
        return False
    return _part_no_selected_barcode_overlaps_band(crop_img, first)


def _part_no_has_text_above_band(crop_img, band_box):
    top = max(0, int(band_box[1]))
    if top < 8:
        return False
    region = crop_img[:top, :]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_dark = (bw > 0).mean(axis=1)
    active_rows = row_dark > 0.018
    min_run = max(3, int(crop_img.shape[0] * 0.025))
    start = None
    for i, active in enumerate(active_rows):
        if active and start is None:
            start = i
        at_end = i == len(active_rows) - 1
        if start is not None and ((not active) or at_end):
            end = i if not active else i + 1
            if end - start >= min_run:
                return True
            start = None
    return False


def _part_no_cutoff_before_two_band_neighbor(crop_img, first, second):
    h, _w = crop_img.shape[:2]
    gap_top = max(0, first[3])
    gap_bottom = max(gap_top, second[1])
    first_h = max(1, first[3] - first[1])
    min_text_gap = max(10, int(first_h * 0.50), int(h * 0.06))
    if gap_bottom - gap_top >= max(8, int(h * 0.035)):
        gap_img = crop_img[gap_top:gap_bottom]
        gray = cv2.cvtColor(gap_img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_dark = (bw > 0).mean(axis=1)
        active_rows = row_dark > 0.018
        min_run = max(3, int(h * 0.018))
        start = None
        for i, active in enumerate(active_rows):
            if active and start is None:
                start = i
            at_end = i == len(active_rows) - 1
            if start is not None and ((not active) or at_end):
                end = i if not active else i + 1
                if end - start >= min_run:
                    cutoff = gap_top + start - max(2, int(h * 0.015))
                    if cutoff - first[3] >= min_text_gap:
                        return max(first[3] + 2, min(h - 1, cutoff))
                start = None

    return max(first[3] + 2, second[1] - max(6, int(h * 0.025)))


def _part_no_primary_barcode_box(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    selected = crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.18,
        row_trans_threshold=0.07,
        active_threshold=0.25,
    )
    if selected is None:
        return None

    bands = _part_no_barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(8, int(h * 0.06)),
        min_span_w=max(60, int(w * 0.18)),
        row_trans_threshold=0.035,
        active_threshold=0.20,
    )
    best_box = None
    best_score = -1
    selected_h = max(1, selected[3] - selected[1])
    for band in bands:
        box = band["box"]
        overlap = max(0, min(selected[3], box[3]) - max(selected[1], box[1]))
        band_h = max(1, box[3] - box[1])
        if overlap < max(4, int(min(selected_h, band_h) * 0.35)):
            continue
        score = overlap * 10 + band.get("span_w", 0)
        if score > best_score:
            best_score = score
            best_box = box
    return best_box or selected


def part_no_crop_has_text_above_barcode(crop_img):
    barcode_box = _part_no_primary_barcode_box(crop_img)
    if barcode_box is None:
        return False
    return _part_no_has_text_above_band(crop_img, barcode_box)


def _part_no_crop_text_neighbor_top(crop_img, barcode_box):
    if crop_img is None or barcode_box is None:
        return None
    h, w = crop_img.shape[:2]
    _x1, _y1, _x2, y2 = barcode_box
    barcode_h = max(1, y2 - _y1)
    start_y = min(h, int(y2 + max(4, barcode_h * 0.08)))
    if start_y >= h - 3:
        return None

    region = crop_img[start_y:, :]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    row_dark = dark.mean(axis=1)
    active_rows = row_dark > 0.018
    min_run = max(3, int(h * 0.018))
    start = None
    for i, active in enumerate(active_rows):
        if active and start is None:
            start = i
        at_end = i == len(active_rows) - 1
        if start is not None and ((not active) or at_end):
            end = i if not active else i + 1
            if end - start >= min_run:
                return start_y + start
            start = None
    return None


def trim_part_no_crop_before_lower_text_neighbor(crop_img):
    if crop_img is None:
        return None
    h, _w = crop_img.shape[:2]
    if h < 24:
        return crop_img
    barcode_box = _part_no_primary_barcode_box(crop_img)
    if barcode_box is None:
        return crop_img
    text_top = _part_no_crop_text_neighbor_top(crop_img, barcode_box)
    if text_top is None:
        return crop_img
    min_cutoff = barcode_box[3] + 2
    cutoff = text_top - max(2, int(h * 0.015))
    if cutoff < min_cutoff and min_cutoff <= text_top:
        cutoff = min_cutoff
    if barcode_box[3] < cutoff < h:
        trimmed = crop_img[:cutoff, :].copy()
        if part_no_crop_contains_1d_barcode(trimmed, min_span_ratio=0.16):
            return trimmed
    return crop_img


def part_no_crop_has_lower_neighbor_content(crop_img):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return False

    bands = _part_no_barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(8, int(h * 0.06)),
        min_span_w=max(60, int(w * 0.18)),
        row_trans_threshold=0.035,
        active_threshold=0.20,
    )
    if len(bands) >= 3:
        part_no_pair_bottom = bands[1]["box"][3] if len(bands) >= 2 else bands[0]["box"][3]
        min_gap = max(8, int(h * 0.035))
        for band in bands[2:]:
            if band["box"][1] - part_no_pair_bottom >= min_gap:
                return True
    elif len(bands) == 2:
        first = bands[0]["box"]
        second = bands[1]["box"]
        first_h = first[3] - first[1]
        if first_h >= max(36, int(h * 0.28)) and second[1] - first[3] >= max(8, int(h * 0.035)):
            return True
        if _part_no_two_band_has_lower_neighbor(crop_img, first, second):
            return True

    return False


def trim_part_no_crop_before_lower_neighbor(crop_img, trim_text_neighbor=False):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 24 or w < 80:
        return crop_img

    bands = _part_no_barcode_like_boxes_in_region(
        crop_img,
        0,
        0,
        w,
        h,
        min_run_h=max(8, int(h * 0.06)),
        min_span_w=max(60, int(w * 0.18)),
        row_trans_threshold=0.035,
        active_threshold=0.20,
    )
    if len(bands) < 3:
        if len(bands) == 2:
            first = bands[0]["box"]
            second = bands[1]["box"]
            first_h = first[3] - first[1]
            if first_h >= max(36, int(h * 0.28)) and second[1] - first[3] >= max(8, int(h * 0.035)):
                cutoff = max(first[3] + 2, second[1] - max(6, int(h * 0.025)))
                if first[3] + 2 <= cutoff < h:
                    return crop_img[:cutoff, :].copy()
            if _part_no_two_band_has_lower_neighbor(crop_img, first, second):
                cutoff = _part_no_cutoff_before_two_band_neighbor(crop_img, first, second)
                if first[3] + 2 <= cutoff < h:
                    return crop_img[:cutoff, :].copy()
        if trim_text_neighbor:
            return trim_part_no_crop_before_lower_text_neighbor(crop_img)
        return crop_img

    part_no_pair_bottom = bands[1]["box"][3]
    lower_top = bands[2]["box"][1]
    if lower_top - part_no_pair_bottom < max(8, int(h * 0.035)):
        if trim_text_neighbor:
            return trim_part_no_crop_before_lower_text_neighbor(crop_img)
        return crop_img

    cutoff = max(part_no_pair_bottom + 2, lower_top - max(6, int(h * 0.025)))
    min_h = max(24, part_no_pair_bottom + 2)
    if cutoff < min_h or cutoff >= h:
        if trim_text_neighbor:
            return trim_part_no_crop_before_lower_text_neighbor(crop_img)
        return crop_img
    return crop_img[:cutoff, :].copy()


def polish_part_no_crop_for_scan_miss(crop_img):
    if crop_img is None:
        return None
    trimmed = trim_part_no_crop_before_lower_neighbor(crop_img, trim_text_neighbor=True)
    return pad_part_no_crop_quiet_zone(trimmed)


def _white_band_is_clean(band):
    if band is None or band.size == 0:
        return True
    return (band[:, :, 0] < 128).mean() < 0.01


def _part_no_crop_embedded_prefix(original_crop, polished_crop):
    oh, ow = original_crop.shape[:2]
    ph, pw = polished_crop.shape[:2]
    if ph < 24 or pw < 80 or pw < ow:
        return None

    max_prefix_h = min(oh, ph)
    min_prefix_h = max(24, int(min(oh, ph) * 0.45))
    for x0 in range(0, pw - ow + 1):
        for prefix_h in range(max_prefix_h, min_prefix_h - 1, -1):
            if not np.array_equal(polished_crop[:prefix_h, x0 : x0 + ow], original_crop[:prefix_h, :]):
                continue
            left_ok = _white_band_is_clean(polished_crop[:prefix_h, :x0])
            right_ok = _white_band_is_clean(polished_crop[:prefix_h, x0 + ow :])
            bottom_ok = _white_band_is_clean(polished_crop[prefix_h:, :])
            if left_ok and right_ok and bottom_ok:
                return x0, prefix_h
    return None


def part_no_polished_crop_is_safe(original_crop, polished_crop):
    if original_crop is None or polished_crop is None:
        return False
    if polished_crop.shape[:2] == original_crop.shape[:2]:
        return False
    if part_no_crop_has_edge_clipped_barcode(original_crop):
        return False
    if part_no_crop_has_edge_clipped_barcode(polished_crop):
        return False
    if not part_no_crop_has_text_above_barcode(polished_crop):
        return False
    if part_no_crop_has_lower_neighbor_content(polished_crop):
        return False
    if not (
        part_no_crop_contains_1d_barcode(polished_crop, min_span_ratio=0.16)
        or part_no_crop_contains_1d_barcode(original_crop, min_span_ratio=0.16)
    ):
        return False
    return _part_no_crop_embedded_prefix(original_crop, polished_crop) is not None


def tighten_recovered_part_no_crop_width(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h < 24 or w < 160:
        return crop_img

    barcode_box = _part_no_primary_barcode_box(crop_img) or crop_1d_barcode_box(
        crop_img,
        min_span_ratio=0.12,
        row_trans_threshold=0.05,
        active_threshold=0.20,
    )
    if barcode_box is None:
        return crop_img

    _bx1, by1, _bx2, by2 = barcode_box
    band = crop_img[max(0, by1) : min(h, by2), :]
    if band.size == 0:
        return crop_img

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw_img > 0
    col_dark = dark.mean(axis=0)
    active_idx = np.where(col_dark > 0.18)[0]
    if active_idx.size == 0:
        return crop_img

    gap_threshold = max(12, int(w * 0.025), int((by2 - by1) * 0.75))
    runs = []
    start = int(active_idx[0])
    prev = int(active_idx[0])
    for idx_value in active_idx[1:]:
        idx = int(idx_value)
        if idx - prev > gap_threshold:
            runs.append((start, prev + 1))
            start = idx
        prev = idx
    runs.append((start, prev + 1))

    clusters = []
    for x1, x2 in runs:
        span = x2 - x1
        if span < max(80, int(w * 0.10)):
            continue
        active = col_dark[x1:x2] > 0.18
        transitions = int(np.count_nonzero(active[1:] != active[:-1])) if active.size > 1 else 0
        if transitions < 8:
            continue
        clusters.append((x1, x2, span, transitions))
    if not clusters:
        return crop_img

    selected = None
    for cluster in clusters:
        center_x = (cluster[0] + cluster[1]) / 2.0
        if center_x <= w * 0.72:
            selected = cluster
            break
    if selected is None:
        selected = clusters[0]

    x1, x2, span, _transitions = selected
    out_x1 = max(0, x1 - max(44, int(span * 0.24), int(w * 0.035)))
    out_x2 = min(w, x2 + max(28, int(span * 0.18), int(w * 0.025)))
    if out_x2 - out_x1 >= w * 0.92:
        return crop_img
    return crop_img[:, out_x1:out_x2].copy()
