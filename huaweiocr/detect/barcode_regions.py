"""Pixel analysis of label-local barcode regions, independent of pipeline state."""

import cv2
import numpy as np

from huaweiocr.core.geometry import crop_from_box

def barcode_like_box_in_region(
    img,
    sx1,
    sy1,
    sx2,
    sy2,
    min_run_h,
    min_span_w,
    row_trans_threshold=0.07,
    active_threshold=0.35,
):
    H, W = img.shape[:2]
    sx1 = max(0, min(W, int(sx1)))
    sx2 = max(0, min(W, int(sx2)))
    sy1 = max(0, min(H, int(sy1)))
    sy2 = max(0, min(H, int(sy2)))
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    roi = img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    if dark.shape[0] < 8 or dark.shape[1] < 80:
        return None

    row_dark = dark.mean(axis=1)
    row_trans = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1) / max(1, dark.shape[1] - 1)
    candidate_rows = (row_dark > 0.08) & (row_dark < 0.75) & (row_trans > row_trans_threshold)

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
                active = col_dark > active_threshold
                active_idx = np.where(active)[0]
                if active_idx.size:
                    span_w = int(active_idx[-1] - active_idx[0] + 1)
                    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
                    aspect = span_w / float(max(run_h, 1))
                    if span_w >= min_span_w and transitions >= 18 and aspect >= 3.0:
                        score = span_w * run_h
                        if score > best_score:
                            pad_x = max(8, int(span_w * 0.04))
                            pad_y = max(4, int(run_h * 0.25))
                            best = (
                                max(0, sx1 + int(active_idx[0]) - pad_x),
                                max(0, sy1 + start - pad_y),
                                min(W, sx1 + int(active_idx[-1]) + 1 + pad_x),
                                min(H, sy1 + end + pad_y),
                            )
                            best_score = score
            start = None

    return best


def _barcode_like_boxes_in_region(
    img,
    sx1,
    sy1,
    sx2,
    sy2,
    min_run_h,
    min_span_w,
    row_trans_threshold=0.07,
    active_threshold=0.35,
):
    H, W = img.shape[:2]
    sx1 = max(0, min(W, int(sx1)))
    sx2 = max(0, min(W, int(sx2)))
    sy1 = max(0, min(H, int(sy1)))
    sy2 = max(0, min(H, int(sy2)))
    if sx2 <= sx1 or sy2 <= sy1:
        return []

    roi = img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    if dark.shape[0] < 8 or dark.shape[1] < 80:
        return []

    return _barcode_like_boxes_from_mask(
        dark,
        sx1,
        sy1,
        W,
        H,
        min_run_h=min_run_h,
        min_span_w=min_span_w,
        row_trans_threshold=row_trans_threshold,
        active_threshold=active_threshold,
    )


def _barcode_like_boxes_from_mask(
    dark,
    sx1,
    sy1,
    image_w,
    image_h,
    min_run_h,
    min_span_w,
    row_trans_threshold=0.07,
    active_threshold=0.35,
):
    row_dark = dark.mean(axis=1)
    row_trans = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1) / max(1, dark.shape[1] - 1)
    candidate_rows = (row_dark > 0.08) & (row_dark < 0.75) & (row_trans > row_trans_threshold)

    candidates = []
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
                active = col_dark > active_threshold
                active_idx = np.where(active)[0]
                if active_idx.size:
                    span_w = int(active_idx[-1] - active_idx[0] + 1)
                    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
                    aspect = span_w / float(max(run_h, 1))
                    if span_w >= min_span_w and transitions >= 18 and aspect >= 3.0:
                        score = span_w * run_h
                        pad_x = max(8, int(span_w * 0.04))
                        pad_y = max(4, int(run_h * 0.25))
                        candidates.append(
                            (
                                (
                                    max(0, sx1 + int(active_idx[0]) - pad_x),
                                    max(0, sy1 + start - pad_y),
                                    min(image_w, sx1 + int(active_idx[-1]) + 1 + pad_x),
                                    min(image_h, sy1 + end + pad_y),
                                ),
                                score,
                            )
                        )
            start = None

    return candidates

def _crop_has_sn_barcode_stripes(crop_img):
    if crop_img is None:
        return False
    h, w = crop_img.shape[:2]
    if h < 24 or w < 100:
        return False
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0

    min_vertical_run = max(18, int(h * 0.16))
    stripe = np.zeros(w, dtype=bool)
    for x in range(w):
        best = 0
        current = 0
        for y in range(h):
            if dark[y, x]:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        stripe[x] = best >= min_vertical_run

    active_idx = np.where(stripe)[0]
    if active_idx.size == 0:
        return False

    runs = []
    start = None
    for i, value in enumerate(stripe):
        if value and start is None:
            start = i
        at_end = i == len(stripe) - 1
        if start is not None and ((not value) or at_end):
            end = i if not value else i + 1
            runs.append((start, end))
            start = None

    widths = [end - start for start, end in runs]
    span_w = int(active_idx[-1] - active_idx[0] + 1)
    transitions = int(np.count_nonzero(stripe[1:] != stripe[:-1]))
    median_width = float(np.median(widths)) if widths else 0.0
    return (
        span_w >= max(85, int(w * 0.20))
        and transitions >= 30
        and len(runs) >= 15
        and median_width <= 12.0
    )


def _part_no_barcode_like_boxes_in_region(
    img,
    sx1,
    sy1,
    sx2,
    sy2,
    min_run_h,
    min_span_w,
    row_trans_threshold=0.035,
    active_threshold=0.20,
):
    H, W = img.shape[:2]
    sx1 = max(0, min(W, int(sx1)))
    sx2 = max(0, min(W, int(sx2)))
    sy1 = max(0, min(H, int(sy1)))
    sy2 = max(0, min(H, int(sy2)))
    if sx2 <= sx1 or sy2 <= sy1:
        return []

    roi = img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0
    if dark.shape[0] < 8 or dark.shape[1] < 80:
        return []

    row_dark = dark.mean(axis=1)
    row_trans = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1) / max(1, dark.shape[1] - 1)
    candidate_rows = (row_dark > 0.08) & (row_dark < 0.75) & (row_trans > row_trans_threshold)

    boxes = []
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
                active = col_dark > active_threshold
                active_idx = np.where(active)[0]
                if active_idx.size:
                    span_w = int(active_idx[-1] - active_idx[0] + 1)
                    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
                    aspect = span_w / float(max(run_h, 1))
                    if span_w >= min_span_w and transitions >= 18 and aspect >= 3.0:
                        boxes.append(
                            {
                                "box": (
                                    sx1 + int(active_idx[0]),
                                    sy1 + start,
                                    sx1 + int(active_idx[-1]) + 1,
                                    sy1 + end,
                                ),
                                "run_h": run_h,
                                "span_w": span_w,
                                "transitions": transitions,
                                "aspect": aspect,
                            }
                        )
            start = None

    return sorted(boxes, key=lambda item: (item["box"][1], -item["span_w"]))


def split_sn_barcode_segments(img, box):
    crop_img = crop_from_box(img, box)
    if crop_img is None:
        return []
    h, w = crop_img.shape[:2]
    if h < 18 or w < 80:
        return []
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = bw > 0

    min_vertical_run = max(16, int(h * 0.22))
    stripe = np.zeros(w, dtype=bool)
    for x in range(w):
        best = 0
        current = 0
        for y in range(h):
            if dark[y, x]:
                current += 1
                best = max(best, current)
            else:
                current = 0
        stripe[x] = best >= min_vertical_run

    active_idx = np.where(stripe)[0]
    if active_idx.size == 0:
        return []

    gap_threshold = max(18, int(w * 0.035))
    segments = []
    start = int(active_idx[0])
    previous = int(active_idx[0])
    for value in active_idx[1:]:
        value = int(value)
        if value - previous > gap_threshold:
            segments.append((start, previous + 1))
            start = value
        previous = value
    segments.append((start, previous + 1))

    x1, y1, x2, y2 = box
    out = []
    for seg_x1, seg_x2 in segments:
        seg_w = seg_x2 - seg_x1
        if seg_w < max(80, int(w * 0.12)):
            continue
        seg_stripe = stripe[seg_x1:seg_x2]
        transitions = int(np.count_nonzero(seg_stripe[1:] != seg_stripe[:-1])) if len(seg_stripe) > 1 else 0
        if transitions < 18:
            continue
        pad_x = max(10, int(seg_w * 0.08))
        out.append(
            (
                max(0, x1 + seg_x1 - pad_x),
                y1,
                min(img.shape[1], x1 + seg_x2 + pad_x),
                y2,
            )
        )
    return out or [box]


def normalize_sn_crop_for_barcode(crop_img):
    if crop_img is None:
        return None
    h, w = crop_img.shape[:2]
    if h <= 0 or w <= 0:
        return crop_img
    resized = cv2.resize(crop_img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(resized, 1.7, blurred, -0.7, 0)
    if sharpened.shape[1] >= 1300:
        return sharpened
    padded = cv2.copyMakeBorder(
        sharpened,
        48,
        48,
        48,
        48,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    blurred = cv2.GaussianBlur(padded, (0, 0), 1.1)
    return cv2.addWeighted(padded, 1.9, blurred, -0.9, 0)
