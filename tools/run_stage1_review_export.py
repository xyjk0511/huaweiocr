import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import crop


def _save_image(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    ext = ".jpg" if suffix in {".jpg", ".jpeg"} else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(buf.tobytes())


def _pred_box(pred):
    x1, y1, w, h = crop.to_xywh_topleft(pred)
    return x1, y1, x1 + w, y1 + h


def _safe_rel_name(rel_path: str) -> str:
    rel_no_ext = os.path.splitext(rel_path)[0]
    return rel_no_ext.replace("/", "__").replace("\\", "__")


def _map_box_back_from_rotation(box, original_shape, rotation):
    if box is None or rotation not in {90, 180, 270}:
        return box
    h, w = original_shape[:2]
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    def inverse_point(xr, yr):
        if rotation == 90:
            return yr, h - xr
        if rotation == 180:
            return w - xr, h - yr
        return w - yr, xr

    mapped = [inverse_point(xr, yr) for xr, yr in corners]
    xs = [pt[0] for pt in mapped]
    ys = [pt[1] for pt in mapped]
    ox1 = max(0, min(w, int(round(min(xs)))))
    oy1 = max(0, min(h, int(round(min(ys)))))
    ox2 = max(0, min(w, int(round(max(xs)))))
    oy2 = max(0, min(h, int(round(max(ys)))))
    if ox2 <= ox1 or oy2 <= oy1:
        return None
    return ox1, oy1, ox2, oy2


def _collect_entries_current_logic(img, preds, min_conf):
    label_preds = [p for p in (preds or []) if crop.pred_class(p) == crop.MODEL1_LABEL_CLASS]
    hard_threshold_mode = abs(float(min_conf) - float(crop.MIN_CONF_1)) > 1e-9
    final_preds = crop.nms(label_preds, min_conf, crop.NMS_1)

    accepted_entries, dropped = crop._stage1_collect_product_label_entries(img, final_preds)
    if accepted_entries:
        existing_boxes = [item["box"] for item in accepted_entries if item.get("box") is not None]
        if not hard_threshold_mode:
            supplement_entries = []
            low_conf_label_preds = sorted(
                (
                    p
                    for p in label_preds
                    if float(p.get("confidence", 0.0)) >= crop.STAGE1_SUPPLEMENT_MIN_CONF
                ),
                key=lambda p: float(p.get("confidence", 0.0)),
                reverse=True,
            )
            for pred in low_conf_label_preds:
                conf = float(pred.get("confidence", 0.0))
                if conf >= min_conf:
                    continue
                box = crop.box_from_pred(img, pred, crop.PADDING_1, slant_guard_max_px=crop.STAGE1_SLANT_GUARD_MAX_PX)
                if crop._stage1_box_conflicts_with_existing(box, existing_boxes):
                    continue
                supplement_candidate_entries, supplement_dropped = crop._stage1_collect_product_label_entries(
                    img,
                    [pred],
                    allow_orientation_variants=True,
                    max_crops=1,
                )
                dropped += supplement_dropped
                if not supplement_candidate_entries:
                    continue
                supplement_entry = supplement_candidate_entries[0]
                supplement_entries.append(supplement_entry)
                if supplement_entry.get("box") is not None:
                    existing_boxes.append(supplement_entry["box"])
                if len(supplement_entries) >= crop.STAGE1_FALLBACK_MAX_CROPS:
                    break
            if supplement_entries:
                accepted_entries.extend(supplement_entries)
        secondary_entries, secondary_dropped = crop._stage1_collect_secondary_model_entries(img, existing_boxes)
        dropped += secondary_dropped
        if secondary_entries:
            if hard_threshold_mode:
                secondary_entries = [
                    entry for entry in secondary_entries if float(entry.get("confidence", 0.0)) >= min_conf
                ]
            accepted_entries.extend(secondary_entries)
        return accepted_entries, dropped

    fallback_min_conf = min_conf if hard_threshold_mode else crop.MIN_CONF_1_FALLBACK
    fallback_preds = crop.nms(label_preds, fallback_min_conf, crop.NMS_1)
    fallback_entries, fallback_dropped = crop._stage1_collect_product_label_entries(
        img,
        fallback_preds,
        allow_orientation_variants=True,
        max_crops=crop.STAGE1_FALLBACK_MAX_CROPS,
    )
    if fallback_entries:
        return fallback_entries, fallback_dropped
    return [], dropped


def _detect_entries(img, img_path: str, min_conf):
    preds = crop.infer_with_resize(img, img_path, model_id=crop.MODEL1_ID)
    entries, dropped = _collect_entries_current_logic(img, preds, min_conf)
    if entries or not crop.stage1_rotation_retry_enabled():
        return entries, 0, dropped

    for rotation in crop.stage1_rotation_retry_angles():
        rotated = crop.stage1_rotated_image(img, rotation)
        rotated_preds = crop.infer_with_resize(rotated, img_path, model_id=crop.MODEL1_ID)
        rotated_entries, rotated_dropped = _collect_entries_current_logic(rotated, rotated_preds, min_conf)
        dropped += rotated_dropped
        if rotated_entries:
            mapped_entries = []
            for entry in rotated_entries:
                mapped_entry = dict(entry)
                mapped_entry["box"] = _map_box_back_from_rotation(entry.get("box"), img.shape, rotation)
                mapped_entries.append(mapped_entry)
            return mapped_entries, rotation, dropped
    return [], 0, dropped


def _draw_overlay(img, entries):
    overlay = img.copy()
    for idx, entry in enumerate(entries, start=1):
        box = entry.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
        text = f"det{idx}:{float(entry.get('confidence', 0.0)):.3f}"
        cv2.putText(
            overlay,
            text,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _build_contact_sheet(image_paths, out_path: Path, thumb_w=420, thumb_h=300, cols=3):
    if not image_paths:
        return
    thumbs = []
    for image_path in image_paths:
        img = crop.read_image(str(image_path))
        if img is None:
            continue
        canvas = cv2.copyMakeBorder(img, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h, w = canvas.shape[:2]
        scale = min(thumb_w / float(max(1, w)), thumb_h / float(max(1, h)))
        resized = cv2.resize(canvas, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        thumb = 255 * (cv2.UMat(thumb_h, thumb_w, cv2.CV_8UC3).get() if hasattr(cv2, "UMat") else None)
        if thumb is None:
            import numpy as np

            thumb = np.full((thumb_h, thumb_w, 3), 255, dtype="uint8")
        y0 = (thumb_h - resized.shape[0]) // 2
        x0 = (thumb_w - resized.shape[1]) // 2
        thumb[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        label = Path(image_path).stem[:50]
        cv2.putText(thumb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
        thumbs.append(thumb)

    if not thumbs:
        return

    rows = int(math.ceil(len(thumbs) / float(cols)))
    import numpy as np

    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype="uint8")
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y0 = row * thumb_h
        x0 = col * thumb_w
        sheet[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
    _save_image(out_path, sheet)


def main():
    parser = argparse.ArgumentParser(description="Batch review exporter for stage1 label detection.")
    parser.add_argument("--input", required=True, help="Input root directory")
    parser.add_argument("--out", required=True, help="Output root directory")
    parser.add_argument("--conf", type=float, default=crop.MIN_CONF_1, help="Hard confidence threshold for exported detections")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    out_root = Path(args.out).resolve()
    min_conf = float(args.conf)
    corrected_root = out_root / "corrected_inputs"
    overlays_root = out_root / "overlays"
    crops_root = out_root / "crops"
    summary_path = out_root / "summary.csv"
    contact_sheet_path = out_root / "overlay_contact_sheet.jpg"

    if out_root.exists():
        shutil.rmtree(out_root)
    corrected_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CROP_INFERENCE_BACKEND", "local")
    crop.configure_paths(input_dir=str(input_root), out_dir=str(out_root / "_runtime"))

    images = crop.list_images(str(input_root))
    overlay_paths = []
    summary_rows = []
    total_boxes = 0

    for index, img_path in enumerate(images, start=1):
        rel_path = os.path.relpath(img_path, str(input_root))
        print(f"[{index}/{len(images)}] {rel_path}")
        img = crop.read_image(img_path)
        if img is None:
            print(f"  skip unreadable: {rel_path}")
            continue

        corrected_path = corrected_root / rel_path
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        _save_image(corrected_path, img)

        entries, rotation, dropped = _detect_entries(img, img_path, min_conf)
        if rotation:
            print(f"  rotation_retry={rotation} boxes={len(entries)} dropped={dropped}")

        overlay = _draw_overlay(img, entries)
        overlay_path = overlays_root / rel_path
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        _save_image(overlay_path, overlay)
        overlay_paths.append(overlay_path)

        rel_stub = _safe_rel_name(rel_path)
        for det_index, entry in enumerate(entries, start=1):
            box = entry.get("box")
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop_img = crop.crop_from_box(img, box)
            if crop_img is None:
                continue
            crop_name = f"{rel_stub}__det{det_index:02d}.jpg"
            crop_path = crops_root / crop_name
            _save_image(crop_path, crop_img)
            total_boxes += 1
            summary_rows.append(
                {
                    "relative_image": rel_path.replace("\\", "/"),
                    "det_index": det_index,
                    "class_id": 0,
                    "confidence": f"{float(entry.get('confidence', 0.0)):.6f}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "crop_path": str(crop_path),
                }
            )

    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_image",
                "det_index",
                "class_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "width",
                "height",
                "crop_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    _build_contact_sheet(overlay_paths, contact_sheet_path)
    print(f"images={len(images)} boxes={total_boxes} out={out_root}")


if __name__ == "__main__":
    main()
