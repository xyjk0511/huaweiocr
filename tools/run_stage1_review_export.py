import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import crop
from tools.stage1_review_images import (
    export_review,
    map_box_back_from_rotation as _map_box_back_from_rotation,
    prepare_output,
)


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


def main():
    parser = argparse.ArgumentParser(description="Batch review exporter for stage1 label detection.")
    parser.add_argument("--input", required=True, help="Input root directory")
    parser.add_argument("--out", required=True, help="Output root directory")
    parser.add_argument("--conf", type=float, default=crop.MIN_CONF_1, help="Hard confidence threshold for exported detections")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    out_root = Path(args.out).resolve()
    try:
        prepare_output(input_root, out_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    os.environ.setdefault("CROP_INFERENCE_BACKEND", "local")
    crop.configure_paths(input_dir=str(input_root), out_dir=str(out_root / "_runtime"))

    images = crop.list_images(str(input_root))
    export_review(
        images, input_root, out_root, crop.read_image,
        lambda img, path: _detect_entries(img, path, args.conf),
        lambda img, entry: crop.crop_from_box(img, entry["box"]),
    )


if __name__ == "__main__":
    main()
