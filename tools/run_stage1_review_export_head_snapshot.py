import argparse
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]


def _save_image(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    ext = ".jpg" if suffix in {".jpg", ".jpeg"} else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(buf.tobytes())


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


def _git_show(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path}"],
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
    )


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_head_snapshot(snapshot_root: Path):
    snapshot_root.mkdir(parents=True, exist_ok=True)
    crop_path = snapshot_root / "crop.py"
    local_yolo_path = snapshot_root / "local_yolo.py"
    crop_path.write_text(_git_show("crop.py"), encoding="utf-8")
    local_yolo_path.write_text(_git_show("local_yolo.py"), encoding="utf-8")

    local_yolo = _load_module_from_path("snapshot_local_yolo", local_yolo_path)
    local_yolo.DEFAULT_MODEL_SPECS = {
        "huawei-2ha7t/7": local_yolo.ModelSpec(
            path=str(REPO_ROOT / "local_models" / "detectors" / "label_detector.onnx"),
            names=("huawei_label",),
        ),
        "sn_model/9": local_yolo.ModelSpec(
            path=str(REPO_ROOT / "local_models" / "detectors" / "field_detector.onnx"),
            names=("model", "partno", "sn"),
        ),
    }
    sys.modules["local_yolo"] = local_yolo
    crop = _load_module_from_path("snapshot_crop", crop_path)
    return crop, local_yolo


def _box_from_pred_ratio(crop_mod, img, pred, pad_ratio, slant_guard_max_px=0):
    h, w = img.shape[:2]
    x = float(pred["x"])
    y = float(pred["y"])
    bw = float(pred["width"])
    bh = float(pred["height"])
    pad_w = bw * pad_ratio
    pad_h = bh * pad_ratio
    slant_guard = crop_mod.slant_guard_px(bw, slant_guard_max_px)

    x1 = int(x - bw / 2 - pad_w)
    y1 = int(y - bh / 2 - pad_h - slant_guard)
    x2 = int(x + bw / 2 + pad_w)
    y2 = int(y + bh / 2 + pad_h + slant_guard)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _collect_entries_from_preds(
    crop_mod,
    img,
    preds,
    allow_orientation_variants=False,
    require_field_evidence=False,
    max_crops=None,
):
    accepted = []
    dropped = 0
    for pred in preds:
        box = _box_from_pred_ratio(
            crop_mod,
            img,
            pred,
            crop_mod.PADDING_1,
            slant_guard_max_px=crop_mod.STAGE1_SLANT_GUARD_MAX_PX,
        )
        raw_crop = crop_mod.crop_from_box(img, box)
        if raw_crop is None:
            continue
        processed_crop = crop_mod._stage1_prepare_product_label_crop(
            raw_crop,
            allow_orientation_variants=allow_orientation_variants,
        )
        if processed_crop is None:
            dropped += 1
            continue
        if require_field_evidence and not crop_mod._stage1_has_fallback_field_evidence(processed_crop):
            dropped += 1
            continue
        accepted.append(
            {
                "confidence": float(pred.get("confidence", 1.0)),
                "class_id": int(pred.get("class_id", 0)),
                "box": box,
                "crop": processed_crop,
            }
        )

    if max_crops is not None and len(accepted) > max_crops:
        accepted = sorted(accepted, key=lambda item: item["confidence"], reverse=True)[:max_crops]
    return accepted, dropped


def _collect_entries_head(crop_mod, img, raw_preds):
    label_preds = [p for p in (raw_preds or []) if crop_mod.pred_class(p) == crop_mod.MODEL1_LABEL_CLASS]
    final_preds = crop_mod.nms(label_preds, crop_mod.MIN_CONF_1, crop_mod.NMS_1)
    entries, dropped = _collect_entries_from_preds(crop_mod, img, final_preds)
    if entries:
        return entries, dropped

    fallback_preds = crop_mod.nms(label_preds, crop_mod.MIN_CONF_1_FALLBACK, crop_mod.NMS_1)
    fallback_entries, fallback_dropped = _collect_entries_from_preds(
        crop_mod,
        img,
        fallback_preds,
        allow_orientation_variants=True,
        require_field_evidence=True,
        max_crops=crop_mod.STAGE1_FALLBACK_MAX_CROPS,
    )
    if fallback_entries:
        return fallback_entries, fallback_dropped
    return entries, dropped


def _detect_entries(crop_mod, img, img_path: str):
    preds = crop_mod.infer_with_resize(img, img_path, model_id=crop_mod.MODEL1_ID)
    entries, dropped = _collect_entries_head(crop_mod, img, preds)
    if entries or not crop_mod.stage1_rotation_retry_enabled():
        return entries, 0, dropped

    for rotation in crop_mod.stage1_rotation_retry_angles():
        rotated = crop_mod.stage1_rotated_image(img, rotation)
        rotated_preds = crop_mod.infer_with_resize(rotated, img_path, model_id=crop_mod.MODEL1_ID)
        rotated_entries, rotated_dropped = _collect_entries_head(crop_mod, rotated, rotated_preds)
        dropped += rotated_dropped
        if rotated_entries:
            mapped = []
            for entry in rotated_entries:
                mapped_entry = dict(entry)
                mapped_entry["box"] = _map_box_back_from_rotation(entry.get("box"), img.shape, rotation)
                mapped.append(mapped_entry)
            return mapped, rotation, dropped
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


def _build_contact_sheet(crop_mod, image_paths, out_path: Path, thumb_w=420, thumb_h=300, cols=3):
    if not image_paths:
        return
    thumbs = []
    for image_path in image_paths:
        img = crop_mod.read_image(str(image_path))
        if img is None:
            continue
        canvas = cv2.copyMakeBorder(img, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h, w = canvas.shape[:2]
        scale = min(thumb_w / float(max(1, w)), thumb_h / float(max(1, h)))
        resized = cv2.resize(canvas, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        import numpy as np
        thumb = np.full((thumb_h, thumb_w, 3), 255, dtype="uint8")
        y0 = (thumb_h - resized.shape[0]) // 2
        x0 = (thumb_w - resized.shape[1]) // 2
        thumb[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        label = Path(image_path).stem[:50]
        cv2.putText(thumb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
        thumbs.append(thumb)

    if not thumbs:
        return

    import numpy as np
    rows = int(math.ceil(len(thumbs) / float(cols)))
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype="uint8")
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y0 = row * thumb_h
        x0 = col * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
    _save_image(out_path, sheet)


def _write_readme(out_root: Path, total_images: int, total_boxes: int, input_root: Path):
    readme = out_root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "HEAD snapshot repro of 2026-06-24 stage1 review export",
                f"repo_root={REPO_ROOT}",
                f"source_images={total_images}",
                f"total_boxes={total_boxes}",
                f"input_root={input_root}",
            ]
        ),
        encoding="utf-8",
    )


def _list_images(input_root: Path):
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        str(path)
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )


def main():
    parser = argparse.ArgumentParser(description="Run stage1 review export using git HEAD snapshot modules.")
    parser.add_argument("--input", required=True, help="Input root directory")
    parser.add_argument("--out", required=True, help="Output root directory")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    out_root = Path(args.out).resolve()
    corrected_root = out_root / "corrected_inputs"
    overlays_root = out_root / "overlays"
    crops_root = out_root / "crops"
    summary_path = out_root / "summary.csv"
    contact_sheet_path = out_root / "overlay_contact_sheet.jpg"
    runtime_root = out_root / "_runtime"
    snapshot_root = out_root / "_head_snapshot"

    if out_root.exists():
        shutil.rmtree(out_root)
    corrected_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    os.environ["CROP_INFERENCE_BACKEND"] = "local"
    crop_mod, _local_yolo = _load_head_snapshot(snapshot_root)
    crop_mod.configure_paths(input_dir=str(input_root), out_dir=str(runtime_root))

    images = _list_images(input_root)
    overlay_paths = []
    summary_rows = []
    total_boxes = 0

    for index, img_path in enumerate(images, start=1):
        rel_path = os.path.relpath(img_path, str(input_root))
        print(f"[{index}/{len(images)}] {rel_path}")
        img = crop_mod.read_image(img_path)
        if img is None:
            print(f"  skip unreadable: {rel_path}")
            continue

        corrected_path = corrected_root / rel_path
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        _save_image(corrected_path, img)

        entries, rotation, dropped = _detect_entries(crop_mod, img, img_path)
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
            crop_img = entry.get("crop")
            if box is None or crop_img is None:
                continue
            crop_name = f"{rel_stub}__det{det_index:02d}.jpg"
            crop_path = crops_root / crop_name
            _save_image(crop_path, crop_img)
            x1, y1, x2, y2 = map(int, box)
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            summary_rows.append(
                [
                    rel_path.replace("\\", "/"),
                    det_index,
                    entry.get("class_id", 0),
                    f"{float(entry.get('confidence', 0.0)):.6f}",
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height,
                    str(crop_path),
                ]
            )
            total_boxes += 1

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
            ]
        )
        writer.writerows(summary_rows)

    _build_contact_sheet(crop_mod, overlay_paths, contact_sheet_path)
    _write_readme(out_root, len(images), total_boxes, input_root)
    print(f"images={len(images)} total_boxes={total_boxes}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
