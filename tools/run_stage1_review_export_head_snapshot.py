import argparse
import csv
import importlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stage1_review_images import (
    build_contact_sheet as _build_contact_sheet,
    draw_overlay as _draw_overlay,
    map_box_back_from_rotation as _map_box_back_from_rotation,
    safe_rel_name as _safe_rel_name,
    save_image as _save_image,
)


def _load_head_snapshot(snapshot_root: Path):
    snapshot_root.mkdir(parents=True, exist_ok=True)
    git_options = {"cwd": str(REPO_ROOT), "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    root_names = subprocess.check_output(["git", "ls-tree", "--name-only", "HEAD"], **git_options).decode("utf-8").splitlines()
    paths = [name for name in root_names if name.endswith(".py") or name == "huaweiocr"]
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD", *paths], **git_options)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            if member.isfile() and member.name.endswith(".py"):
                path = snapshot_root / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.extractfile(member).read())

    # HEAD includes both legacy entry points and their package implementations.
    # Discard already imported runtime modules so none come from the working tree.
    for name in list(sys.modules):
        if name == "huaweiocr" or name.startswith("huaweiocr.") or name + ".py" in paths:
            sys.modules.pop(name, None)
    sys.path.insert(0, str(snapshot_root))
    local_yolo = importlib.import_module("local_yolo")
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
    return importlib.import_module("crop"), local_yolo


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

    _build_contact_sheet(overlay_paths, contact_sheet_path, crop_mod.read_image)
    _write_readme(out_root, len(images), total_boxes, input_root)
    print(f"images={len(images)} total_boxes={total_boxes}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
