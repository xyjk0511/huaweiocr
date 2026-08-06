import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import crop

DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "huawei_stage1_hardcases_20260622" / "manifest.csv"
DEFAULT_RAW_DIR = REPO_ROOT / "datasets" / "huawei_stage1_hardcases_20260622" / "raw_images"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "stage1_hardcase_diagnose_20260622"


def _pred_rect(pred):
    x = float(pred["x"])
    y = float(pred["y"])
    w = float(pred["width"])
    h = float(pred["height"])
    x1 = max(0, int(round(x - w / 2.0)))
    y1 = max(0, int(round(y - h / 2.0)))
    x2 = max(x1 + 1, int(round(x + w / 2.0)))
    y2 = max(y1 + 1, int(round(y + h / 2.0)))
    return x1, y1, x2, y2


def _analyze_pred(img, pred):
    details = {
        "confidence": round(float(pred.get("confidence", 0.0)), 6),
        "class": crop.pred_class(pred),
    }
    pred_crop = crop.crop_from_pred(
        img,
        pred,
        crop.PADDING_1,
        slant_guard_max_px=crop.STAGE1_SLANT_GUARD_MAX_PX,
    )
    if pred_crop is None:
        details["status"] = "rejected"
        details["reason"] = "crop_from_pred_none"
        return details, None, None

    details["candidate_ok"] = bool(crop.stage1_is_product_label_candidate_crop(pred_crop))
    if not details["candidate_ok"]:
        details["status"] = "rejected"
        details["reason"] = "candidate_filter"
        return details, pred_crop, None

    tightened = crop.stage1_tighten_label_crop(pred_crop)
    details["tightened"] = tightened is not None
    if tightened is not None and crop.stage1_is_product_label_crop(tightened):
        details["status"] = "accepted"
        details["reason"] = "tightened_pass"
        return details, pred_crop, tightened

    direct_ok = bool(crop.stage1_is_product_label_crop(pred_crop))
    details["direct_ok"] = direct_ok
    if direct_ok:
        details["status"] = "accepted"
        details["reason"] = "direct_pass"
        return details, pred_crop, pred_crop

    details["status"] = "rejected"
    details["reason"] = "product_filter"
    return details, pred_crop, tightened


def _save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    path.write_bytes(buf.tobytes())


def _draw_preds(img, all_preds, nms_preds, accepted_rects):
    out = img.copy()
    nms_keys = {id(pred): True for pred in nms_preds}
    for pred in all_preds:
        x1, y1, x2, y2 = _pred_rect(pred)
        color = (0, 165, 255)
        thickness = 2
        if id(pred) in nms_keys:
            color = (255, 0, 0)
            thickness = 3
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{float(pred.get('confidence', 0.0)):.2f}"
        cv2.putText(out, label, (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in accepted_rects:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description="Diagnose Stage1 hardcases with visual evidence.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    os.environ.setdefault("CROP_INFERENCE_BACKEND", "local")
    crop.configure_paths(input_dir=args.raw_dir, out_dir=args.out)

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        filename = row["filename"]
        img_path = raw_dir / filename
        img = crop.read_image(str(img_path))
        preds = crop.infer_with_resize(img, str(img_path), model_id=crop.MODEL1_ID)
        label_preds = [p for p in preds if crop.pred_class(p) == crop.MODEL1_LABEL_CLASS]
        nms_preds = crop.nms(label_preds, crop.MIN_CONF_1, crop.NMS_1)

        item_dir = out_dir / Path(filename).stem
        accepted_rects = []
        pred_records = []
        for idx, pred in enumerate(nms_preds, start=1):
            details, raw_crop, final_crop = _analyze_pred(img, pred)
            rect = _pred_rect(pred)
            details["rect"] = rect
            pred_records.append(details)
            accepted = details["status"] == "accepted"
            if accepted:
                accepted_rects.append(rect)
            if raw_crop is not None:
                _save_image(item_dir / "nms_crops" / f"{idx:02d}_{details['status']}_{details['reason']}_raw.png", raw_crop)
            if final_crop is not None:
                _save_image(item_dir / "nms_crops" / f"{idx:02d}_{details['status']}_{details['reason']}_final.png", final_crop)

        annotated = _draw_preds(img, label_preds, nms_preds, accepted_rects)
        _save_image(item_dir / "annotated.png", annotated)

        summary = {
            "filename": filename,
            "source_path": str(img_path),
            "current_stage1_count": int(row["current_stage1_count"]),
            "target_label_count": int(row["target_label_count"]),
            "raw_label_preds": len(label_preds),
            "nms_preds": len(nms_preds),
            "accepted": len(accepted_rects),
            "note": row.get("note", ""),
            "preds": pred_records,
        }
        (item_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(summary)

    (out_dir / "summary_all.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
