import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(".png" if ext == ".png" else ".jpg", image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    buf.tofile(str(path))


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def main():
    parser = argparse.ArgumentParser(description="Batch predict Huawei labels with aspect filter.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="0")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--min-aspect", type=float, default=0.0)
    args = parser.parse_args()

    model = YOLO(args.model)
    input_root = Path(args.input)
    out_root = Path(args.out)
    predict_dir = out_root / "predict"
    crop_dir = out_root / "crops" / "huawei_label"
    labels_dir = out_root / "labels"
    predict_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    counts_rows = []
    summary = {}
    zero_detections = []
    total_detections = 0

    for image_path in iter_images(input_root):
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        image = imread_unicode(image_path)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")

        kept = []
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls != args.class_id:
                continue
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            width = x2 - x1
            height = y2 - y1
            aspect = width / max(height, 1e-6)
            if args.min_aspect > 0 and aspect < args.min_aspect:
                continue
            kept.append(
                {
                    "class_id": cls,
                    "class_name": result.names[cls],
                    "confidence": conf,
                    "aspect": aspect,
                    "xyxy": [x1, y1, x2, y2],
                }
            )

        kept.sort(key=lambda item: item["confidence"], reverse=True)
        total_detections += len(kept)
        rel_name = image_path.name
        summary[rel_name] = len(kept)
        counts_rows.append(f"{rel_name}\t{len(kept)}")
        if not kept:
            zero_detections.append(rel_name)

        label_lines = []
        annotated = image.copy()
        for idx, item in enumerate(kept, start=1):
            x1, y1, x2, y2 = item["xyxy"]
            width = x2 - x1
            height = y2 - y1
            xc = x1 + width / 2.0
            yc = y1 + height / 2.0
            h, w = image.shape[:2]
            label_lines.append(
                f"{args.class_id} {xc / w:.6f} {yc / h:.6f} {width / w:.6f} {height / h:.6f} {item['confidence']:.6f}"
            )

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
            text = f"{item['class_name']} {item['confidence']:.2f} a={item['aspect']:.2f}"
            cv2.putText(
                annotated,
                text,
                (int(x1), max(40, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3,
                cv2.LINE_AA,
            )

            x1i, y1i, x2i, y2i = map(int, (max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
            crop = image[y1i:y2i, x1i:x2i]
            if crop.size:
                crop_path = crop_dir / f"{image_path.stem}__det{idx:03d}{image_path.suffix}"
                imwrite_unicode(crop_path, crop)

        imwrite_unicode(predict_dir / rel_name, annotated)
        (labels_dir / f"{image_path.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")

    (out_root / "counts.tsv").write_text("\n".join(counts_rows) + "\n", encoding="utf-8")
    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "total_images": len(summary),
                "total_detections": total_detections,
                "zero_count": len(zero_detections),
                "counts": summary,
                "zero_detections": zero_detections,
                "model": args.model,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "class_id": args.class_id,
                "min_aspect": args.min_aspect,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_root / "zero_detections.txt").write_text("\n".join(zero_detections) + "\n", encoding="utf-8")

    print(f"TOTAL_IMAGES={len(summary)}")
    print(f"TOTAL_DETECTIONS={total_detections}")
    print(f"ZERO_COUNT={len(zero_detections)}")
    print(f"OUT_DIR={out_root}")


if __name__ == "__main__":
    main()
