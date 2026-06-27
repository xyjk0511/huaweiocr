import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import crop


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def read_image_unicode(path: Path):
    if not path or not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    buf.tofile(str(path))


def fit_on_canvas(image, width, height, bg=245):
    if image is None or image.size == 0:
        return np.full((height, width, 3), bg, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def make_banner(text, width, height=34):
    banner = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        banner,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return banner


def make_stage1_tile(source_path: Path, label_paths, cell_width=280, cell_height=180):
    cells = [fit_on_canvas(read_image_unicode(source_path), cell_width, cell_height, bg=250)]
    for label_path in label_paths[:3]:
        cells.append(fit_on_canvas(read_image_unicode(Path(label_path)), cell_width, cell_height))
    while len(cells) < 4:
        cells.append(np.full((cell_height, cell_width, 3), 255, dtype=np.uint8))
    body = np.hstack(cells)
    title = f"{source_path.name} | stage1_labels={len(label_paths)}"
    return np.vstack([make_banner(title, body.shape[1]), body])


def make_stage2_tile(row, cell_width=280, cell_height=170):
    label_path = Path(row["label_crop"])
    sn_path = Path(row["sn_path"]) if row.get("sn_path") else None
    partno_path = Path(row["part_no_path"]) if row.get("part_no_path") else None
    model_path = Path(row["model_path"]) if row.get("model_path") else None
    cells = [
        fit_on_canvas(read_image_unicode(label_path), cell_width, cell_height, bg=250),
        fit_on_canvas(read_image_unicode(sn_path), cell_width, cell_height),
        fit_on_canvas(read_image_unicode(partno_path), cell_width, cell_height),
        fit_on_canvas(read_image_unicode(model_path), cell_width, cell_height),
    ]
    body = np.hstack(cells)
    title = (
        f"{label_path.name} | sn={int(bool(row.get('sn_path')))} "
        f"pn={int(bool(row.get('part_no_path')))} model={int(bool(row.get('model_path')))} "
        f"rot={row.get('stage2_rotation', 0)} src={row.get('part_no_crop_source') or 'none'}"
    )
    part_no_codes = list(row.get("part_no_codes") or [])
    if part_no_codes:
        title += f" code={part_no_codes[0][:20]}"
    return np.vstack([make_banner(title, body.shape[1]), body])


def make_grid(tiles, cols):
    if not tiles:
        return None
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    padded = []
    for tile in tiles:
        canvas = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        canvas[: tile.shape[0], : tile.shape[1]] = tile
        padded.append(canvas)
    rows = []
    for idx in range(0, len(padded), cols):
        row = padded[idx : idx + cols]
        while len(row) < cols:
            row.append(np.full((tile_h, tile_w, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def read_manifest(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_summary_markdown(path: Path, summary):
    stage1 = summary["stage1"]
    stage2 = summary["stage2"]
    lines = [
        "# Stage1 / Stage2 Audit",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- out_dir: `{summary['out_dir']}`",
        f"- input_images: `{summary['input_images']}`",
        f"- stage1 labels: `{stage1['label_count']}`",
        f"- stage1 zero-label images: `{stage1['zero_label_images']}`",
        f"- stage2 manifest rows: `{stage2['manifest_rows']}`",
        f"- stage2 sn: `{stage2['has_sn']}` / `{stage2['label_count']}`",
        f"- stage2 partno: `{stage2['has_part_no']}` / `{stage2['label_count']}`",
        f"- stage2 both(sn+partno): `{stage2['has_both']}` / `{stage2['label_count']}`",
        f"- stage2 total-miss labels: `{stage2['missing_all']}`",
        "",
        "## Files",
        "",
        "- `summary.json`: machine-readable summary",
        "- `stage1_preview.png`: stage1 sample grid",
        "- `stage2_success_preview.png`: stage2 success grid",
        "- `stage2_missing_preview.png`: stage2 missing grid when needed",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_out_dir(repo_root: Path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "runs" / f"stage1_stage2_audit_{stamp}"


def main():
    parser = argparse.ArgumentParser(description="Run and organize HuaweiOCR stage1/stage2 audit outputs.")
    parser.add_argument("--input", default="new_images", help="Input directory of source images.")
    parser.add_argument("--out", default="", help="Audit output root. Default: runs/stage1_stage2_audit_<timestamp>")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warn", "error"])
    parser.add_argument("--preview-count", type=int, default=8, help="How many cases to include in preview grids.")
    parser.add_argument("--clean", action="store_true", help="Clean the output root if it already exists.")
    parser.add_argument("--save-model", action="store_true", help="Also save model crops in stage2.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = (repo_root / input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    out_dir = Path(args.out) if args.out else default_out_dir(repo_root)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    os.environ["CROP_STAGE2_SAVE_MODEL"] = "1" if args.save_model else "0"
    result = crop.main(
        input_dir=str(input_dir),
        out_dir=str(out_dir),
        log_level=args.log_level,
        clean=args.clean,
    )

    stage1_dir = Path(result["stage1_dir"])
    stage2_dir = Path(result["stage2_dir"])
    manifest_path = Path(result["manifest_path"])
    manifest_rows = read_manifest(manifest_path)

    input_images = list(iter_images(input_dir))
    stage1_by_source = defaultdict(list)
    for label_path in sorted(stage1_dir.glob("*.png")):
        source_name = label_path.name.split("__label_", 1)[0]
        stage1_by_source[source_name].append(str(label_path))

    stage1_zero = []
    for image_path in input_images:
        if not stage1_by_source.get(image_path.name):
            stage1_zero.append(image_path.name)

    manifest_by_label = {row["label_id"]: row for row in manifest_rows}
    stage1_label_files = sorted(stage1_dir.glob("*.png"))
    missing_all_labels = []
    for label_path in stage1_label_files:
        if label_path.stem not in manifest_by_label:
            missing_all_labels.append(label_path.name)

    missing_sn_rows = [row for row in manifest_rows if not row.get("sn_path")]
    missing_partno_rows = [row for row in manifest_rows if not row.get("part_no_path")]
    both_rows = [row for row in manifest_rows if row.get("sn_path") and row.get("part_no_path")]
    model_rows = [row for row in manifest_rows if row.get("model_path")]

    summary = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "input_images": len(input_images),
        "stage1": {
            "label_count": len(stage1_label_files),
            "images_with_labels": len(input_images) - len(stage1_zero),
            "zero_label_images": len(stage1_zero),
            "zero_label_image_names": stage1_zero,
            "max_labels_per_image": max((len(v) for v in stage1_by_source.values()), default=0),
        },
        "stage2": {
            "label_count": len(stage1_label_files),
            "manifest_rows": len(manifest_rows),
            "has_sn": sum(1 for row in manifest_rows if row.get("sn_path")),
            "has_part_no": sum(1 for row in manifest_rows if row.get("part_no_path")),
            "has_model": len(model_rows),
            "has_both": len(both_rows),
            "missing_sn": len(missing_sn_rows),
            "missing_part_no": len(missing_partno_rows),
            "missing_all": len(missing_all_labels),
            "missing_all_label_names": missing_all_labels,
        },
        "paths": {
            "stage1_dir": str(stage1_dir),
            "stage2_dir": str(stage2_dir),
            "manifest_path": str(manifest_path),
        },
    }

    stage1_preview_cases = input_images[: args.preview_count]
    stage1_tiles = [
        make_stage1_tile(image_path, stage1_by_source.get(image_path.name, []))
        for image_path in stage1_preview_cases
    ]
    stage1_grid = make_grid(stage1_tiles, cols=1)
    if stage1_grid is not None:
        write_image_unicode(out_dir / "stage1_preview.png", stage1_grid)

    success_tiles = [make_stage2_tile(row) for row in both_rows[: args.preview_count]]
    success_grid = make_grid(success_tiles, cols=1)
    if success_grid is not None:
        write_image_unicode(out_dir / "stage2_success_preview.png", success_grid)

    missing_tiles = []
    missing_tiles.extend(make_stage2_tile(row) for row in missing_partno_rows[: args.preview_count])
    missing_tiles.extend(make_stage2_tile(row) for row in missing_sn_rows[: args.preview_count])
    if missing_all_labels:
        for label_name in missing_all_labels[: args.preview_count]:
            label_path = stage1_dir / label_name
            missing_tiles.append(
                make_stage2_tile(
                    {
                        "label_crop": str(label_path),
                        "sn_path": None,
                        "part_no_path": None,
                        "model_path": None,
                        "stage2_rotation": 0,
                        "part_no_crop_source": "",
                        "part_no_codes": [],
                    }
                )
            )
    missing_grid = make_grid(missing_tiles, cols=1)
    if missing_grid is not None:
        write_image_unicode(out_dir / "stage2_missing_preview.png", missing_grid)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(out_dir / "README.audit.md", summary)

    print(f"INPUT_DIR={input_dir}")
    print(f"OUT_DIR={out_dir}")
    print(f"STAGE1_LABELS={summary['stage1']['label_count']}")
    print(f"STAGE2_MANIFEST_ROWS={summary['stage2']['manifest_rows']}")
    print(f"STAGE2_HAS_SN={summary['stage2']['has_sn']}")
    print(f"STAGE2_HAS_PARTNO={summary['stage2']['has_part_no']}")
    print(f"STAGE2_MISSING_ALL={summary['stage2']['missing_all']}")


if __name__ == "__main__":
    main()
