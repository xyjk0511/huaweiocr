from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--out", required=True, dest="out_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def image_stem_from_zip_name(name: str) -> str:
    # Roboflow export names follow: "<orig>_jpg.rf.<hash>.jpg"
    marker = "_jpg.rf."
    if marker in name:
        return name.split(marker, 1)[0]
    return Path(name).stem


def parse_label_classes(text: str) -> set[int]:
    classes: set[int] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        classes.add(int(parts[0]))
    return classes


def split_items(items: list[dict], seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, list[dict]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")

    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        has_ignore = 1 in item["classes"]
        key = "has_ignore" if has_ignore else "label_only"
        buckets[key].append(item)

    rng = random.Random(seed)
    split_map = {"train": [], "valid": [], "test": []}
    for bucket_items in buckets.values():
        rng.shuffle(bucket_items)
        total = len(bucket_items)
        train_n = round(total * train_ratio)
        val_n = round(total * val_ratio)
        if train_n + val_n > total:
            val_n = max(0, total - train_n)
        test_n = total - train_n - val_n

        split_map["train"].extend(bucket_items[:train_n])
        split_map["valid"].extend(bucket_items[train_n : train_n + val_n])
        split_map["test"].extend(bucket_items[train_n + val_n : train_n + val_n + test_n])

    for split_items_list in split_map.values():
        split_items_list.sort(key=lambda x: x["stem"])
    return split_map


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = out_dir / "_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(temp_dir)

    image_entries: dict[str, Path] = {}
    label_entries: dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        for path in (temp_dir / split / "images").glob("*"):
            if path.is_file():
                image_entries[image_stem_from_zip_name(path.name)] = path
        for path in (temp_dir / split / "labels").glob("*.txt"):
            if path.is_file():
                label_entries[image_stem_from_zip_name(path.name)] = path

    stems = sorted(set(image_entries) & set(label_entries))
    items: list[dict] = []
    for stem in stems:
        label_path = label_entries[stem]
        label_text = label_path.read_text(encoding="utf-8").strip()
        if not label_text:
            continue
        items.append(
            {
                "stem": stem,
                "image_path": image_entries[stem],
                "label_path": label_path,
                "label_text": label_text,
                "classes": parse_label_classes(label_text),
            }
        )

    split_map = split_items(
        items,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    for split_name, split_items_list in split_map.items():
        images_out = out_dir / split_name / "images"
        labels_out = out_dir / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for item in split_items_list:
            shutil.copy2(item["image_path"], images_out / item["image_path"].name)
            shutil.copy2(item["label_path"], labels_out / item["label_path"].name)

    yaml_text = "\n".join(
        [
            f"path: {out_dir.as_posix()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            "names:",
            "  0: huawei_label",
            "  1: shipping_ignore",
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    split_lines = []
    for split_name in ("train", "valid", "test"):
        split_labels = list((out_dir / split_name / "labels").glob("*.txt"))
        class_counts = defaultdict(int)
        ignore_images = 0
        for label_file in split_labels:
            text = label_file.read_text(encoding="utf-8").strip()
            classes = parse_label_classes(text)
            if 1 in classes:
                ignore_images += 1
            for raw_line in text.splitlines():
                if not raw_line.strip():
                    continue
                class_counts[int(raw_line.split()[0])] += 1
        split_lines.append(
            f"{split_name}\timages={len(split_labels)}\tclass0={class_counts[0]}\tclass1={class_counts[1]}\tignore_images={ignore_images}"
        )
    (out_dir / "split_summary.tsv").write_text("\n".join(split_lines) + "\n", encoding="utf-8")

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
