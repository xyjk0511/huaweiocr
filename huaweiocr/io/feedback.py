"""Build feedback packages from one OCR run directory."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path


STAGE2_DIR = "stage2_fields"
ROOT_FILES = ("run_summary.json", "source_manifest.jsonl")
STAGE2_FILES = ("manifest.jsonl", "model_sn_ocr.jsonl")
MISS_DIRS = ("miss_model", "miss_sn", "miss_both", "failed")


def build_feedback_package(run_dir: str, out_zip: str) -> dict:
    """把一次运行目录中的失败证据打包成 zip，返回统计 dict。"""
    run_root = Path(run_dir)
    zip_path = Path(out_zip)
    if zip_path.parent:
        zip_path.parent.mkdir(parents=True, exist_ok=True)

    files_added = 0
    skipped = 0
    misses = {name: 0 for name in MISS_DIRS}

    def add_file(zf: zipfile.ZipFile, path: Path, miss_kind: str | None = None) -> None:
        nonlocal files_added, skipped
        if not path.is_file():
            return
        arcname = path.relative_to(run_root).as_posix()
        try:
            data = path.read_bytes()
        except OSError:
            skipped += 1
            return
        zf.writestr(arcname, data)
        files_added += 1
        if miss_kind is not None:
            misses[miss_kind] += 1

    def iter_files(root: Path):
        if not root.is_dir():
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for filename in sorted(filenames):
                yield Path(dirpath) / filename

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ROOT_FILES:
            add_file(zf, run_root / name)
        stage2_root = run_root / STAGE2_DIR
        for name in STAGE2_FILES:
            add_file(zf, stage2_root / name)
        for miss_kind in MISS_DIRS:
            for path in iter_files(stage2_root / miss_kind):
                add_file(zf, path, miss_kind)

    return {
        "files": files_added,
        "misses": misses,
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "skipped": skipped,
    }
