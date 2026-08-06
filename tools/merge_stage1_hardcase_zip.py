from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DATASET = REPO_ROOT / "datasets" / "huawei_yolov8_v2_boxonly"
OUT_DATASET = REPO_ROOT / "datasets" / "huawei_yolov8_v3_boxonly"
ZIP_PATH = Path(r"C:\Users\55093\Downloads\hardcase.yolov8.zip")

# Keep the most diagnostic hard cases in validation so detector tuning is judged on them.
VALID_BASENAMES = {
    "佳木斯",
    "常州",
    "上海",
    "武汉",
    "珠海",
    "上海嘉定",
}


def copy_base_dataset() -> None:
    if OUT_DATASET.exists():
        shutil.rmtree(OUT_DATASET)
    shutil.copytree(BASE_DATASET, OUT_DATASET)


def zip_image_stem(entry_name: str) -> str:
    stem = Path(entry_name).stem
    if "_jpg.rf." in stem:
        return stem.split("_jpg.rf.", 1)[0]
    if "_png.rf." in stem:
        return stem.split("_png.rf.", 1)[0]
    if ".rf." in stem:
        return stem.split(".rf.", 1)[0]
    return stem


def target_split(stem: str) -> str:
    return "valid" if stem in VALID_BASENAMES else "train"


def write_local_yaml() -> None:
    text = (
        f"path: {OUT_DATASET.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['huawei_label']\n"
    )
    (OUT_DATASET / "local_data.yaml").write_text(text, encoding="utf-8")


def import_zip() -> tuple[int, int]:
    imported_train = 0
    imported_valid = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue
            if not entry.filename.startswith("train/"):
                continue

            parts = Path(entry.filename).parts
            if len(parts) < 3:
                continue
            kind = parts[1]
            name = parts[2]
            stem = zip_image_stem(name)
            split = target_split(stem)
            out_dir = OUT_DATASET / split / kind
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / name
            with zf.open(entry) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            if kind == "images":
                if split == "train":
                    imported_train += 1
                else:
                    imported_valid += 1

    return imported_train, imported_valid


def main() -> None:
    if not BASE_DATASET.is_dir():
        raise FileNotFoundError(f"Base dataset missing: {BASE_DATASET}")
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Hardcase zip missing: {ZIP_PATH}")

    copy_base_dataset()
    train_count, valid_count = import_zip()
    write_local_yaml()
    print(f"Created dataset: {OUT_DATASET}")
    print(f"Imported hardcases -> train: {train_count}, valid: {valid_count}")


if __name__ == "__main__":
    main()
