import datetime
import hashlib
import json
import os
import shutil


def load_pipeline_modules():
    from app_paths import ensure_paddle_libs_on_path

    ensure_paddle_libs_on_path()
    import crop
    import scan2

    return crop, scan2


def copy_images_to_unique_run_dir(image_paths, root_dir, run_prefix="gui_run"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.abspath(os.path.join(root_dir, f"{run_prefix}_{timestamp}"))
    os.makedirs(run_dir, exist_ok=False)

    used_names = set()
    records = []
    for index, source in enumerate(image_paths, 1):
        _, ext = os.path.splitext(source)
        target_name = f"input_{index:04d}{ext.lower()}"
        while target_name.lower() in used_names:
            target_name = f"input_{index:04d}_{len(used_names)}{ext.lower()}"
        used_names.add(target_name.lower())

        target = os.path.join(run_dir, target_name)
        shutil.copy2(source, target)
        records.append(
            {
                "source_index": index,
                "source_name": os.path.basename(source),
                "input_name": target_name,
                "sha256": _sha256_file(target),
            }
        )

    manifest_path = os.path.join(run_dir, "source_manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(
                json.dumps(
                    {
                        "source_index": record["source_index"],
                        "input_name": record["input_name"],
                        "sha256": record["sha256"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return run_dir, records


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
