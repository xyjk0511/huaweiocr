import datetime
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
        base_name = os.path.basename(source)
        stem, ext = os.path.splitext(base_name)
        target_name = base_name
        if target_name.lower() in used_names:
            target_name = f"{stem}__gui_{index:04d}{ext}"
        while target_name.lower() in used_names:
            target_name = f"{stem}__gui_{index:04d}_{len(used_names)}{ext}"
        used_names.add(target_name.lower())

        target = os.path.join(run_dir, target_name)
        shutil.copy2(source, target)
        records.append(
            {
                "source_path": os.path.abspath(source),
                "input_path": target,
                "input_name": target_name,
            }
        )

    manifest_path = os.path.join(run_dir, "source_manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    return run_dir, records
