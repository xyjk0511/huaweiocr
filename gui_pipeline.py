import datetime
import hashlib
import json
import os
import shutil
import threading


def _env_flag_default(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def load_pipeline_modules():
    from app_paths import ensure_paddle_libs_on_path

    ensure_paddle_libs_on_path()
    import crop
    import scan2

    return crop, scan2


def load_scan2_module():
    from app_paths import ensure_paddle_libs_on_path

    ensure_paddle_libs_on_path()
    import scan2

    return scan2


def start_ocr_prewarm_thread(log=None):
    default_prewarm = _env_flag_default("SCAN2_OCR_FALLBACK", False)
    if not _env_flag_default("HUAWEIOCR_PREWARM_OCR", default_prewarm):
        return None

    def _worker():
        try:
            scan2 = load_scan2_module()
            scan2.prewarm_ocr_engine(log=log)
        except Exception as exc:
            if log:
                log(f"OCR预热启动失败，将在识别时再初始化：{exc.__class__.__name__}: {exc}")

    thread = threading.Thread(
        target=_worker,
        name="HuaweiOCR-OCR-Prewarm",
        daemon=True,
    )
    thread.start()
    return thread


def copy_images_to_unique_run_dir(image_paths, root_dir, run_prefix="gui_run"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.abspath(os.path.join(root_dir, f"{run_prefix}_{timestamp}"))
    os.makedirs(run_dir, exist_ok=False)

    used_names = set()
    records = []
    for index, source in enumerate(image_paths, 1):
        original_name = os.path.basename(source)
        name_stem, ext = os.path.splitext(original_name)
        target_name = original_name
        suffix = 2
        while target_name.lower() in used_names:
            target_name = f"{name_stem}_{suffix}{ext}"
            suffix += 1
        used_names.add(target_name.lower())

        target = os.path.join(run_dir, target_name)
        shutil.copy2(source, target)
        records.append(
            {
                "source_index": index,
                "input_name": target_name,
                "sha256": _sha256_file(target),
            }
        )

    manifest_path = os.path.join(run_dir, "source_manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    return run_dir, records


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
