import os
import sys
import shutil
import tempfile
import time

MODEL_INSTALL_MARKER = ".huaweiocr_complete"


def get_base_dir():
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        internal = os.path.join(exe_dir, "_internal")
        if os.path.isdir(internal):
            return internal
        return exe_dir
    return os.path.dirname(os.path.abspath(__file__))


def get_resource_path(*parts):
    return os.path.join(get_base_dir(), *parts)


def ensure_models_installed():
    bundled = get_resource_path("models", "official_models")
    if not os.path.isdir(bundled):
        return

    target_root = os.path.join(os.path.expanduser("~"), ".paddlex", "official_models")
    os.makedirs(target_root, exist_ok=True)
    lock_path = os.path.join(target_root, ".huaweiocr_install.lock")
    lock_fd = None
    deadline = time.time() + 30
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for model install lock: {lock_path}")
            time.sleep(0.2)

    try:
        for name in os.listdir(bundled):
            src = os.path.join(bundled, name)
            dst = os.path.join(target_root, name)
            marker = os.path.join(dst, MODEL_INSTALL_MARKER)
            if not os.path.isdir(src):
                continue
            if os.path.isdir(dst) and os.path.isfile(marker):
                continue
            if os.path.exists(dst):
                shutil.rmtree(dst)
            tmp = tempfile.mkdtemp(prefix=f".{name}.", dir=target_root)
            try:
                shutil.copytree(src, tmp, dirs_exist_ok=True)
                with open(os.path.join(tmp, MODEL_INSTALL_MARKER), "w", encoding="utf-8") as f:
                    f.write("ok\n")
                os.replace(tmp, dst)
            finally:
                if os.path.exists(tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def get_barcode_cli_path():
    candidates = [
        get_resource_path("BarcodeReaderCLI", "bin", "BarcodeReaderCLI.exe"),
        get_resource_path("bundle", "BarcodeReaderCLI", "bin", "BarcodeReaderCLI.exe"),
        get_resource_path("dist", "HuaweiOCR", "_internal", "BarcodeReaderCLI", "bin", "BarcodeReaderCLI.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def ensure_paddle_libs_on_path():
    libs_dir = get_resource_path("paddle", "libs")
    if not os.path.isdir(libs_dir):
        return
    path = os.environ.get("PATH", "")
    if libs_dir not in path:
        os.environ["PATH"] = libs_dir + os.pathsep + path
