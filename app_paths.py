import os
import sys
import shutil


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

    for name in os.listdir(bundled):
        src = os.path.join(bundled, name)
        dst = os.path.join(target_root, name)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)


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
