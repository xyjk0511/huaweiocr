import os
import sys
import shutil
import tempfile
import time
import ctypes

MODEL_INSTALL_MARKER = ".huaweiocr_complete"
MODEL_INSTALL_LOCK_MALFORMED_GRACE_SECONDS = 1


def _pid_is_running(pid):
    if not pid or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_lock_metadata(lock_path):
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        return int(lines[0]), float(lines[1])
    except Exception:
        return None, None


def _lock_is_stale(lock_path):
    now = time.time()
    pid, created = _read_lock_metadata(lock_path)
    if pid is not None and created is not None:
        return not _pid_is_running(pid)
    try:
        age = now - os.path.getmtime(lock_path)
    except OSError:
        return True
    return age > MODEL_INSTALL_LOCK_MALFORMED_GRACE_SECONDS


def _write_lock_metadata(lock_fd):
    payload = f"{os.getpid()}\n{time.time()}\n".encode("ascii")
    os.write(lock_fd, payload)
    os.fsync(lock_fd)


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
            _write_lock_metadata(lock_fd)
        except FileExistsError:
            if _lock_is_stale(lock_path):
                try:
                    os.remove(lock_path)
                    continue
                except FileNotFoundError:
                    continue
                except PermissionError:
                    pass
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
