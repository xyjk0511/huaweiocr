import os
import subprocess


def hide_subprocess_windows():
    if os.name != "nt" or getattr(subprocess, "_huaweiocr_hidden_windows", False):
        return

    original_popen = subprocess.Popen
    hidden_flags = subprocess.CREATE_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0)

    class HiddenPopen(original_popen):
        def __init__(self, *args, **kwargs):
            startupinfo = kwargs.get("startupinfo")
            if startupinfo is None:
                startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | hidden_flags
            super().__init__(*args, **kwargs)

    HiddenPopen.__name__ = original_popen.__name__
    HiddenPopen.__qualname__ = original_popen.__qualname__
    HiddenPopen.__module__ = original_popen.__module__

    subprocess.Popen = HiddenPopen
    subprocess._huaweiocr_hidden_windows = True
