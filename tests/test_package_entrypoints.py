import os
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class PackageEntrypointsTest(unittest.TestCase):
    def test_cli_image_enumeration_keeps_its_configurable_extension_set(self):
        import run_all

        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "sample.tif"
            image.touch()
            (Path(folder) / "sample.png").touch()
            with mock.patch.object(run_all, "SUPPORTED_INPUT_EXTS", {".tif"}):
                self.assertEqual(run_all._list_images(folder), [str(image)])

    def test_legacy_imports_alias_the_canonical_modules(self):
        # A fresh interpreter also verifies that GUI import stays lazy. Other
        # tests deliberately install incomplete cv2/numpy modules in this process.
        code = """
import importlib
import sys
import gui_app
assert 'crop' not in sys.modules
assert 'scan2' not in sys.modules
for legacy, canonical in (
    ('crop', 'huaweiocr.detect.crop_pipeline'),
    ('scan2', 'huaweiocr.recognition.pipeline'),
    ('run_all', 'huaweiocr.cli'),
    ('gui_app', 'huaweiocr.ui.app'),
    ('gui_pipeline', 'huaweiocr.ui.pipeline'),
    ('gui_i18n', 'huaweiocr.ui.i18n'),
):
    left = importlib.import_module(legacy)
    right = importlib.import_module(canonical)
    assert left is right, legacy
    left._compatibility_probe = object()
    assert right._compatibility_probe is left._compatibility_probe, legacy
print('module aliases and lazy GUI import verified')
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=30,
            creationflags=HIDDEN_FLAGS,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_and_script_have_the_same_cli_help(self):
        results = [
            subprocess.run(
                [sys.executable, *args, "--help"], cwd=REPO_ROOT, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=15,
                creationflags=HIDDEN_FLAGS,
            )
            for args in (("run_all.py",), ("-m", "huaweiocr"))
        ]
        for result in results:
            self.assertEqual(result.returncode, 0, result.stderr)
        # argparse's program name differs; the accepted options and defaults do not.
        left, right = (result.stdout[result.stdout.index("options:"):] for result in results)
        self.assertEqual(left, right)

    def test_head_review_snapshot_loads_its_own_package_dependencies(self):
        code = """
import importlib
from pathlib import Path
import tempfile
from tools.run_stage1_review_export_head_snapshot import _load_head_snapshot
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    crop, local_yolo = _load_head_snapshot(root)
    assert callable(crop.configure_paths)
    for module in (crop, local_yolo, importlib.import_module('huaweiocr.core.geometry')):
        assert Path(module.__file__).is_relative_to(root), module.__file__
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=30,
            creationflags=HIDDEN_FLAGS,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
