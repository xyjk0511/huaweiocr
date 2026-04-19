import importlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


def _install_crop_import_fakes():
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.IMWRITE_JPEG_QUALITY = 1
    cv2.imdecode = lambda *args, **kwargs: None
    cv2.imencode = lambda *args, **kwargs: (False, b"")
    cv2.dnn = types.SimpleNamespace(NMSBoxes=lambda *args, **kwargs: [])
    sys.modules["cv2"] = cv2

    numpy = types.ModuleType("numpy")
    numpy.uint8 = object()
    numpy.fromfile = lambda *args, **kwargs: b""
    sys.modules["numpy"] = numpy

    inference_sdk = types.ModuleType("inference_sdk")

    class DummyInferenceHTTPClient:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, *args, **kwargs):
            return {"predictions": []}

    inference_sdk.InferenceHTTPClient = DummyInferenceHTTPClient
    sys.modules["inference_sdk"] = inference_sdk


def _install_scan2_import_fakes():
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.THRESH_BINARY = 0
    cv2.THRESH_OTSU = 0
    cv2.COLOR_BGR2GRAY = 1
    cv2.COLOR_GRAY2BGR = 2
    cv2.BORDER_CONSTANT = 0
    cv2.ROTATE_90_COUNTERCLOCKWISE = 0
    cv2.INTER_CUBIC = 0
    cv2.imread = lambda *args, **kwargs: None
    sys.modules["cv2"] = cv2

    numpy = types.ModuleType("numpy")
    sys.modules["numpy"] = numpy

    ocr = types.ModuleType("ocr")
    ocr.init_ocr = lambda: object()
    ocr.ocr_one_image = lambda *args, **kwargs: ([], "")
    sys.modules["ocr"] = ocr

    barcode = types.ModuleType("barcode")
    barcode.decode_small_patch = lambda *args, **kwargs: {"results": []}
    sys.modules["barcode"] = barcode

    app_paths = types.ModuleType("app_paths")
    app_paths.ensure_models_installed = lambda: None
    sys.modules["app_paths"] = app_paths


def _import_crop():
    os.environ["API_KEY"] = "test-key"
    sys.modules.pop("crop", None)
    _install_crop_import_fakes()
    return importlib.import_module("crop")


def _import_scan2():
    sys.modules.pop("scan2", None)
    _install_scan2_import_fakes()
    return importlib.import_module("scan2")


def _install_barcode_import_fakes():
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2GRAY = 1
    cv2.COLOR_GRAY2BGR = 2
    cv2.BORDER_CONSTANT = 0
    cv2.INTER_CUBIC = 0
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 0
    cv2.THRESH_BINARY = 0
    cv2.MORPH_RECT = 0
    cv2.MORPH_CLOSE = 0
    cv2.ROTATE_90_CLOCKWISE = 0
    cv2.cvtColor = lambda img, *args, **kwargs: img
    cv2.resize = lambda img, *args, **kwargs: img
    cv2.bitwise_not = lambda img: img
    cv2.copyMakeBorder = lambda img, *args, **kwargs: img
    cv2.createCLAHE = lambda *args, **kwargs: types.SimpleNamespace(apply=lambda img: img)
    cv2.GaussianBlur = lambda img, *args, **kwargs: img
    cv2.addWeighted = lambda img, *args, **kwargs: img
    cv2.adaptiveThreshold = lambda img, *args, **kwargs: img
    cv2.getStructuringElement = lambda *args, **kwargs: object()
    cv2.morphologyEx = lambda img, *args, **kwargs: img
    cv2.imwrite = lambda *args, **kwargs: True
    sys.modules["cv2"] = cv2

    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    numpy.rot90 = lambda img, k: img
    sys.modules["numpy"] = numpy

    pyzbar_pkg = types.ModuleType("pyzbar")
    pyzbar_mod = types.ModuleType("pyzbar.pyzbar")
    pyzbar_mod.ZBarSymbol = types.SimpleNamespace(CODE128=object())
    pyzbar_mod.decode = lambda *args, **kwargs: []
    pyzbar_pkg.pyzbar = pyzbar_mod
    sys.modules["pyzbar"] = pyzbar_pkg
    sys.modules["pyzbar.pyzbar"] = pyzbar_mod


def _import_barcode():
    sys.modules.pop("barcode", None)
    _install_barcode_import_fakes()
    return importlib.import_module("barcode")


@unittest.skipUnless(os.name == "nt", "Windows file locking behavior only")
class LockedOutputDirsTest(unittest.TestCase):
    def test_locked_manifest_switches_stage2_to_run_directory(self):
        crop = _import_crop()
        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "new_images")
            original_stage2 = os.path.join(root, "stage2_fields")
            os.makedirs(original_stage2)
            locked_manifest = os.path.join(original_stage2, "manifest.jsonl")

            with open(locked_manifest, "w", encoding="utf-8") as locked:
                locked.write("locked\n")
                locked.flush()

                crop.configure_paths(input_dir=input_dir, out_dir=root)
                crop.ensure_dirs()

                self.assertNotEqual(
                    os.path.abspath(original_stage2),
                    os.path.abspath(crop.STAGE2_DIR),
                )
                self.assertTrue(os.path.isdir(crop.OUT_MODEL_DIR))
                self.assertTrue(os.path.isdir(crop.OUT_SN_DIR))
                self.assertTrue(os.path.isdir(crop.FAILED_DIR))
                self.assertEqual(
                    crop.MANIFEST_PATH,
                    os.path.join(crop.STAGE2_DIR, "manifest.jsonl"),
                )
                self.assertTrue(os.path.exists(locked_manifest))


class RunAllPathPropagationTest(unittest.TestCase):
    def test_scan2_reads_actual_crop_output_dirs(self):
        import run_all

        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "input")
            out_dir = os.path.join(root, "out")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "sample.png"), "wb") as image:
                image.write(b"not-a-real-image")

            crop = types.ModuleType("crop")
            crop.OUT_MODEL_DIR = os.path.join(out_dir, "stage2_fields_run_x", "model")
            crop.OUT_SN_DIR = os.path.join(out_dir, "stage2_fields_run_x", "sn")
            crop.STAGE1_DIR = os.path.join(out_dir, "stage1_labels_run_x")
            crop.set_log_level = lambda level: None
            crop.main = lambda **kwargs: None

            calls = {}
            scan2 = types.ModuleType("scan2")
            scan2.set_log_level = lambda level: None

            def fake_scan2_main(**kwargs):
                calls.update(kwargs)
                return {"sn_total": 0}

            scan2.main = fake_scan2_main

            argv = [
                "run_all.py",
                "--input",
                input_dir,
                "--out",
                out_dir,
            ]
            with mock.patch.dict(sys.modules, {"crop": crop, "scan2": scan2}):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(run_all.main(), 0)

            self.assertEqual(calls["model_dir"], crop.OUT_MODEL_DIR)
            self.assertEqual(calls["sn_dir"], crop.OUT_SN_DIR)
            self.assertEqual(
                calls["out_jsonl"],
                os.path.join(out_dir, "model_sn_ocr.jsonl"),
            )

    def test_empty_input_returns_nonzero_without_running_pipeline(self):
        import run_all

        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "empty")
            os.makedirs(input_dir)
            crop = types.ModuleType("crop")
            crop.set_log_level = lambda level: None
            crop.main = mock.Mock()

            scan2 = types.ModuleType("scan2")
            scan2.set_log_level = lambda level: None
            scan2.main = mock.Mock()

            argv = ["run_all.py", "--input", input_dir, "--out", os.path.join(root, "out")]
            with mock.patch.dict(sys.modules, {"crop": crop, "scan2": scan2}):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(run_all.main(), 2)

            crop.main.assert_not_called()
            scan2.main.assert_not_called()


class Scan2ManifestTest(unittest.TestCase):
    def test_manifest_keeps_labels_without_model_or_sn_crops(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)

            manifest_path = os.path.join(stage2, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "img__with__sep__label_1"}) + "\n")
                manifest.write(json.dumps({"label_id": "missing_both__label_1"}) + "\n")

            with open(os.path.join(model_dir, "img__with__sep__label_1__model.png"), "wb") as image:
                image.write(b"model")

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.object(scan2, "recognize_model", return_value=("M1", "raw", "test")):
                stats = scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=out_jsonl,
                    debug_log=os.path.join(root, "debug.log"),
                )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            self.assertEqual([row["label_id"] for row in rows], ["img__with__sep__label_1", "missing_both__label_1"])
            self.assertEqual(rows[1]["model_src"], "missing")
            self.assertEqual(rows[1]["sn_src"], "missing")
            self.assertEqual(stats["sn_total"], 0)


class CropTempFileTest(unittest.TestCase):
    def test_infer_with_resize_uses_unique_temp_file_and_cleans_it(self):
        crop = _import_crop()

        calls = []

        class FakeClient:
            def infer(self, path, model_id):
                calls.append(path)
                self.seen_exists = os.path.exists(path)
                return {"predictions": []}

        fake_client = FakeClient()
        fake_img = types.SimpleNamespace(shape=(100, 200, 3))

        def write_tmp(_bgr, path, quality=85):
            with open(path, "wb") as f:
                f.write(b"tmp")
            return True

        with tempfile.TemporaryDirectory() as root:
            crop.TMP_DIR = root
            with mock.patch.object(crop, "get_inference_client", return_value=fake_client):
                with mock.patch.object(crop, "_write_tmp_jpg", side_effect=write_tmp):
                    self.assertEqual(crop.infer_with_resize(fake_img, "same__name.png", "model/1"), [])
                    self.assertEqual(crop.infer_with_resize(fake_img, "same__name.png", "model/1"), [])

            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0], calls[1])
            self.assertFalse(os.path.exists(calls[0]))
            self.assertFalse(os.path.exists(calls[1]))


class GuiPipelineTest(unittest.TestCase):
    def test_same_basename_sources_are_staged_with_unique_names(self):
        import gui_pipeline

        with tempfile.TemporaryDirectory() as root:
            source_a = os.path.join(root, "a")
            source_b = os.path.join(root, "b")
            os.makedirs(source_a)
            os.makedirs(source_b)
            path_a = os.path.join(source_a, "same.png")
            path_b = os.path.join(source_b, "same.png")
            with open(path_a, "wb") as f:
                f.write(b"a")
            with open(path_b, "wb") as f:
                f.write(b"b")

            run_dir, records = gui_pipeline.copy_images_to_unique_run_dir(
                [path_a, path_b],
                os.path.join(root, "new_images"),
            )

            names = [record["input_name"] for record in records]
            self.assertEqual(len(names), 2)
            self.assertEqual(len(set(names)), 2)
            self.assertEqual(names, ["input_0001.png", "input_0002.png"])
            self.assertTrue(os.path.exists(os.path.join(run_dir, names[0])))
            self.assertTrue(os.path.exists(os.path.join(run_dir, names[1])))
            with open(os.path.join(run_dir, "source_manifest.jsonl"), "r", encoding="utf-8") as f:
                manifest_rows = [json.loads(line) for line in f]
            manifest_text = json.dumps(manifest_rows, ensure_ascii=False)
            self.assertNotIn("source_path", manifest_text)
            self.assertNotIn("input_path", manifest_text)
            self.assertNotIn(os.path.abspath(root), manifest_text)
            for row in manifest_rows:
                self.assertEqual(set(row), {"source_index", "input_name", "sha256"})
                self.assertNotIn("same.png", row["input_name"])
            self.assertEqual([row["source_index"] for row in manifest_rows], [1, 2])
            self.assertEqual([row["input_name"] for row in manifest_rows], names)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest_rows))

    def test_gui_import_does_not_import_pipeline_modules(self):
        sys.modules.pop("gui_app", None)
        sys.modules.pop("crop", None)
        sys.modules.pop("scan2", None)
        sys.modules.pop("barcode", None)
        sys.modules.pop("cv2", None)
        sys.modules.pop("numpy", None)
        sys.modules.pop("pyzbar", None)
        sys.modules.pop("pyzbar.pyzbar", None)

        importlib.import_module("gui_app")

        self.assertNotIn("crop", sys.modules)
        self.assertNotIn("scan2", sys.modules)


class BarcodeCliBudgetTest(unittest.TestCase):
    def test_decode_small_patch_caps_cli_attempts(self):
        barcode = _import_barcode()
        fake_img = types.SimpleNamespace(shape=(10, 10))
        calls = []

        def fake_cli(_img, _tag):
            calls.append(_tag)
            return []

        with mock.patch.object(barcode, "decode_with_transforms", return_value=[]):
            with mock.patch.object(barcode, "crop_bar_band", return_value=fake_img):
                with mock.patch.object(barcode, "enhance_band", return_value=(fake_img, fake_img)):
                    with mock.patch.object(barcode, "_rotate90", side_effect=lambda img, _k: img):
                        with mock.patch.object(barcode, "decode_with_cli", side_effect=fake_cli):
                            barcode.decode_small_patch(fake_img)

        self.assertEqual(len(calls), barcode.CLI_MAX_CALLS_PER_PATCH)


class Scan2DebugLogTest(unittest.TestCase):
    def test_info_mode_does_not_create_debug_log(self):
        scan2 = _import_scan2()
        with tempfile.TemporaryDirectory() as root:
            log_path = os.path.join(root, "debug.log")
            scan2.DEBUG_LOG_PATH = log_path
            scan2.set_log_level("info")

            scan2.append_debug("MODEL path C:/secret/customer_4E25A017ABCDEFG")

            self.assertFalse(os.path.exists(log_path))

    def test_debug_log_masks_sensitive_text(self):
        scan2 = _import_scan2()
        with tempfile.TemporaryDirectory() as root:
            log_path = os.path.join(root, "debug.log")
            scan2.DEBUG_LOG_PATH = log_path
            scan2.set_log_level("debug")

            scan2.append_sensitive_debug(r"SN C:\customer\asset_4E25A017ABCDEFG.png | 4E25A017ABCDEFG")

            with open(log_path, "r", encoding="utf-8") as f:
                data = f.read()
            self.assertNotIn("4E25A017ABCDEFG", data)
            self.assertNotIn("customer", data)
            self.assertNotIn("asset_4E25A017ABCDEFG.png", data)
            self.assertIn("4E25", data)


class AppPathsInstallTest(unittest.TestCase):
    def test_incomplete_model_dir_is_replaced(self):
        import app_paths

        with tempfile.TemporaryDirectory() as root:
            bundled = os.path.join(root, "bundled", "models", "official_models")
            source_model = os.path.join(bundled, "model_a")
            os.makedirs(source_model)
            with open(os.path.join(source_model, "weights.bin"), "wb") as f:
                f.write(b"complete")

            home = os.path.join(root, "home")
            target = os.path.join(home, ".paddlex", "official_models", "model_a")
            os.makedirs(target)
            with open(os.path.join(target, "partial.bin"), "wb") as f:
                f.write(b"partial")

            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)

            with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                with mock.patch.object(app_paths.os.path, "expanduser", return_value=home):
                    app_paths.ensure_models_installed()

            self.assertTrue(os.path.exists(os.path.join(target, "weights.bin")))
            self.assertTrue(os.path.exists(os.path.join(target, app_paths.MODEL_INSTALL_MARKER)))
            self.assertFalse(os.path.exists(os.path.join(target, "partial.bin")))

    def test_stale_model_install_lock_is_recovered(self):
        import app_paths

        with tempfile.TemporaryDirectory() as root:
            bundled = os.path.join(root, "bundled", "models", "official_models")
            source_model = os.path.join(bundled, "model_a")
            os.makedirs(source_model)
            with open(os.path.join(source_model, "weights.bin"), "wb") as f:
                f.write(b"complete")

            home = os.path.join(root, "home")
            target_root = os.path.join(home, ".paddlex", "official_models")
            os.makedirs(target_root)
            lock_path = os.path.join(target_root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))

            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)

            with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                with mock.patch.object(app_paths.os.path, "expanduser", return_value=home):
                    app_paths.ensure_models_installed()

            target = os.path.join(target_root, "model_a")
            self.assertTrue(os.path.exists(os.path.join(target, "weights.bin")))
            self.assertTrue(os.path.exists(os.path.join(target, app_paths.MODEL_INSTALL_MARKER)))
            self.assertFalse(os.path.exists(lock_path))

    def test_stale_lock_reclaim_does_not_remove_changed_lock(self):
        import app_paths

        with tempfile.TemporaryDirectory() as root:
            lock_path = os.path.join(root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))
            real_read = app_paths._read_lock_snapshot
            calls = {"count": 0}

            def racing_read(path):
                snapshot = real_read(path)
                calls["count"] += 1
                if calls["count"] == 1:
                    with open(lock_path, "w", encoding="utf-8") as f:
                        f.write(f"{os.getpid()}\n{time.time()}\n")
                return snapshot

            with mock.patch.object(app_paths, "_read_lock_snapshot", side_effect=racing_read):
                self.assertFalse(app_paths._reclaim_stale_lock(lock_path))

            self.assertTrue(os.path.exists(lock_path))
            with open(lock_path, "r", encoding="utf-8") as f:
                data = f.read()
            self.assertIn(str(os.getpid()), data)


if __name__ == "__main__":
    unittest.main()
