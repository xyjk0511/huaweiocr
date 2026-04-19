import importlib
import json
import os
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
