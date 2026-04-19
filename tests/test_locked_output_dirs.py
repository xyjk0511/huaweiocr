import importlib
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


def _import_crop():
    os.environ["API_KEY"] = "test-key"
    sys.modules.pop("crop", None)
    _install_crop_import_fakes()
    return importlib.import_module("crop")


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


if __name__ == "__main__":
    unittest.main()
