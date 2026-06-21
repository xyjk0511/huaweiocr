import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import local_yolo


class LocalYoloInferenceTests(unittest.TestCase):
    def _load_huaweiocr_spec_datas(self, cwd):
        hooks = types.ModuleType("PyInstaller.utils.hooks")
        hooks.collect_data_files = lambda *args, **kwargs: []
        hooks.collect_dynamic_libs = lambda *args, **kwargs: []
        hooks.copy_metadata = lambda *args, **kwargs: []

        fake_modules = {
            "PyInstaller": types.ModuleType("PyInstaller"),
            "PyInstaller.utils": types.ModuleType("PyInstaller.utils"),
            "PyInstaller.utils.hooks": hooks,
        }

        class StubAnalysis:
            def __init__(self, scripts, pathex=None, binaries=None, datas=None, **_kwargs):
                self.scripts = scripts
                self.pure = []
                self.binaries = list(binaries or [])
                self.datas = list(datas or [])

        spec_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "HuaweiOCR.spec")
        namespace = {
            "Analysis": StubAnalysis,
            "PYZ": lambda pure: object(),
            "EXE": lambda *args, **kwargs: object(),
            "COLLECT": lambda *args, **kwargs: object(),
        }
        with open(spec_path, "r", encoding="utf-8") as f:
            source = f.read()
        with mock.patch.dict(sys.modules, fake_modules):
            old_cwd = os.getcwd()
            os.chdir(cwd)
            try:
                exec(compile(source, spec_path, "exec"), namespace)
            finally:
                os.chdir(old_cwd)
        return namespace["datas"]

    def test_pyinstaller_spec_requires_local_detector_onnx(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError) as raised:
                self._load_huaweiocr_spec_datas(root)

        message = str(raised.exception)
        self.assertIn("Missing required local ONNX detector model(s)", message)
        self.assertIn(os.path.join("local_models", "detectors", "label_detector.onnx"), message)
        self.assertIn(os.path.join("local_models", "detectors", "field_detector.onnx"), message)

    def test_pyinstaller_spec_bundles_required_local_detector_onnx_without_dotenv(self):
        with tempfile.TemporaryDirectory() as root:
            detector_dir = os.path.join(root, "local_models", "detectors")
            os.makedirs(detector_dir)
            for filename in (
                "label_detector.onnx",
                "field_detector.onnx",
                "ignored_extra_detector.onnx",
            ):
                with open(os.path.join(detector_dir, filename), "wb") as f:
                    f.write(b"onnx")
            with open(os.path.join(root, ".env"), "w", encoding="utf-8") as f:
                f.write("API_KEY=should-not-bundle\n")

            datas = self._load_huaweiocr_spec_datas(root)

        normalized = {
            (src.replace("/", "\\"), dst.replace("/", "\\"))
            for src, dst in datas
            if dst.replace("/", "\\") == "local_models\\detectors"
        }

        self.assertEqual(
            normalized,
            {
                (
                    "local_models\\detectors\\label_detector.onnx",
                    "local_models\\detectors",
                ),
                (
                    "local_models\\detectors\\field_detector.onnx",
                    "local_models\\detectors",
                ),
            },
        )
        self.assertFalse(any(os.path.basename(src) == ".env" for src, _dst in datas))

    @unittest.skipUnless(os.name == "nt", "Windows batch entrypoint only")
    def _run_start_bat(self, env_updates):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        with tempfile.TemporaryDirectory() as root:
            shutil.copy2(os.path.join(repo_root, "start.bat"), os.path.join(root, "start.bat"))
            os.makedirs(os.path.join(root, "new_images"))
            with open(os.path.join(root, "new_images", "sample.jpg"), "wb") as f:
                f.write(b"image")
            with open(os.path.join(root, "run_all.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import os, sys\n"
                    "with open('args.txt', 'w', encoding='utf-8') as out:\n"
                    "    out.write('\\n'.join(sys.argv[1:]))\n"
                    "sys.exit(int(os.environ.get('STUB_EXIT', '0')))\n"
                )

            env = os.environ.copy()
            for name in ("API_KEY", "CROP_INFERENCE_BACKEND", "HUAWEIOCR_NO_PAUSE", "CI"):
                env.pop(name, None)
            env.update(env_updates)

            result = subprocess.run(
                ["cmd.exe", "/c", "start.bat"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            args_path = os.path.join(root, "args.txt")
            args = ""
            if os.path.exists(args_path):
                with open(args_path, "r", encoding="utf-8") as f:
                    args = f.read()
            return result, args

    @unittest.skipUnless(os.name == "nt", "Windows batch entrypoint only")
    def test_start_bat_defaults_to_local_backend_without_api_key_and_returns_python_exit(self):
        result, args = self._run_start_bat({"HUAWEIOCR_NO_PAUSE": "1", "STUB_EXIT": "7"})

        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("--input\nnew_images\n--out\nruns", args)
        self.assertNotIn("--pause", args)
        self.assertNotIn("API_KEY is not set", result.stdout + result.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows batch entrypoint only")
    def test_start_bat_requires_api_key_for_roboflow_backend(self):
        result, args = self._run_start_bat(
            {
                "CROP_INFERENCE_BACKEND": "roboflow",
                "HUAWEIOCR_NO_PAUSE": "1",
                "STUB_EXIT": "0",
            }
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(args, "")
        self.assertIn("API_KEY", result.stdout + result.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows batch entrypoint only")
    def test_start_bat_requires_api_key_for_remote_backend(self):
        result, args = self._run_start_bat(
            {
                "CROP_INFERENCE_BACKEND": "remote",
                "HUAWEIOCR_NO_PAUSE": "1",
                "STUB_EXIT": "0",
            }
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(args, "")
        self.assertIn("API_KEY", result.stdout + result.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows batch entrypoint only")
    def test_start_bat_requires_api_key_for_cloud_backend(self):
        result, args = self._run_start_bat(
            {
                "CROP_INFERENCE_BACKEND": "cloud",
                "HUAWEIOCR_NO_PAUSE": "1",
                "STUB_EXIT": "0",
            }
        )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(args, "")
        self.assertIn("API_KEY", result.stdout + result.stderr)

    def test_default_field_detector_names_include_partno(self):
        self.assertEqual(
            local_yolo.DEFAULT_MODEL_SPECS["sn_model/9"].names,
            ("model", "partno", "sn"),
        )

    def test_decode_yolov8_output_returns_roboflow_style_predictions(self):
        output = np.zeros((1, 6, 10), dtype=np.float32)
        output[0, 0, 0] = 100.0
        output[0, 1, 0] = 80.0
        output[0, 2, 0] = 40.0
        output[0, 3, 0] = 20.0
        output[0, 4, 0] = 0.9
        output[0, 5, 0] = 0.1

        output[0, 0, 1] = 300.0
        output[0, 1, 1] = 200.0
        output[0, 2, 1] = 60.0
        output[0, 3, 1] = 30.0
        output[0, 4, 1] = 0.2
        output[0, 5, 1] = 0.8

        preds = local_yolo.decode_yolov8_output(
            output,
            names=("model", "sn"),
            original_shape=(640, 640, 3),
            scale=1.0,
            pad=(0, 0),
            conf_threshold=0.25,
            nms_threshold=0.45,
        )

        self.assertEqual([p["class"] for p in preds], ["model", "sn"])
        self.assertEqual([p["class_name"] for p in preds], ["model", "sn"])
        self.assertAlmostEqual(preds[0]["x"], 100.0)
        self.assertAlmostEqual(preds[0]["y"], 80.0)
        self.assertAlmostEqual(preds[0]["width"], 40.0)
        self.assertAlmostEqual(preds[0]["height"], 20.0)
        self.assertAlmostEqual(preds[0]["confidence"], 0.9, places=5)

    def test_local_yolo_client_returns_predictions_dict(self):
        class FakeDetector:
            def __init__(self, spec, **_kwargs):
                self.spec = spec

            def predict(self, image_path):
                return [{"class": self.spec.names[0], "image_path": image_path}]

        client = local_yolo.LocalYoloClient(
            model_specs={"model/1": local_yolo.ModelSpec("unused.onnx", ("label",))},
            detector_cls=FakeDetector,
        )

        self.assertEqual(
            client.infer("image.jpg", model_id="model/1"),
            {"predictions": [{"class": "label", "image_path": "image.jpg"}]},
        )

    def test_crop_defaults_to_local_backend_without_api_key(self):
        fake_inference_sdk = types.ModuleType("inference_sdk")
        fake_inference_sdk.InferenceHTTPClient = mock.Mock()

        class FakeLocalYoloClient:
            pass

        fake_local_yolo = types.ModuleType("local_yolo")
        fake_local_yolo.LocalYoloClient = FakeLocalYoloClient

        old_crop = sys.modules.pop("crop", None)
        try:
            with mock.patch.dict(
                sys.modules,
                {"inference_sdk": fake_inference_sdk, "local_yolo": fake_local_yolo},
            ):
                with mock.patch.dict(
                    os.environ,
                    {"CROP_INFERENCE_BACKEND": "", "API_KEY": ""},
                ):
                    os.environ.pop("CROP_INFERENCE_BACKEND", None)
                    os.environ.pop("API_KEY", None)
                    crop = importlib.import_module("crop")
                    self.assertEqual(crop.inference_backend(), "local")
                    self.assertIsInstance(crop.get_inference_client(), FakeLocalYoloClient)
                    fake_inference_sdk.InferenceHTTPClient.assert_not_called()
        finally:
            sys.modules.pop("crop", None)
            if old_crop is not None:
                sys.modules["crop"] = old_crop


if __name__ == "__main__":
    unittest.main()
