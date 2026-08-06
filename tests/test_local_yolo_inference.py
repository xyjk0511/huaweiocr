import importlib
import importlib.util
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
    def _load_local_yolo_module(self, env_updates, existing_paths):
        module_name = "local_yolo_config_test"
        module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_yolo.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        normalized_existing = {
            os.path.normcase(os.path.abspath(path))
            for path in existing_paths
        }

        original_exists = os.path.exists

        def fake_exists(path):
            normalized = os.path.normcase(os.path.abspath(path))
            if normalized in normalized_existing:
                return True
            return original_exists(path)

        with mock.patch.dict(os.environ, env_updates, clear=False):
            with mock.patch("os.path.exists", side_effect=fake_exists):
                spec.loader.exec_module(module)

        sys.modules.pop(module_name, None)
        return module

    def _load_huaweiocr_spec_datas(self, cwd):
        hooks = types.ModuleType("PyInstaller.utils.hooks")
        # Non-empty so the spec's Cython-collect guard (which raises when the
        # build venv lacks Cython) is satisfied; entries carry a distinctive dst
        # that the detector-bundling assertions below filter out.
        hooks.collect_data_files = lambda *args, **kwargs: [("stub_data_file", "stub_dest")]
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
            ocr_models_dir = os.path.join(root, "bundle", "models", "official_models")
            for model_dir in (
                "PP-OCRv5_server_det",
                "PP-OCRv5_server_rec",
                "en_PP-OCRv5_mobile_rec",
                "PP-LCNet_x1_0_textline_ori",
            ):
                os.makedirs(os.path.join(ocr_models_dir, model_dir))
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

    def test_primary_label_model_prefers_yolo26_2class_latest_by_default(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        legacy_path = os.path.join(repo_root, "local_models", "detectors", "label_detector.onnx")
        latest_path = os.path.join(
            repo_root,
            "local_models",
            "training",
            "label_detector_yolo26s_960_2class_ignore_v1",
            "weights",
            "best.pt",
        )
        module = self._load_local_yolo_module(
            {
                "LOCAL_YOLO_LABEL_MODEL": "",
                "LOCAL_YOLO_LABEL_MODEL_PREFER_LATEST": "1",
            },
            [legacy_path, latest_path],
        )

        self.assertEqual(
            module.get_model_path("huawei-2ha7t/7"),
            module._normalized_path(latest_path),
        )
        self.assertNotIn("huawei-2ha7t-hardcase/1", module.DEFAULT_MODEL_SPECS)
        self.assertEqual(
            module.DEFAULT_MODEL_SPECS["huawei-2ha7t/7"].names,
            ("huawei_label", "shipping_ignore"),
        )

    def test_primary_label_model_falls_back_to_release_detector_when_latest_is_missing(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        legacy_path = os.path.join(repo_root, "local_models", "detectors", "label_detector.onnx")
        module = self._load_local_yolo_module(
            {
                "LOCAL_YOLO_LABEL_MODEL": "",
                "LOCAL_YOLO_LABEL_MODEL_PREFER_LATEST": "1",
                "LOCAL_YOLO_LATEST_LABEL_MODEL": os.path.join(
                    repo_root,
                    "local_models",
                    "training",
                    "__missing_latest_label_detector__",
                    "weights",
                    "best.onnx",
                ),
            },
            [legacy_path],
        )

        self.assertEqual(
            module.get_model_path("huawei-2ha7t/7"),
            module._normalized_path(legacy_path),
        )
        self.assertNotIn("huawei-2ha7t-hardcase/1", module.DEFAULT_MODEL_SPECS)

    def test_explicit_hardcase_override_keeps_secondary_supplement_model(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        legacy_path = os.path.join(repo_root, "local_models", "detectors", "label_detector.onnx")
        latest_path = os.path.join(
            repo_root,
            "local_models",
            "training",
            "label_detector_yolo26s_960_2class_ignore_v1",
            "weights",
            "best.pt",
        )
        hardcase_path = os.path.join(
            repo_root,
            "local_models",
            "training",
            "label_detector_v3s_hardcases_skipcpu",
            "weights",
            "best.onnx",
        )
        module = self._load_local_yolo_module(
            {
                "LOCAL_YOLO_LABEL_MODEL": "",
                "LOCAL_YOLO_LABEL_MODEL_PREFER_LATEST": "1",
                "LOCAL_YOLO_HARDCASE_LABEL_MODEL": hardcase_path,
            },
            [legacy_path, latest_path, hardcase_path],
        )

        self.assertEqual(
            module.get_model_path("huawei-2ha7t/7"),
            module._normalized_path(latest_path),
        )
        self.assertEqual(
            module.get_model_path("huawei-2ha7t-hardcase/1"),
            module._normalized_path(hardcase_path),
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

    def test_decode_yolov8_output_supports_postprocessed_xyxy_onnx_output(self):
        output = np.array(
            [
                [
                    [110.0, 70.0, 150.0, 90.0, 0.91, 0.0],
                    [300.0, 185.0, 360.0, 215.0, 0.83, 1.0],
                    [50.0, 50.0, 50.0, 60.0, 0.99, 0.0],
                ]
            ],
            dtype=np.float32,
        )

        preds = local_yolo.decode_yolov8_output(
            output,
            names=("huawei_label", "shipping_ignore"),
            original_shape=(640, 640, 3),
            scale=1.0,
            pad=(0, 0),
            conf_threshold=0.25,
            nms_threshold=0.45,
        )

        self.assertEqual([p["class"] for p in preds], ["huawei_label", "shipping_ignore"])
        self.assertAlmostEqual(preds[0]["x"], 130.0)
        self.assertAlmostEqual(preds[0]["y"], 80.0)
        self.assertAlmostEqual(preds[0]["width"], 40.0)
        self.assertAlmostEqual(preds[0]["height"], 20.0)
        self.assertAlmostEqual(preds[0]["confidence"], 0.91, places=5)
        self.assertEqual(preds[1]["class_id"], 1)

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

    def test_local_yolo_client_uses_020_default_conf_for_stage1_label_onnx_model(self):
        class FakeDetector:
            def __init__(self, spec, **kwargs):
                self.spec = spec
                self.kwargs = kwargs

            def predict(self, image_path):
                return [{"class": self.spec.names[0], "image_path": image_path}]

        client = local_yolo.LocalYoloClient(
            model_specs={
                "huawei-2ha7t/7": local_yolo.ModelSpec("unused.onnx", ("huawei_label", "shipping_ignore")),
            },
            detector_cls=FakeDetector,
        )

        client.infer("image.jpg", model_id="huawei-2ha7t/7")
        detector = client.detectors["huawei-2ha7t/7"]
        self.assertAlmostEqual(detector.kwargs["conf_threshold"], 0.20)

    def test_local_yolo_client_uses_035_default_conf_for_stage1_label_pt_model(self):
        class FakeDetector:
            def __init__(self, spec, **kwargs):
                self.spec = spec
                self.kwargs = kwargs

            def predict(self, image_path):
                return [{"class": self.spec.names[0], "image_path": image_path}]

        client = local_yolo.LocalYoloClient(
            model_specs={
                "huawei-2ha7t/7": local_yolo.ModelSpec("unused.pt", ("huawei_label", "shipping_ignore")),
            },
            detector_cls=FakeDetector,
        )

        client.infer("image.jpg", model_id="huawei-2ha7t/7")
        detector = client.detectors["huawei-2ha7t/7"]
        self.assertAlmostEqual(detector.kwargs["conf_threshold"], 0.35)

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
