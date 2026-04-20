import importlib
import inspect
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
            crop.STAGE2_DIR = os.path.join(out_dir, "stage2_fields_run_x")
            crop.STAGE1_DIR = os.path.join(out_dir, "stage1_labels_run_x")
            crop.set_log_level = lambda level: None
            crop.main = lambda **kwargs: {"label_count": 1, "manifest_rows": 1}

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
                os.path.join(crop.STAGE2_DIR, "model_sn_ocr.jsonl"),
            )

    def test_zero_label_crop_returns_nonzero_without_scanning(self):
        import run_all

        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "input")
            out_dir = os.path.join(root, "out")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "sample.png"), "wb") as image:
                image.write(b"not-a-real-image")

            crop = types.ModuleType("crop")
            crop.set_log_level = lambda level: None
            crop.main = mock.Mock(return_value={"label_count": 0, "manifest_rows": 0})

            scan2 = types.ModuleType("scan2")
            scan2.set_log_level = lambda level: None
            scan2.main = mock.Mock()

            argv = ["run_all.py", "--input", input_dir, "--out", out_dir]
            with mock.patch.dict(sys.modules, {"crop": crop, "scan2": scan2}):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(run_all.main(), 1)

            scan2.main.assert_not_called()

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


class PaddleOcrModelKwargsTest(unittest.TestCase):
    def test_local_model_dirs_include_matching_model_names(self):
        for name in ("ocr", "paddle", "paddleocr", "app_paths"):
            sys.modules.pop(name, None)

        paddle = types.ModuleType("paddle")
        paddle.set_device = lambda _device: None
        sys.modules["paddle"] = paddle

        paddleocr = types.ModuleType("paddleocr")

        class DummyPaddleOCR:
            def __init__(
                self,
                use_doc_orientation_classify=None,
                use_doc_unwarping=None,
                use_textline_orientation=None,
                text_detection_model_name=None,
                text_detection_model_dir=None,
                text_recognition_model_name=None,
                text_recognition_model_dir=None,
                textline_orientation_model_name=None,
                textline_orientation_model_dir=None,
            ):
                pass

        paddleocr.PaddleOCR = DummyPaddleOCR
        sys.modules["paddleocr"] = paddleocr

        app_paths = types.ModuleType("app_paths")
        app_paths.ensure_models_installed = lambda: None
        sys.modules["app_paths"] = app_paths

        import ocr

        with tempfile.TemporaryDirectory() as root:
            models = os.path.join(root, "official_models")
            for name in (
                "PP-OCRv5_server_det",
                "en_PP-OCRv5_mobile_rec",
                "PP-LCNet_x1_0_textline_ori",
            ):
                os.makedirs(os.path.join(models, name))

            kwargs = ocr._paddleocr_model_kwargs(models)

        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_server_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "en_PP-OCRv5_mobile_rec")
        self.assertEqual(kwargs["textline_orientation_model_name"], "PP-LCNet_x1_0_textline_ori")
        self.assertIs(kwargs["use_doc_orientation_classify"], False)
        self.assertIs(kwargs["use_doc_unwarping"], False)
        self.assertIs(kwargs["use_textline_orientation"], True)
        self.assertTrue(kwargs["text_detection_model_dir"].endswith("PP-OCRv5_server_det"))
        self.assertTrue(kwargs["text_recognition_model_dir"].endswith("en_PP-OCRv5_mobile_rec"))
        self.assertTrue(kwargs["textline_orientation_model_dir"].endswith("PP-LCNet_x1_0_textline_ori"))


class Scan2ManifestTest(unittest.TestCase):
    def test_main_signature_keeps_legacy_arguments(self):
        scan2 = _import_scan2()

        signature = inspect.signature(scan2.main)
        self.assertEqual(
            list(signature.parameters),
            ["out_dir", "model_dir", "sn_dir", "out_jsonl", "debug_log", "log_level"],
        )
        self.assertEqual(
            str(signature),
            "(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=None, log_level='info')",
        )

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

    def test_manifest_bad_json_fails_fast(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            os.makedirs(os.path.join(stage2, "model"))
            os.makedirs(os.path.join(stage2, "sn"))
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write("{bad json}\n")

            with self.assertRaisesRegex(ValueError, "Invalid manifest JSON"):
                scan2.main(
                    model_dir=os.path.join(stage2, "model"),
                    sn_dir=os.path.join(stage2, "sn"),
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

    def test_manifest_missing_label_id_fails_fast(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            os.makedirs(os.path.join(stage2, "model"))
            os.makedirs(os.path.join(stage2, "sn"))
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"model_path": None}) + "\n")

            with self.assertRaisesRegex(ValueError, "missing label_id"):
                scan2.main(
                    model_dir=os.path.join(stage2, "model"),
                    sn_dir=os.path.join(stage2, "sn"),
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

    def test_raw_fields_keep_full_values_by_default(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(model_path, "wb").close()
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path, "sn_path": sn_path}) + "\n")

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.object(scan2, "recognize_model", return_value=("MODEL1", "RAW_MODEL_SECRET_123456", "test")):
                with mock.patch.object(scan2, "recognize_sn", return_value=("SN1", "RAW_SN_SECRET_123456", "test", {})):
                    with mock.patch.dict(os.environ, {"SCAN2_MASK_RAW": "", "SCAN2_UNSAFE_RAW": ""}, clear=False):
                        scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=out_jsonl,
                            debug_log=os.path.join(root, "debug.log"),
                        )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertEqual(row["model"], "MODEL1")
            self.assertEqual(row["sn"], "SN1")
            self.assertEqual(row["model_raw"], "RAW_MODEL_SECRET_123456")
            self.assertEqual(row["sn_raw"], "RAW_SN_SECRET_123456")

    def test_raw_fields_can_be_masked_with_env_flag(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(model_path, "wb").close()
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path, "sn_path": sn_path}) + "\n")

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.object(scan2, "recognize_model", return_value=("MODEL1", "RAW_MODEL_SECRET_123456", "test")):
                with mock.patch.object(scan2, "recognize_sn", return_value=("SN1", "RAW_SN_SECRET_123456", "test", {})):
                    with mock.patch.dict(os.environ, {"SCAN2_MASK_RAW": "1", "SCAN2_UNSAFE_RAW": ""}, clear=False):
                        scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=out_jsonl,
                            debug_log=os.path.join(root, "debug.log"),
                        )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertNotIn("RAW_MODEL_SECRET_123456", row["model_raw"])
            self.assertNotIn("RAW_SN_SECRET_123456", row["sn_raw"])

    def test_info_log_keeps_full_model_and_sn_values(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(model_path, "wb").close()
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path, "sn_path": sn_path}) + "\n")

            logs = []
            old_sink = scan2.LOG_SINK
            scan2.set_log_sink(logs.append)
            try:
                with mock.patch.object(scan2, "recognize_model", return_value=("S380-S8P2T", "raw", "ocr_color")):
                    with mock.patch.object(scan2, "recognize_sn", return_value=("4E25B0105849", "raw", "barcode", {})):
                        scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=os.path.join(root, "out.jsonl"),
                            debug_log=os.path.join(root, "debug.log"),
                        )
            finally:
                scan2.set_log_sink(old_sink)

            joined = "\n".join(logs)
            self.assertIn("MODEL=S380-S8P2T", joined)
            self.assertIn("SN=4E25B0105849", joined)
            self.assertNotIn("S380**8P2T", joined)
            self.assertNotIn("4E25****5849", joined)

    def test_label_crop_barcode_is_used_when_sn_crop_is_missing(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            label_path = os.path.join(root, "label.png")
            open(label_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "label_crop": label_path}) + "\n")

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.object(scan2, "read_barcodes", return_value=["SN:4E25A0170000"]):
                with mock.patch.object(scan2, "load_for_ocr_color") as load_ocr:
                    stats = scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=out_jsonl,
                        debug_log=os.path.join(root, "debug.log"),
                    )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertEqual(row["sn"], "4E25A0170000")
            self.assertEqual(row["sn_src"], "barcode")
            self.assertEqual(stats["sn_total"], 1)
            self.assertEqual(stats["sn_success"], 1)
            load_ocr.assert_not_called()

    def test_model_barcode_is_enabled_by_default(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            open(model_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path}) + "\n")

            with mock.patch.object(scan2, "recognize_model", return_value=("MODEL1", "raw", "barcode")) as recognize_model:
                with mock.patch.dict(os.environ, {}, clear=True):
                    stats = scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=os.path.join(root, "out.jsonl"),
                        debug_log=os.path.join(root, "debug.log"),
                    )

            self.assertTrue(recognize_model.call_args.kwargs["use_barcode"])
            self.assertEqual(stats["model_total"], 1)
            self.assertEqual(stats["model_success"], 1)
            self.assertEqual(stats["model_barcode_hits"], 1)
            self.assertEqual(stats["model_barcode_hit_rate"], 1.0)

    def test_model_barcode_can_be_disabled_with_env_flag(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            open(model_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path}) + "\n")

            with mock.patch.object(scan2, "recognize_model", return_value=("MODEL1", "raw", "test")) as recognize_model:
                with mock.patch.dict(os.environ, {"SCAN2_MODEL_BARCODE": "0"}, clear=True):
                    scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=os.path.join(root, "out.jsonl"),
                        debug_log=os.path.join(root, "debug.log"),
                    )

            self.assertFalse(recognize_model.call_args.kwargs["use_barcode"])

    def test_model_recognition_skips_barcode_cli_when_disabled(self):
        scan2 = _import_scan2()
        fake_img = object()

        with mock.patch.object(scan2, "try_model_from_barcode") as barcode_mock:
            with mock.patch.object(scan2, "load_for_ocr_color", return_value=None):
                with mock.patch.object(scan2, "load_and_preprocess", return_value=fake_img):
                    with mock.patch.object(scan2, "ocr_text_with_details", return_value=("", "", [])):
                        model, _raw, source = scan2.recognize_model("model.png", use_barcode=False)

        self.assertEqual(model, "")
        self.assertEqual(source, "none")
        barcode_mock.assert_not_called()


class CropTempFileTest(unittest.TestCase):
    def test_original_path_for_label_id_resolves_current_input_dir(self):
        crop = _import_crop()
        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "input")
            os.makedirs(input_dir)
            original = os.path.join(input_dir, "image_01.jpg")
            with open(original, "wb") as f:
                f.write(b"image")

            crop.configure_paths(input_dir=input_dir, out_dir=root)

            self.assertEqual(
                crop.original_path_for_label_id("image_01.jpg__label_3"),
                original,
            )

    def test_stage1_uses_extension_in_label_name_to_avoid_same_stem_collision(self):
        crop = _import_crop()
        fake_img = types.SimpleNamespace(shape=(100, 100, 3), size=1)
        pred = {"x": 50, "y": 50, "width": 20, "height": 20, "class": crop.MODEL1_LABEL_CLASS}

        with tempfile.TemporaryDirectory() as root:
            crop.configure_paths(input_dir=os.path.join(root, "input"), out_dir=root)
            os.makedirs(crop.INPUT_DIR)
            path_png = os.path.join(root, "a", "same.png")
            path_jpg = os.path.join(root, "b", "same.jpg")

            with mock.patch.object(crop, "read_image", return_value=fake_img):
                with mock.patch.object(crop, "infer_with_resize", return_value=[pred]):
                    with mock.patch.object(crop, "crop_from_pred", return_value=fake_img):
                        with mock.patch.object(crop, "save_png_required", side_effect=lambda path, _img, _ctx: path):
                            out_png = crop.stage1_crop_labels(path_png)
                            out_jpg = crop.stage1_crop_labels(path_jpg)

            self.assertEqual(os.path.basename(out_png[0]), "same.png__label_1.png")
            self.assertEqual(os.path.basename(out_jpg[0]), "same.jpg__label_1.png")
            self.assertNotEqual(os.path.basename(out_png[0]), os.path.basename(out_jpg[0]))

    def test_save_png_required_raises_when_write_fails(self):
        crop = _import_crop()
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "out.png")
            with mock.patch.object(crop, "save_png", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "Failed to write test crop"):
                    crop.save_png_required(target, object(), "test crop")

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

    def test_gui_log_mask_keeps_app_relative_output_paths(self):
        sys.modules.pop("gui_app", None)
        import gui_app

        app_path = os.path.join(os.getcwd(), "stage2_fields", "manifest.jsonl")
        external_path = r"F:\wechat\xwechat_files\sample.jpg"

        masked = gui_app._mask_path_text(
            f"Manifest: {app_path}; source: {external_path}"
        )

        self.assertIn(os.path.join("stage2_fields", "manifest.jsonl"), masked)
        self.assertIn("source: [path]", masked)
        self.assertNotIn(external_path, masked)

    def test_gui_run_pipeline_requests_clean_crop_outputs(self):
        sys.modules.pop("gui_app", None)
        import gui_app

        source = inspect.getsource(gui_app.App.run_pipeline)

        self.assertIn("crop_module.main(input_dir=input_dir, clean=True)", source)


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

            data_dir = os.path.join(root, "data")
            target = os.path.join(data_dir, "models", "official_models", "model_a")
            os.makedirs(target)
            with open(os.path.join(target, "partial.bin"), "wb") as f:
                f.write(b"partial")

            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)

            with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                with mock.patch.dict(os.environ, {"HUAWEIOCR_DATA_DIR": data_dir, "HUAWEIOCR_MODEL_DIR": ""}, clear=False):
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

            data_dir = os.path.join(root, "data")
            target_root = os.path.join(data_dir, "models", "official_models")
            os.makedirs(target_root)
            lock_path = os.path.join(target_root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))

            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)

            with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                with mock.patch.dict(os.environ, {"HUAWEIOCR_DATA_DIR": data_dir, "HUAWEIOCR_MODEL_DIR": ""}, clear=False):
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
