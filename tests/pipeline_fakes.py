"""Import fixtures for tests that exercise orchestration without OCR engines."""
import importlib
import math
import os
import sys
import tempfile
import types
import unittest


_DEPENDENCIES = ("cv2", "numpy", "inference_sdk", "pyzbar", "pyzbar.pyzbar")
_original_modules = {}


def _remember_dependencies():
    if not _original_modules:
        _original_modules.update((name, sys.modules.get(name)) for name in _DEPENDENCIES)
        unittest.addModuleCleanup(clear_import_fakes)


def clear_import_fakes():
    if not _original_modules:
        return
    for name in (
        "app_paths", "huaweiocr.io.paths_runtime", "barcode", "huaweiocr.barcode.generic",
        "crop", "huaweiocr.detect.crop_pipeline", "huaweiocr.detect.barcode_regions",
        "huaweiocr.detect.text_regions", "huaweiocr.detect.field_quality", "cv2",
        "gui_app", "huaweiocr.ui.app", "gui_app_en", "gui_pipeline", "huaweiocr.ui.pipeline",
        "numpy", "ocr", "huaweiocr.detect.ocr_engine", "paddle", "paddleocr", "pyzbar",
        "pyzbar.pyzbar", "scan2", "huaweiocr.recognition.pipeline", "sn_barcode",
        "huaweiocr.barcode.sn", "win_subprocess", "huaweiocr.io.win_subprocess",
    ):
        sys.modules.pop(name, None)
    for name, module in _original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    _original_modules.clear()


def import_crop():
    _remember_dependencies()
    os.environ["API_KEY"] = "test-key"
    for name in (
        "crop", "huaweiocr.detect.crop_pipeline", "huaweiocr.core.geometry",
        "huaweiocr.detect.barcode_regions", "huaweiocr.detect.text_regions",
        "huaweiocr.detect.field_quality",
    ):
        sys.modules.pop(name, None)
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.IMWRITE_JPEG_QUALITY = 1
    cv2.imdecode = lambda *args, **kwargs: None
    cv2.imencode = lambda *args, **kwargs: (False, b"")
    cv2.dnn = types.SimpleNamespace(NMSBoxes=lambda *args, **kwargs: [])
    sys.modules["cv2"] = cv2
    numpy = types.ModuleType("numpy")
    numpy.uint8 = object()
    numpy.ceil = math.ceil
    numpy.tan = math.tan
    numpy.deg2rad = math.radians
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
    return importlib.import_module("crop")


def import_scan2():
    _remember_dependencies()
    sys.modules.pop("scan2", None)
    sys.modules.pop("huaweiocr.recognition.pipeline", None)
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
    sys.modules["numpy"] = types.ModuleType("numpy")
    ocr = types.ModuleType("ocr")
    ocr.init_ocr = lambda: object()
    ocr.ocr_one_image = lambda *args, **kwargs: ([], "")
    sys.modules["ocr"] = ocr
    barcode = types.ModuleType("barcode")
    barcode.decode_small_patch = lambda *args, **kwargs: {"results": []}
    sys.modules["barcode"] = barcode
    app_paths = types.ModuleType("app_paths")
    user_data = tempfile.TemporaryDirectory(prefix="huaweiocr_test_data_")
    unittest.addModuleCleanup(user_data.cleanup)
    app_paths.ensure_models_installed = lambda: None
    app_paths.get_user_data_dir = lambda: user_data.name
    sys.modules["app_paths"] = app_paths
    return importlib.import_module("scan2")


def import_barcode():
    _remember_dependencies()
    sys.modules.pop("barcode", None)
    sys.modules.pop("huaweiocr.barcode.generic", None)
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
    return importlib.import_module("barcode")
