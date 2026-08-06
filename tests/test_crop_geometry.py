import importlib
import math
import os
import sys
import types
import unittest


def tearDownModule():
    for name in ("crop", "cv2", "inference_sdk", "numpy"):
        sys.modules.pop(name, None)


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


def _import_crop():
    os.environ["API_KEY"] = "test-key"
    sys.modules.pop("crop", None)
    _install_crop_import_fakes()
    return importlib.import_module("crop")


class _Image:
    shape = (100, 200, 3)


class CropGeometryTest(unittest.TestCase):
    def setUp(self):
        self.crop = _import_crop()

    def test_box_iou_none(self):
        self.assertEqual(self.crop.box_iou(None, (0, 0, 10, 10)), 0.0)

    def test_box_iou_disjoint(self):
        self.assertEqual(self.crop.box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_box_iou_partial(self):
        self.assertAlmostEqual(self.crop.box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 0.14285714285714285)

    def test_box_iou_contained(self):
        self.assertEqual(self.crop.box_iou((0, 0, 10, 10), (2, 2, 8, 8)), 0.36)

    def test_box_iou_zero_area(self):
        self.assertEqual(self.crop.box_iou((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)

    def test_box_overlap_ratio_none(self):
        self.assertEqual(self.crop.box_overlap_ratio(None, (0, 0, 10, 10)), 0.0)

    def test_box_overlap_ratio_disjoint(self):
        self.assertEqual(self.crop.box_overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_box_overlap_ratio_partial(self):
        self.assertEqual(self.crop.box_overlap_ratio((0, 0, 10, 10), (5, 5, 15, 15)), 0.25)

    def test_box_overlap_ratio_contained(self):
        self.assertEqual(self.crop.box_overlap_ratio((0, 0, 10, 10), (2, 2, 8, 8)), 1.0)

    def test_box_overlap_ratio_zero_area(self):
        self.assertEqual(self.crop.box_overlap_ratio((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)

    def test_union_boxes_empty(self):
        self.assertIsNone(self.crop.union_boxes())

    def test_union_boxes_none_and_invalid(self):
        self.assertIsNone(self.crop.union_boxes(None, (0, 0, 0, 10)))

    def test_union_boxes_two_valid(self):
        self.assertEqual(self.crop.union_boxes((5, 5, 10, 10), (1, 2, 3, 4)), (1, 2, 10, 10))

    def test_union_boxes_skips_invalid(self):
        self.assertEqual(self.crop.union_boxes((5, 5, 10, 10), (9, 9, 8, 12), None), (5, 5, 10, 10))

    def test_slant_guard_zero_width(self):
        self.assertEqual(self.crop.slant_guard_px(0, 10), 0)

    def test_slant_guard_zero_max(self):
        self.assertEqual(self.crop.slant_guard_px(100, 0), 0)

    def test_slant_guard_basic(self):
        self.assertEqual(self.crop.slant_guard_px(100, 20), 6)

    def test_slant_guard_clamped_max(self):
        self.assertEqual(self.crop.slant_guard_px(1000, 5), 5)

    def test_slant_guard_min_px(self):
        self.assertEqual(self.crop.slant_guard_px(1, 10), 2)

    def test_expand_box_pixels_none(self):
        self.assertIsNone(self.crop.expand_box_pixels(_Image(), None, 5, 5))

    def test_expand_box_pixels_basic(self):
        self.assertEqual(self.crop.expand_box_pixels(_Image(), (10, 20, 30, 40), 5, 7), (5, 13, 35, 47))

    def test_expand_box_pixels_clamps_to_image_bounds(self):
        self.assertEqual(self.crop.expand_box_pixels(_Image(), (2, 3, 198, 99), 10, 10), (0, 0, 200, 100))

    def test_expand_box_pixels_truncates_float_padding(self):
        self.assertEqual(self.crop.expand_box_pixels(_Image(), (10, 20, 30, 40), 2.5, 3.5), (7, 16, 32, 43))


if __name__ == "__main__":
    unittest.main()
