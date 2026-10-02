import unittest

import numpy as np

from huaweiocr.core import geometry


class _Image:
    shape = (100, 200, 3)


class CropGeometryTest(unittest.TestCase):
    def setUp(self):
        self.crop = geometry

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


class CropFromBoxClampTest(unittest.TestCase):
    """Negative coordinates must not wrap around to the opposite image edge."""

    def test_negative_x1_does_not_wrap_to_image_end(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, 150:160] = 255
        crop = geometry.crop_from_box(img, (-195, 0, 10, 10))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (10, 10))
        self.assertEqual(int(crop.max()), 0)

    def test_box_fully_left_of_image_returns_none(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        self.assertIsNone(geometry.crop_from_box(img, (-50, 0, -10, 10)))

    def test_in_bounds_box_is_unchanged(self):
        img = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        crop = geometry.crop_from_box(img, (10, 20, 30, 40))
        self.assertEqual(crop.shape[:2], (20, 20))
        self.assertTrue(np.array_equal(crop, img[20:40, 10:30]))

    def test_over_bounds_box_clips_as_before(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = geometry.crop_from_box(img, (190, 90, 500, 500))
        self.assertEqual(crop.shape[:2], (10, 10))


if __name__ == "__main__":
    unittest.main()
