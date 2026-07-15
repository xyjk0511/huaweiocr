import unittest

import numpy as np

from huaweiocr.core import geometry


class CropFromBoxClampTest(unittest.TestCase):
    """crop_from_box must clamp to image bounds so a negative coordinate cannot
    be silently reinterpreted by numpy as an index 'from the end'."""

    def test_negative_x1_does_not_wrap_to_image_end(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, 150:160] = 255  # bright strip near the right edge
        # Without clamping, numpy would read img[:, 5:10] (a 10x5 strip) instead
        # of the intended top-left 10x10 region.
        crop = geometry.crop_from_box(img, (-195, 0, 10, 10))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (10, 10))
        self.assertEqual(int(crop.max()), 0)  # left region is black, not the strip

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
