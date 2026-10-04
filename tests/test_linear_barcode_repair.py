import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from huaweiocr.barcode import linear_repair as repair


class LinearBarcodeRepairTest(unittest.TestCase):
    def test_checksum_ties_keep_the_original_deterministic_order(self):
        for length, mode, expected in ((1, "digits", "0"), (3, "alnum", "1KB"), (8, "digits", "00395799")):
            with self.subTest(length=length, mode=mode):
                sampled = np.full(11 * (length + 2) + 13, 0.5, dtype=np.float32)
                self.assertEqual(repair.decode_dp(sampled, length, mode)[0], expected)

    def test_batched_geometry_scores_match_scalar_sampling_exactly(self):
        rng = np.random.default_rng(42)
        profile = rng.random(901, dtype=np.float32)
        geometries = [
            (length, float(x), float(module), 11 * (length + 2) + 13)
            for length in (8, 12, 20)
            for x, module in zip(rng.uniform(-5, 300, 40), rng.uniform(1, 8, 40))
        ]
        starts, stops = repair._start_stop_scores(profile, geometries)
        expected = [repair.start_stop_score(profile, x, module, count)[1:] for _, x, module, count in geometries]
        np.testing.assert_array_equal(np.column_stack((starts, stops)), expected)

    def test_checksum_search_matches_exhaustive_two_character_optimum(self):
        rng = np.random.default_rng(29)
        sampled = rng.random(11 * 4 + 13, dtype=np.float32)
        segments = sampled[:44].reshape(4, 11)
        start = repair.symbol_costs(segments[0], repair.BITS11[[repair.START_B]])[0]
        stop = ((repair.STOP_BITS - sampled[44:]) ** 2).mean()
        costs = [repair.symbol_costs(segment, repair.BITS11) for segment in segments[1:]]
        options = []
        for first, second in itertools.product(repair.ALNUM_CODES, repeat=2):
            checksum = (repair.START_B + first + 2 * second) % 103
            total = np.float32(start + costs[0][first])
            total = np.float32(total + costs[1][second])
            total = np.float32(total + costs[2][checksum])
            total = np.float32(total + stop)
            options.append((float(total), chr(first + 32) + chr(second + 32)))
        score, text = min(options)
        decoded = repair.decode_dp(sampled, 2, "alnum")
        self.assertEqual((decoded[1], decoded[0]), (score, text))

    def test_template_geometry_reuse_does_not_share_results_between_images(self):
        templates = repair.parse_templates("4E26########,4E25########")
        with tempfile.TemporaryDirectory() as directory:
            for text in ("4E2612345678", "4E2587654321"):
                path = Path(directory) / "label.png"
                repair.render_code128b(text).save(path)
                candidates, _, _ = repair.decode_image(path, "alnum", "12", templates=templates)
                self.assertEqual(candidates[0].text, text)
                self.assertIsNone(repair.CURRENT_ALLOWED_OVERRIDE)


if __name__ == "__main__":
    unittest.main()
