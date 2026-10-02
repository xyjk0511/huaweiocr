import os
import unittest
from unittest import mock

from huaweiocr.core import extract
from tests.pipeline_fakes import import_scan2


class TextExtractTest(unittest.TestCase):
    def test_model_normalization(self):
        cases = (
            ("S1108T", "S110-8T"),
            ("S110-6T", "S110-5T"),
            ("S380S8P2T", "S380-S8P2T"),
            ("S8P27", "S380-S8P2T"),
            ("MO8S-O802", "S380-S8P2T"),
            ("AR180PRO", "AR180Pro"),
            ("AP162EAP162E", "AP162E"),
            ("AP162E DESC", "AP162E"),
            ("S380-S8P2DESC", "S380-S8P2T"),
            ("AP162E", "AP162E"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(extract.normalize_model(text), expected)

    def test_part_number_extraction(self):
        cases = (
            ("", []),
            ("PN: 50012345", ["50012345"]),
            ("Part No: 98054321 text 50012345", ["98054321", "50012345"]),
            ("abc50012345def", []),
            ("1P50000001 ZZ 1P50000001", ["50000001"]),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(extract.extract_part_numbers_from_text(text), expected)

    def test_code_cleanup(self):
        for text, expected in (
            ("sn: 4e25-a017 0000", "SN4E25A0170000"),
            ("ap162e desc", "AP162EDESC"),
            ("\u4e2d\u6587A-1", "A1"),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract._clean_code(text), expected)

    def test_sn_extraction(self):
        for text, expected in (
            ("", ""),
            ("SN: 4E25-A017-0000", "4E25A0170000"),
            ("noise 2151234567ERA123456 end", "2151234567ERA123456"),
            ("abc 4E25A0170000 xyz", "4E25A0170000"),
            ("MODEL AP162E", ""),
        ):
            with self.subTest(text=text):
                self.assertEqual(extract.extract_sn_from_text(text), expected)

    def test_model_candidates(self):
        cases = (
            ("MODEL: AP162E DESC", False, "AP162E"),
            ("Model S380 S8P2T", False, "S380-S8P2T"),
            ("MO8S O802", False, "S380-S8P2T"),
            ("MODEL: ZZ99X", False, ""),
            ("MODEL: ZZ99X", True, "ZZ99X"),
            ("MODEL: MAC1234", True, ""),
            ("Label has AX99Z extra", True, "AX99Z"),
            ("Label has AX99Z extra", False, ""),
        )
        with mock.patch.dict(os.environ):
            for name in (
                "SCAN2_ALLOW_UNKNOWN_MODELS", "SCAN2_LEARNED_MODEL_CODES_PATH",
                "SCAN2_PART_NO_MODEL_MAP_PATH",
            ):
                os.environ.pop(name, None)
            pipeline = import_scan2()
            for text, allow_unknown, expected in cases:
                with self.subTest(text=text, allow_unknown=allow_unknown):
                    self.assertEqual(
                        pipeline.extract_model_candidate_from_text(text, allow_unknown=allow_unknown),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
