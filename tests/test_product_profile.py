import importlib
import json
import os
import sys
import tempfile
import unittest

from huaweiocr.core import profile


class ProductProfileTest(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.get("HUAWEIOCR_PRODUCT_PROFILE")
        os.environ.pop("HUAWEIOCR_PRODUCT_PROFILE", None)
        profile.reload_profile()

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop("HUAWEIOCR_PRODUCT_PROFILE", None)
        else:
            os.environ["HUAWEIOCR_PRODUCT_PROFILE"] = self._old_env
        profile.reload_profile()
        if "huaweiocr.core.extract" in sys.modules:
            importlib.reload(sys.modules["huaweiocr.core.extract"])

    def _write_profile(self, directory, payload):
        path = os.path.join(directory, "product_profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.environ["HUAWEIOCR_PRODUCT_PROFILE"] = path
        return path

    def test_builtin_profile_tables_are_preserved(self):
        builtin = profile.BUILTIN_PROFILE
        self.assertEqual(1, builtin["schema_version"])
        self.assertEqual("AP362E", builtin["part_no_model_map"]["50087147"])
        self.assertEqual("AP265E", builtin["part_no_model_map"]["50087144"])
        self.assertEqual("AR180Pro", builtin["part_no_model_map"]["50010843"])
        self.assertEqual("AR280", builtin["part_no_model_map"]["50010845"])
        self.assertEqual("S380-S8P2T", builtin["part_no_model_map"]["98012125"])
        self.assertEqual("S110-8P1T", builtin["part_no_model_map"]["98012406"])
        self.assertIn("AP162E", builtin["known_model_codes"])
        self.assertIn("AP362E", builtin["known_model_codes"])
        self.assertIn("AR180Pro", builtin["known_model_codes"])
        self.assertIn("AR280", builtin["known_model_codes"])
        self.assertIn("S380-S8P2T", builtin["known_model_codes"])
        self.assertEqual(profile.KNOWN_MODEL_CODES, set(builtin["known_model_codes"]))
        self.assertEqual(14, len(builtin["part_no_model_map"]))
        self.assertEqual(11, len(builtin["known_model_codes"]))

    def test_external_profile_adds_entries_for_profile_and_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_profile(tmp, {
                "schema_version": 1,
                "known_model_codes": ["AP999E"],
                "part_no_model_map": {"50123456": "AP999E"},
            })
            merged = profile.reload_profile()
            self.assertEqual("AP999E", merged["part_no_model_map"]["50123456"])
            self.assertIn("AP999E", merged["known_model_codes"])

            from huaweiocr.core import extract

            importlib.reload(extract)
            self.assertEqual("AP999E", extract.PART_NO_MODEL_MAP["50123456"])
            self.assertIn("AP999E", extract.KNOWN_MODEL_CODES)

    def test_external_profile_cannot_override_builtin_part_no(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_profile(tmp, {
                "part_no_model_map": {"50087147": "AP999E"},
            })
            merged = profile.reload_profile()
            self.assertEqual("AP362E", merged["part_no_model_map"]["50087147"])
            self.assertNotEqual("AP999E", merged["part_no_model_map"]["50087147"])
            self.assertNotIn("AP999E", merged["part_no_model_map"].values())

    def test_corrupt_external_profile_falls_back_to_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "product_profile.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{")
            os.environ["HUAWEIOCR_PRODUCT_PROFILE"] = path

            merged = profile.reload_profile()
            self.assertEqual("AP362E", merged["part_no_model_map"]["50087147"])
            self.assertIn("AP162E", merged["known_model_codes"])
            self.assertEqual(profile.BUILTIN_PROFILE["part_no_model_map"], merged["part_no_model_map"])
            self.assertEqual(
                set(profile.BUILTIN_PROFILE["known_model_codes"]),
                set(merged["known_model_codes"]),
            )
