import importlib
import os
import sys
import tempfile
import types
import unittest


def tearDownModule():
    for name in (
        "app_paths",
        "barcode",
        "cv2",
        "numpy",
        "ocr",
        "scan2",
        "sn_barcode",
    ):
        sys.modules.pop(name, None)


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
    user_data_dir = tempfile.mkdtemp(prefix="huaweiocr_test_data_")
    app_paths.ensure_models_installed = lambda: None
    app_paths.get_user_data_dir = lambda: user_data_dir
    sys.modules["app_paths"] = app_paths


def _import_scan2():
    for name in (
        "SCAN2_ALLOW_UNKNOWN_MODELS",
        "SCAN2_LEARNED_MODEL_CODES_PATH",
        "SCAN2_PART_NO_MODEL_MAP_PATH",
    ):
        os.environ.pop(name, None)
    sys.modules.pop("scan2", None)
    sys.modules.pop("sn_barcode", None)
    _install_scan2_import_fakes()
    return importlib.import_module("scan2")


class NormalizeModelTest(unittest.TestCase):
    def setUp(self):
        self.scan2 = _import_scan2()

    def test_s1108t_adds_dash(self):
        self.assertEqual(self.scan2.normalize_model("S1108T"), "S110-8T")

    def test_s110_6t_corrects_to_s110_5t(self):
        self.assertEqual(self.scan2.normalize_model("S110-6T"), "S110-5T")

    def test_s380_compact_adds_dash(self):
        self.assertEqual(self.scan2.normalize_model("S380S8P2T"), "S380-S8P2T")

    def test_s380_short_ocr_fix(self):
        self.assertEqual(self.scan2.normalize_model("S8P27"), "S380-S8P2T")

    def test_s380_noisy_ocr_fix(self):
        self.assertEqual(self.scan2.normalize_model("MO8S-O802"), "S380-S8P2T")

    def test_ar180pro_restores_camel_case(self):
        self.assertEqual(self.scan2.normalize_model("AR180PRO"), "AR180Pro")

    def test_repeated_model_halves(self):
        self.assertEqual(self.scan2.normalize_model("AP162EAP162E"), "AP162E")

    def test_trailing_desc_is_removed(self):
        self.assertEqual(self.scan2.normalize_model("AP162E DESC"), "AP162E")

    def test_s380_desc_is_removed_after_series_fix(self):
        self.assertEqual(self.scan2.normalize_model("S380-S8P2DESC"), "S380-S8P2T")

    def test_plain_model_passes_through(self):
        self.assertEqual(self.scan2.normalize_model("AP162E"), "AP162E")


class Scan2TextExtractTest(unittest.TestCase):
    def setUp(self):
        self.scan2 = _import_scan2()

    def test_extract_part_numbers_empty(self):
        self.assertEqual(self.scan2.extract_part_numbers_from_text(""), [])

    def test_extract_part_numbers_pn_prefix(self):
        self.assertEqual(self.scan2.extract_part_numbers_from_text("PN: 50012345"), ["50012345"])

    def test_extract_part_numbers_multiple_ordered(self):
        self.assertEqual(
            self.scan2.extract_part_numbers_from_text("Part No: 98054321 text 50012345"),
            ["98054321", "50012345"],
        )

    def test_extract_part_numbers_embedded_rejected(self):
        self.assertEqual(self.scan2.extract_part_numbers_from_text("abc50012345def"), [])

    def test_extract_part_numbers_deduplicates(self):
        self.assertEqual(self.scan2.extract_part_numbers_from_text("1P50000001 ZZ 1P50000001"), ["50000001"])

    def test_clean_code_removes_sn_punctuation(self):
        self.assertEqual(self.scan2._clean_code("sn: 4e25-a017 0000"), "SN4E25A0170000")

    def test_clean_code_uppercases_and_keeps_desc(self):
        self.assertEqual(self.scan2._clean_code("ap162e desc"), "AP162EDESC")

    def test_clean_code_drops_non_ascii(self):
        self.assertEqual(self.scan2._clean_code("\u4e2d\u6587A-1"), "A1")

    def test_extract_model_candidate_model_line_known(self):
        self.assertEqual(self.scan2.extract_model_candidate_from_text("MODEL: AP162E DESC", allow_unknown=False), "AP162E")

    def test_extract_model_candidate_s380_spaced(self):
        self.assertEqual(self.scan2.extract_model_candidate_from_text("Model S380 S8P2T", allow_unknown=False), "S380-S8P2T")

    def test_extract_model_candidate_s380_noisy(self):
        self.assertEqual(self.scan2.extract_model_candidate_from_text("MO8S O802", allow_unknown=False), "S380-S8P2T")

    def test_extract_model_candidate_unknown_rejected_by_default(self):
        self.assertEqual(self.scan2.extract_model_candidate_from_text("MODEL: ZZ99X", allow_unknown=False), "")

    def test_extract_model_candidate_unknown_allowed(self):
        self.assertEqual(self.scan2.extract_model_candidate_from_text("MODEL: ZZ99X", allow_unknown=True), "ZZ99X")

    def test_extract_model_candidate_bad_word_rejected_when_unknown_allowed(self):
        self.assertEqual(self.scan2.extract_model_candidate_from_text("MODEL: MAC1234", allow_unknown=True), "")

    def test_extract_model_candidate_token_unknown_allowed(self):
        self.assertEqual(self.scan2.extract_model_candidate_from_text("Label has AX99Z extra", allow_unknown=True), "AX99Z")

    def test_extract_model_candidate_token_unknown_rejected_by_default(self):
        self.assertEqual(self.scan2.extract_model_candidate_from_text("Label has AX99Z extra", allow_unknown=False), "")

    def test_extract_sn_empty(self):
        self.assertEqual(self.scan2.extract_sn_from_text(""), "")

    def test_extract_sn_hyphenated_12_char(self):
        self.assertEqual(self.scan2.extract_sn_from_text("SN: 4E25-A017-0000"), "4E25A0170000")

    def test_extract_sn_bounded_215_pattern(self):
        self.assertEqual(self.scan2.extract_sn_from_text("noise 2151234567ERA123456 end"), "2151234567ERA123456")

    def test_extract_sn_embedded_12_char(self):
        self.assertEqual(self.scan2.extract_sn_from_text("abc 4E25A0170000 xyz"), "4E25A0170000")

    def test_extract_sn_ignores_model_text(self):
        self.assertEqual(self.scan2.extract_sn_from_text("MODEL AP162E"), "")


if __name__ == "__main__":
    unittest.main()
