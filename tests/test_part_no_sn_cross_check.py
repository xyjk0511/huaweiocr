import os
import tempfile
import unittest
from unittest import mock

from huaweiocr.barcode.sn import DecoderResult, SnBarcodeReport
from huaweiocr.core import part_no_sn_kinds as kinds
from huaweiocr.core.extract import extract_part_numbers_from_text
from huaweiocr.recognition import pipeline


class GenericPartNoExtractionTest(unittest.TestCase):
    def test_known_series_and_standalone_generic_part_numbers(self):
        self.assertEqual(extract_part_numbers_from_text("P/N 50087149"), ["50087149"])
        self.assertEqual(extract_part_numbers_from_text("part_no:02312345"), ["02312345"])
        self.assertEqual(extract_part_numbers_from_text("1P98013170"), ["98013170"])

    def test_generic_part_numbers_reject_dates_repeats_and_embedded_digits(self):
        self.assertEqual(extract_part_numbers_from_text("20261113"), [])
        self.assertEqual(extract_part_numbers_from_text("11111111"), [])
        self.assertTrue(kinds.is_generic_part_no("50000000"))
        self.assertEqual(extract_part_numbers_from_text("2150087149LDS4014472"), [])
        self.assertEqual(extract_part_numbers_from_text("123456789"), [])
        self.assertEqual(extract_part_numbers_from_text("AB12345678"), [])


class RepairedSnConflictTest(unittest.TestCase):
    def test_embedded_part_no_must_match_label(self):
        self.assertEqual(
            kinds.repaired_sn_conflict("21500772777ERW049560", "50087149", ()),
            "part_no_mismatch",
        )
        self.assertEqual(kinds.repaired_sn_conflict("2150087149LDS4014472", "50087149", ()), "")

    def test_4e_serial_conflicts_only_with_21_only_evidence(self):
        self.assertEqual(
            kinds.repaired_sn_conflict("4E2616602199", "50087149", {"21"}),
            "sn_family_conflict",
        )
        self.assertEqual(kinds.repaired_sn_conflict("4E2616602199", "98012125", {"4E"}), "")
        self.assertEqual(kinds.repaired_sn_conflict("4E2616602199", "98012125", {"21", "4E"}), "")
        self.assertEqual(kinds.repaired_sn_conflict("4E2616602199", "98012125", ()), "")
        self.assertEqual(kinds.repaired_sn_conflict("4E2616602199", "", {"21"}), "")

    def test_kind_evidence_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "kinds.json")
            with mock.patch.dict(os.environ, {"SCAN2_PART_NO_SN_KINDS_PATH": path}):
                self.assertEqual(
                    kinds.record_part_no_sn_kinds(
                        [("50087149", "21"), ("98012125", "4E"), ("20261113", "21"), ("50087149", "X")]
                    ),
                    2,
                )
                self.assertEqual(
                    kinds.load_part_no_sn_kinds(),
                    {"50087149": {"21"}, "98012125": {"4E"}},
                )


def _report(sn, decoder):
    return SnBarcodeReport(
        status="hit",
        sn=sn,
        raw_text=sn,
        source="sn",
        source_region=f"sn.{decoder}",
        decoder_name=decoder,
        decoded_count=1,
        results=[DecoderResult(decoder, sn, "sn", f"sn.{decoder}")],
    )


class PipelineCrossCheckTest(unittest.TestCase):
    def test_rejects_conflicting_repairs_and_derives_part_no_from_genuine_sn(self):
        cases = {
            "genuine": ("21500871494ERB006054", "pyzbar", "50087149"),
            "forged_4e": ("4E2616602199", "code128_pixel_repair", "50087149"),
            "forged_21": ("21500772777ERW049560", "code128_pixel_repair", "50087149"),
            "s380_repair": ("4E2616602199", "code128_pixel_repair", "98012125"),
            "no_part_no": ("21980131704ES6000762", "zxingcpp", ""),
            "s110_repair": ("4E2616602199", "code128_pixel_repair", "98013170"),
        }
        keys = list(cases)
        records = {key: {"sn_path": f"{key}.png"} for key in keys}
        sn_reports = {key: _report(sn, decoder) for key, (sn, decoder, _) in cases.items()}
        sn_results = {
            key: (sn, f"[BARCODE] {sn}", "barcode", sn_reports[key].to_meta())
            for key, (sn, _, _) in cases.items()
        }
        part_no_by_key = {key: part_no for key, (_, _, part_no) in cases.items() if part_no}
        part_no_src_by_key = {}
        ocr_jobs = []
        part_no_scan_miss_keys = {"no_part_no"}

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "kinds.json")
            with mock.patch.dict(os.environ, {"SCAN2_PART_NO_SN_KINDS_PATH": path}):
                observations = pipeline._cross_check_sn_with_part_no(
                    keys, records, sn_results, sn_reports,
                    part_no_by_key, part_no_src_by_key, ocr_jobs, part_no_scan_miss_keys,
                )

        # The SN-derived PartNo also contributes family evidence, so a forged
        # 4E repair on another label of the same product is rejected.
        self.assertEqual(observations, [("50087149", "21"), ("98013170", "21")])
        self.assertEqual(part_no_scan_miss_keys, set())
        self.assertEqual(sn_results["s110_repair"][2], "barcode_part_no_conflict")
        for key in ("forged_4e", "forged_21"):
            self.assertEqual(sn_results[key][0], "")
            self.assertEqual(sn_results[key][2], "barcode_part_no_conflict")
            self.assertEqual(sn_results[key][3]["barcode_status"], "part_no_conflict")
            self.assertEqual(sn_reports[key].status, "part_no_conflict")
        self.assertEqual(sn_results["forged_4e"][3]["barcode_part_no_conflict"], "sn_family_conflict")
        self.assertEqual(sn_results["forged_21"][3]["barcode_part_no_conflict"], "part_no_mismatch")
        self.assertEqual([job[1] for job in ocr_jobs], ["forged_4e", "forged_21", "s110_repair"])
        self.assertEqual(sn_results["genuine"][0], "21500871494ERB006054")
        self.assertEqual(sn_results["s380_repair"][0], "4E2616602199")
        self.assertEqual(part_no_by_key["no_part_no"], "98013170")
        self.assertEqual(part_no_src_by_key["no_part_no"], "sn_barcode")

    def test_conflict_counts_as_barcode_failure_and_ocr_recovery(self):
        from huaweiocr.recognition.metrics import new_stats, record_sn_metrics

        stats = new_stats()
        meta = {"barcode_status": "part_no_conflict", "barcode_found": True, "barcode_attempts": 1}
        record_sn_metrics(stats, True, "2150087149LDS4014472", "ocr", meta)
        self.assertEqual(stats["sn_barcode_hits"], 0)
        self.assertEqual(stats["sn_ocr_recoveries"], 1)
        self.assertEqual(stats["sn_barcode_part_no_conflicts"], 1)


class ModelCodeRuleTest(unittest.TestCase):
    def test_accepts_ekit_and_ar_model_shapes(self):
        from huaweiocr.core.extract import MODEL_CODE_ACCEPT_RE

        for code in ("AP162E", "AP361", "AR180PRO", "AR303W", "AR651W-8P", "AR6140E-9G-2AC",
                     "S110-4P1T-BU-V2", "S310S-4P8UN2X", "S380-H8T3ST"):
            self.assertTrue(MODEL_CODE_ACCEPT_RE.fullmatch(code), code)
        for text in ("E3", "S110", "S310S", "MODEL", "2150087149"):
            self.assertIsNone(MODEL_CODE_ACCEPT_RE.fullmatch(text), text)

    def test_ekit_models_are_recognized_from_ocr_text(self):
        self.assertEqual(pipeline.extract_model_from_text("MODEL: S220-48P4X"), "S220-48P4X")
        self.assertEqual(pipeline.extract_model_from_text("Model: AP361 Wi-Fi 6"), "AP361")
        self.assertTrue(pipeline.model_code_is_plausible("AR303W"))


if __name__ == "__main__":
    unittest.main()
