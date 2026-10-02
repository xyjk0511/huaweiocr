import unittest

from huaweiocr.core.sources import SOURCE_LABELS, display_source
from huaweiocr.recognition.metrics import (
    finalize_rates,
    new_stats,
    record_model_metrics,
    record_sn_metrics,
)


class RecognitionMetricsTest(unittest.TestCase):
    def test_model_ocr_verification_with_sn_hint_is_still_ocr_recovery(self):
        for source in ("barcode_visual", "barcode_ocr_consensus"):
            with self.subTest(source=source):
                stats = new_stats()
                record_model_metrics(stats, True, "S380-S8P2T", source + "+sn_hint")
                finalize_rates(stats)
                self.assertEqual(stats["model_success"], 1)
                self.assertEqual(stats["model_barcode_hits"], 0)
                self.assertEqual(stats["model_barcode_hit_rate"], 0.0)
                self.assertEqual(stats["model_ocr_recoveries"], 1)

    def test_sn_consensus_and_fallback_do_not_inflate_barcode_hit_rate(self):
        stats = new_stats()
        for source, status in (
            ("barcode", "selected"),
            ("barcode_ocr_consensus", "parse_failure"),
            ("ocr_color", "decoder_miss"),
        ):
            problem, reason = record_sn_metrics(
                stats, True, "4E25A0170000", source,
                {"barcode_status": status, "barcode_attempts": 2, "barcode_found": True},
            )
            self.assertEqual((problem, reason), (False, ""))
        finalize_rates(stats)
        self.assertEqual(stats["sn_success"], 3)
        self.assertEqual(stats["sn_barcode_hits"], 1)
        self.assertEqual(stats["sn_ocr_recoveries"], 2)
        self.assertAlmostEqual(stats["sn_barcode_hit_rate"], 1 / 3)

    def test_missing_source_photo_does_not_create_a_recognition_attempt(self):
        stats = new_stats()
        self.assertEqual(record_sn_metrics(stats, False, "", "missing", {}), (False, ""))
        self.assertEqual(stats["sn_total"], 0)
        self.assertEqual(stats["sn_problem"], 0)

    def test_failed_sn_preserves_error_accounting(self):
        stats = new_stats()
        problem, reason = record_sn_metrics(
            stats, True, "", "ocr_no_match",
            {"barcode_status": "decoder_miss", "ocr_text_found": True},
        )
        self.assertEqual((problem, reason), (True, "ocr_no_match"))
        self.assertEqual(stats["sn_problem"], 1)
        self.assertEqual(stats["regex_fail"], 1)
        self.assertEqual(stats["barcode_fail"], 1)
        self.assertEqual(stats["ocr_fail"], 0)

    def test_source_labels_keep_evidence_and_hint_in_both_languages(self):
        for lang in ("zh", "en"):
            labels = SOURCE_LABELS[lang]
            with self.subTest(lang=lang):
                self.assertEqual(
                    display_source("barcode_ocr_consensus+sn_hint", labels),
                    labels["source_barcode_ocr_consensus"] + labels["source_sn_hint_suffix"],
                )
                self.assertEqual(display_source("part_no_barcode", labels), labels["source_part_no"])
                self.assertEqual(display_source("custom_source", labels), "custom_source")
