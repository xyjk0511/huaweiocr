import json
import os
import tempfile
import types
import unittest
from unittest import mock

import sn_barcode
import validate_sn_barcodes
from tests.test_locked_output_dirs import _import_scan2


class SnBarcodeSelectionTest(unittest.TestCase):
    def test_non_sn_payloads_are_rejected(self):
        self.assertEqual(sn_barcode.extract_sn_from_payload("SF3260776146675"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("EAN: 6971234567890"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("MAC:E49024187A70"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("Part No: 50087149"), "")

    def test_valid_sn_payload_is_extracted(self):
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("SN:4E25A0170000"),
            "4E25A0170000",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("S/N 21500671494ERA050003"),
            "21500671494ERA050003",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("[)>06 1P50087149 18VLEHWT S21500871494ERB006054"),
            "21500871494ERB006054",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("21500872904ERC000382"),
            "21500872904ERC000382",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("21500872904ERC000382AP"),
            "21500872904ERC000382",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("21500871474ES1016428"),
            "21500871474ES1016428",
        )

    def test_bad_direct_barcode_payloads_are_not_normalized_into_sn(self):
        self.assertEqual(sn_barcode.extract_sn_from_payload("F'500872904ERB000951"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("532570497307251622"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("215BROKEN4ERC000382AP"), "")

    def test_direct_barcode_selection_filters_bad_payloads(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "F'500872904ERB000951", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "21500872904ERB000951", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "532570497307251622", "sn", "sn"),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "21500872904ERB000951")
        self.assertIn("F'500872904ERB000951", report.non_sn_payloads)

    def test_direct_barcode_selection_accepts_short_sn_over_logistics_code(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "532570497307251622", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "4E25B0105906", "sn", "sn"),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25B0105906")

    def test_full_sn_is_preferred_over_short_sn_from_same_source(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "4E25B0105906", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "21500871494ERB006499", "sn", "sn"),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "21500871494ERB006499")

    def test_conflicting_barcode_sns_are_ambiguous(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "SN:4E25A0170001", "sn", "sn.region.1"),
            ]
        )

        self.assertEqual(report.status, "ambiguous")
        self.assertEqual(report.ambiguous_sns, ["4E25A0170000", "4E25A0170001"])

    def test_duplicate_sn_uses_source_priority(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "label", "label"),
                sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "sn", "sn"),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(report.source_region, "sn")

    def test_unique_higher_priority_source_wins_over_original_conflicts(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "SN:21500872884ERA005572", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "SN:21500872884ERA005405", "original", "original"),
                sn_barcode.DecoderResult("fake", "SN:21500872884ERA005765", "original", "original"),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "21500872884ERA005572")
        self.assertEqual(report.source_region, "sn")

    def test_label_source_wins_over_original_region_conflicts(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult(
                    "fake",
                    "SN:21500872884ERA005572",
                    "label",
                    "label.rot0.full",
                ),
                sn_barcode.DecoderResult(
                    "fake",
                    "SN:21500872884ERA005405",
                    "original",
                    "original.rot0.region.1",
                ),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "21500872884ERA005572")
        self.assertEqual(report.source_region, "label.rot0.full")

    def test_attempt_budget_is_isolated_per_source(self):
        candidates_by_source = {
            "sn": [sn_barcode.CandidateImage(object(), "sn", "sn", "raw")],
            "label": [sn_barcode.CandidateImage(object(), "label", "label", "raw")],
        }

        def fake_candidates(_image, source, max_candidates=96):
            return candidates_by_source[source]

        def fake_decode(candidate):
            if candidate.source == "label":
                return [
                    sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "label", "label")
                ], []
            return [], []

        with mock.patch.object(sn_barcode, "_read_image", return_value=object()):
            with mock.patch.object(sn_barcode, "generate_candidate_images", side_effect=fake_candidates):
                with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_decode):
                    report = sn_barcode.scan_sn_barcodes(
                        [("sn", "sn.png"), ("label", "label.png")],
                        max_decoder_attempts=1,
                    )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(report.attempts, 2)


class Scan2BarcodeAccountingTest(unittest.TestCase):
    def test_ambiguous_barcode_is_not_silently_selected(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="ambiguous",
            attempts=2,
            decoded_count=2,
            ambiguous_sns=["4E25A0170000", "4E25A0170001"],
            results=[
                sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "SN:4E25A0170001", "label", "label"),
            ],
        )

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            sn, raw, source, meta = scan2.recognize_sn(
                "sn.png",
                label_id="a__label_1",
                label_path="label.png",
                allow_ocr=False,
            )

        self.assertEqual(sn, "")
        self.assertEqual(source, "barcode_ambiguous")
        self.assertIn("BARCODE_AMBIGUOUS", raw)
        self.assertEqual(meta["barcode_status"], "ambiguous")

    def test_main_disables_original_sn_fallback(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            sn_dir = os.path.join(stage2, "sn")
            model_dir = os.path.join(stage2, "model")
            label_dir = os.path.join(root, "stage1_labels")
            os.makedirs(sn_dir)
            os.makedirs(model_dir)
            os.makedirs(label_dir)
            original_path = os.path.join(root, "photo.jpg")
            open(original_path, "wb").close()

            rows = []
            for index in (1, 2):
                label_id = f"photo.jpg__label_{index}"
                label_path = os.path.join(label_dir, f"{label_id}.png")
                sn_path = os.path.join(sn_dir, f"{label_id}__sn.png")
                open(label_path, "wb").close()
                open(sn_path, "wb").close()
                rows.append(
                    {
                        "label_id": label_id,
                        "label_crop": label_path,
                        "sn_path": sn_path,
                        "original_image_path": original_path,
                    }
                )

            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                for row in rows:
                    manifest.write(json.dumps(row) + "\n")

            meta = {"barcode_found": False, "ocr_text_found": False, "barcode_status": "decoder_miss"}
            with mock.patch.object(scan2, "recognize_sn", return_value=("", "", "none", meta)) as recognize_sn:
                scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

        self.assertEqual(recognize_sn.call_count, 2)
        self.assertEqual([call.kwargs["original_path"] for call in recognize_sn.call_args_list], ["", ""])

    def test_main_reports_barcode_hit_rate_and_ocr_recovery_separately(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "sn_path": sn_path}) + "\n")

            meta = {
                "barcode_found": False,
                "ocr_text_found": True,
                "barcode_status": "decoder_miss",
                "barcode_attempts": 3,
                "barcode_decoded_count": 0,
            }
            with mock.patch.object(scan2, "recognize_sn", return_value=("4E25A0170000", "ocr raw", "ocr", meta)):
                stats = scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

        self.assertEqual(stats["sn_success"], 1)
        self.assertEqual(stats["sn_barcode_hits"], 0)
        self.assertEqual(stats["sn_ocr_recoveries"], 1)
        self.assertEqual(stats["sn_barcode_attempts"], 3)
        self.assertEqual(stats["sn_barcode_hit_rate"], 0.0)


class ValidationCommandTest(unittest.TestCase):
    def test_template_builder_creates_manual_review_rows_without_accepting_them(self):
        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            label_dir = os.path.join(root, "stage1_labels")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(label_dir)
            os.makedirs(sn_dir)
            label_path = os.path.join(label_dir, "a__label_1.png")
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            label_path_2 = os.path.join(label_dir, "a__label_2.png")
            open(label_path, "wb").close()
            open(sn_path, "wb").close()
            open(label_path_2, "wb").close()

            stage2_manifest = os.path.join(stage2, "manifest.jsonl")
            with open(stage2_manifest, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "label_id": "a__label_1",
                    "label_crop": label_path,
                    "sn_path": sn_path,
                    "sn_conf": 0.91,
                }) + "\n")
                manifest.write(json.dumps({
                    "label_id": "a__label_2",
                    "label_crop": label_path_2,
                    "sn_path": None,
                }) + "\n")

            recognized_jsonl = os.path.join(stage2, "recognized.jsonl")
            with open(recognized_jsonl, "w", encoding="utf-8") as recognized:
                recognized.write(json.dumps({
                    "label_id": "a__label_1",
                    "sn": "SN:4E25A0170000",
                    "sn_src": "ocr",
                }) + "\n")

            output_path = os.path.join(root, "validation", "candidates.jsonl")
            summary = validate_sn_barcodes.build_manifest_template_from_stage2(
                stage2_manifest,
                output_path,
                recognized_jsonl=recognized_jsonl,
            )

            with open(output_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(summary["rows_written"], 2)
        self.assertEqual(rows[0]["label_id"], "a__label_1")
        self.assertEqual(rows[0]["expected_sn"], "")
        self.assertFalse(rows[0]["barcode_present"])
        self.assertFalse(rows[0]["accepted_quality"])
        self.assertEqual(rows[0]["pipeline_candidate_sn"], "4E25A0170000")
        self.assertIn("not ground truth", rows[0]["notes"])
        self.assertNotIn("sn_path", rows[1])

    def test_validation_fails_when_accepted_sample_count_is_too_small(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "label.png")
            open(image_path, "wb").close()
            manifest_path = os.path.join(root, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "image_path": image_path,
                    "label_id": "a__label_1",
                    "expected_sn": "4E25A0170000",
                    "barcode_present": True,
                    "accepted_quality": True,
                    "notes": "unit test",
                }) + "\n")

            fake_report = sn_barcode.SnBarcodeReport(
                status="hit",
                sn="4E25A0170000",
                raw_text="SN:4E25A0170000",
                source="label",
                source_region="label",
                decoder_name="fake",
                attempts=1,
                decoded_count=1,
            )
            with mock.patch.object(validate_sn_barcodes, "scan_sn_barcodes", return_value=fake_report):
                summary = validate_sn_barcodes.evaluate_manifest(manifest_path, min_accepted=50)

        self.assertFalse(summary.passed)
        self.assertEqual(summary.exact_hits, 1)
        self.assertIn("below required minimum 50", "\n".join(summary.errors))

    def test_validation_minimum_counts_only_accepted_barcode_rows(self):
        with tempfile.TemporaryDirectory() as root:
            manifest_path = os.path.join(root, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                for index in range(49):
                    image_path = os.path.join(root, f"label_{index}.png")
                    open(image_path, "wb").close()
                    manifest.write(json.dumps({
                        "image_path": image_path,
                        "label_id": f"a__label_{index}",
                        "expected_sn": "4E25A0170000",
                        "barcode_present": True,
                        "accepted_quality": True,
                        "notes": "unit test",
                    }) + "\n")
                non_barcode_path = os.path.join(root, "non_barcode.png")
                open(non_barcode_path, "wb").close()
                manifest.write(json.dumps({
                    "image_path": non_barcode_path,
                    "label_id": "non_barcode",
                    "expected_sn": "",
                    "barcode_present": False,
                    "accepted_quality": True,
                    "notes": "unit test",
                }) + "\n")

            fake_report = sn_barcode.SnBarcodeReport(
                status="hit",
                sn="4E25A0170000",
                raw_text="SN:4E25A0170000",
                source="label",
                source_region="label",
                decoder_name="fake",
                attempts=1,
                decoded_count=1,
            )
            with mock.patch.object(validate_sn_barcodes, "scan_sn_barcodes", return_value=fake_report):
                summary = validate_sn_barcodes.evaluate_manifest(manifest_path, min_accepted=50)

        self.assertFalse(summary.passed)
        self.assertEqual(summary.accepted_quality_rows, 50)
        self.assertEqual(summary.accepted_barcode_rows, 49)
        self.assertEqual(summary.denominator, 49)
        self.assertIn("accepted-quality barcode sample count 49", "\n".join(summary.errors))

    def test_validation_reports_below_threshold_failures(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "label.png")
            open(image_path, "wb").close()
            manifest_path = os.path.join(root, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "image_path": image_path,
                    "label_id": "a__label_1",
                    "expected_sn": "4E25A0170000",
                    "barcode_present": True,
                    "accepted_quality": True,
                    "notes": "unit test",
                }) + "\n")

            fake_report = sn_barcode.SnBarcodeReport(status="decoder_miss", attempts=1)
            with mock.patch.object(validate_sn_barcodes, "scan_sn_barcodes", return_value=fake_report):
                summary = validate_sn_barcodes.evaluate_manifest(
                    manifest_path,
                    threshold=0.90,
                    min_accepted=1,
                )

        self.assertFalse(summary.passed)
        self.assertEqual(summary.failure_counts["decoder_miss"], 1)
        self.assertIn("below threshold", "\n".join(summary.errors))


if __name__ == "__main__":
    unittest.main()
