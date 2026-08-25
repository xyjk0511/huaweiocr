import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2
import numpy as np

import barcode as barcode_module
import sn_barcode
import validate_sn_barcodes
from tests.test_locked_output_dirs import _import_scan2


class BarcodeDecoderReliabilityTest(unittest.TestCase):
    def test_barcode_module_import_survives_missing_pyzbar(self):
        module_name = "barcode_without_pyzbar_test"
        # barcode.py 已是别名垫片，导入期行为要从实现文件本身验证。
        module_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "huaweiocr", "barcode", "generic.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)

        with mock.patch.dict(
            sys.modules,
            {"cv2": cv2, "numpy": np, "pyzbar": None, "pyzbar.pyzbar": None},
        ):
            try:
                spec.loader.exec_module(module)
                errors = []
                img = np.full((40, 160), 255, dtype=np.uint8)
                self.assertEqual(module.decode_with_transforms(img, "unit", decoder_errors=errors), [])
            finally:
                sys.modules.pop(module_name, None)

        self.assertTrue(any(err.startswith("decoder_unavailable:pyzbar:") for err in errors))

    def test_sn_barcode_read_image_prefers_unicode_safe_path_for_non_ascii_filename(self):
        path = r"F:\HuaweiOCR\stage2_fields\sn\滑坳掳.jpg__label_1__sn.png"
        sentinel = object()

        with mock.patch.object(sn_barcode.np, "fromfile", return_value=b"123") as fromfile:
            with mock.patch.object(sn_barcode.cv2, "imdecode", return_value=sentinel, create=True) as imdecode:
                with mock.patch.object(
                    sn_barcode.cv2,
                    "imread",
                    side_effect=AssertionError("non-ascii path should bypass cv2.imread first"),
                ):
                    self.assertIs(sn_barcode._read_image(path), sentinel)

        fromfile.assert_called_once_with(path, dtype=sn_barcode.np.uint8)
        imdecode.assert_called_once()

    def test_cli_infra_failures_are_reported_separately_from_decoder_miss(self):
        cases = [
            (
                FileNotFoundError("missing exe"),
                "BarcodeReaderCLI_file_not_found:",
            ),
            (
                subprocess.TimeoutExpired(["BarcodeReaderCLI.exe"], 0.25),
                "BarcodeReaderCLI_timeout:",
            ),
            (
                subprocess.CompletedProcess(
                    ["BarcodeReaderCLI.exe"],
                    7,
                    stdout="",
                    stderr="license failure",
                ),
                "BarcodeReaderCLI_error:returncode=7 stderr=license failure",
            ),
        ]

        for side_effect, expected in cases:
            with self.subTest(expected=expected):
                errors = []
                barcode_module._CLI_UNAVAILABLE = False
                with mock.patch.object(barcode_module, "BARCODE_CLI_PATH", "BarcodeReaderCLI.exe"):
                    with mock.patch.object(barcode_module.os.path, "exists", return_value=True):
                        if isinstance(side_effect, subprocess.CompletedProcess):
                            run_patch = mock.patch.object(barcode_module, "_run_cli", return_value=side_effect)
                        else:
                            run_patch = mock.patch.object(barcode_module, "_run_cli", side_effect=side_effect)
                        with run_patch:
                            self.assertEqual(
                                barcode_module.read_barcodes_cli("image.png", decoder_errors=errors),
                                [],
                            )

                self.assertTrue(
                    any(err.startswith(expected) or err == expected for err in errors),
                    errors,
                )

    def test_sn_cli_decoder_propagates_barcode_module_decoder_errors(self):
        candidate = sn_barcode.CandidateImage(
            np.full((40, 160), 255, dtype=np.uint8),
            "sn",
            "sn.rot0.full",
            "raw",
        )

        def fake_decode_with_cli(_image, _tag, decoder_errors=None):
            if decoder_errors is not None:
                decoder_errors.append("BarcodeReaderCLI_timeout:0.25s")
            return []

        fake_barcode_module = types.SimpleNamespace(
            decode_with_cli=fake_decode_with_cli,
            decode_cli_multi=lambda *_args, **_kwargs: [],
        )
        with mock.patch.dict(sys.modules, {"barcode": fake_barcode_module}):
            results, errors = sn_barcode._decode_cli(candidate)

        self.assertEqual(results, [])
        self.assertIn("BarcodeReaderCLI_timeout:0.25s", errors)


class SnBarcodeSelectionTest(unittest.TestCase):
    def test_sn_base_candidates_insert_focus_bands_before_full_image(self):
        img = np.full((240, 640, 3), 255, dtype=np.uint8)

        bases = sn_barcode._base_candidate_images(img, "sn", max_bases=4)

        self.assertGreaterEqual(len(bases), 4)
        self.assertTrue(bases[0][0].startswith("sn.rot0.focus."))
        self.assertTrue(bases[1][0].startswith("sn.rot0.focus."))
        self.assertTrue(bases[2][0].startswith("sn.rot0.focus."))
        self.assertEqual(bases[3][0], "sn.rot0.full")

    def test_scan_sn_barcodes_tries_sn_focus_band_before_full_ean_noise(self):
        img = np.full((240, 640, 3), 255, dtype=np.uint8)
        seen_regions = []

        def fake_decode(candidate):
            seen_regions.append(candidate.source_region)
            if ".focus." in candidate.source_region:
                return [
                    sn_barcode.DecoderResult(
                        decoder_name="pyzbar",
                        raw_text="21500872904ERC000340",
                        source=candidate.source,
                        source_region=candidate.source_region,
                        barcode_type="CODE128",
                    )
                ], []
            if candidate.source_region.endswith(".full"):
                return [
                    sn_barcode.DecoderResult(
                        decoder_name="pyzbar",
                        raw_text="6901443456987",
                        source=candidate.source,
                        source_region=candidate.source_region,
                        barcode_type="EAN13",
                    )
                ], []
            return [], []

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                    with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_decode):
                        report = sn_barcode.scan_sn_barcodes(
                            [("sn", "sn.png")],
                            max_decoder_attempts=1,
                        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "21500872904ERC000340")
        self.assertTrue(seen_regions)
        self.assertIn(".focus.", seen_regions[0])

    def test_code128_visual_verifier_accepts_matching_payload(self):
        bits = barcode_module._code128b_module_bits("AP162E")
        modules = np.repeat(bits, 2)
        img = np.full((70, modules.size + 80), 255, dtype=np.uint8)
        dark_cols = modules > 0
        img[15:55, 40:40 + modules.size][:, dark_cols] = 0
        img = cv2.GaussianBlur(img, (3, 3), 0)

        result = barcode_module.verify_code128b_text_in_image(img, "AP162E")

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["score"], 0.68)
        self.assertIsNone(barcode_module.verify_code128b_text_in_image(img, "AP362E"))

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
            sn_barcode.extract_sn_from_payload("4E2630067512"),
            "4E2630067512",
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
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("S/N: 2150010843LDRC000793"),
            "2150010843LDRC000793",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("SN: 2150087147LDS4024590"),
            "2150087147LDS4024590",
        )

    def test_ss_series_payload_is_extracted_from_direct_and_structured_barcode(self):
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("21500108452SS1500363"),
            "21500108452SS1500363",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload(
                "[)>06 1P50010843 18VLEHWT S21500108432SS4500577"
            ),
            "21500108432SS4500577",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("21500772777SS4049560"),
            "",
        )

    def test_agqa_series_payload_is_accepted_by_sn_rules(self):
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("2150087144AGQA001288"),
            "2150087144AGQA001288",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("2150087144AGQC000131"),
            "2150087144AGQC000131",
        )

    def test_structured_serial_field_payload_extracts_short_sn_without_leading_s(self):
        self.assertEqual(
            sn_barcode.extract_sn_from_payload(
                "[)>06 1P98012125 18VLEHWT S4E2630067512"
            ),
            "4E2630067512",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("[)>061P9801240318VLEHWTS4E2610074724"),
            "4E2610074724",
        )

    def test_bad_direct_barcode_payloads_are_not_normalized_into_sn(self):
        self.assertEqual(sn_barcode.extract_sn_from_payload("F'500872904ERB000951"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("532570497307251622"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("20457271304ERA904527"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("215BROKEN4ERC000382AP"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("AASCMAS7SN2150010843"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("MSN2150087147LDS4023"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("4ERA005537EA"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("21500872884ERA007680EAN"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("4E25MODELS38"), "")

    def test_part_and_ean_payloads_are_not_accepted_as_sn(self):
        self.assertEqual(sn_barcode.extract_sn_from_payload("98012125"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("6901443421480"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("Part No: 98012125"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("EAN: 6901443421480"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("S4E2630067512EXTRA"), "")

    def test_ocr_sn_extraction_rejects_user_reported_prefix_junk(self):
        scan2 = _import_scan2()

        self.assertEqual(scan2.extract_sn_from_text("AASCMAS7SN2150010843"), "")
        self.assertEqual(scan2.extract_sn_from_text("MSN2150087147LDS4023"), "")
        self.assertEqual(scan2.extract_sn_from_text("20457271304ERA904527"), "")
        self.assertEqual(scan2.extract_sn_from_text("4ERA005537EA"), "")
        self.assertEqual(
            scan2.extract_sn_from_text("21500872884ERA007680EAN"),
            "21500872884ERA007680",
        )
        self.assertEqual(
            scan2.extract_sn_from_text("S/N: 2150010843LDRC000793"),
            "2150010843LDRC000793",
        )
        self.assertEqual(
            scan2.extract_sn_from_text("SN: 2150087147LDS4024590"),
            "2150087147LDS4024590",
        )
        self.assertEqual(
            scan2.extract_sn_from_text("S/N:21500872884ERB011934 966515"),
            "21500872884ERB011934",
        )
        self.assertEqual(
            scan2.extract_sn_from_text("S/N : 21500872884ERB011934 515996"),
            "21500872884ERB011934",
        )
        self.assertEqual(
            scan2.extract_sn_from_text("S/N: 21500108432SS4500060"),
            "21500108432SS4500060",
        )
        self.assertEqual(
            scan2.extract_sn_from_text("S/N: 21500772777SS4049560"),
            "",
        )

    def test_model_barcode_candidate_rejects_sn_payloads(self):
        scan2 = _import_scan2()

        self.assertEqual(scan2.extract_model_from_barcode_candidate("2150087147LDS4024590"), "")
        self.assertEqual(scan2.extract_model_from_barcode_candidate("2150010843LDRC000793"), "")
        self.assertEqual(scan2.extract_model_from_barcode_candidate("AP362E"), "AP362E")

    def test_model_barcode_band_fallback_can_override_invalid_numeric_scan(self):
        scan2 = _import_scan2()

        with mock.patch.object(
            scan2,
            "_scan_barcode_sources",
            return_value=[{"source": "model", "data": "0227000540"}],
        ):
            with mock.patch.object(
                scan2,
                "_scan_model_barcode_band_entries",
                return_value=[{"source": "model_band", "data": "AP362E"}],
            ):
                model, raw = scan2.try_model_from_barcode("model.png")

        self.assertEqual(model, "AP362E")
        self.assertEqual(raw, "AP362E")

    def test_model_barcode_band_second_scan_can_find_lower_model_code(self):
        scan2 = _import_scan2()
        scan2_barcode = sys.modules["barcode"]
        fake_img = np.full((80, 220, 3), 255, dtype=np.uint8)
        detected_band = np.full((24, 180), 255, dtype=np.uint8)
        midband = np.full((70, 220), 255, dtype=np.uint8)

        def fake_small_patch(img_bgr):
            if img_bgr.shape[0] == detected_band.shape[0]:
                return {"results": [{"data": "50087289"}]}
            return {"results": [{"data": "AP162E"}]}

        with mock.patch.object(scan2, "_try_model_from_fast_zxing", return_value=("", "")):
            with mock.patch.object(
                scan2,
                "_scan_barcode_sources",
                return_value=[{"source": "model", "data": "50087289"}],
            ):
                with mock.patch.object(scan2, "_read_image", return_value=fake_img):
                    with mock.patch.object(scan2.cv2, "cvtColor", side_effect=lambda img, *_args, **_kwargs: img, create=True):
                        with mock.patch.object(scan2.cv2, "COLOR_GRAY2BGR", 2, create=True):
                            with mock.patch.object(scan2_barcode, "auto_rotate_to_horizontal", side_effect=lambda img: img, create=True):
                                with mock.patch.object(
                                    scan2_barcode,
                                    "crop_detected_barcode_band",
                                    return_value=detected_band,
                                    create=True,
                                ):
                                    with mock.patch.object(
                                        scan2_barcode,
                                        "crop_bar_band",
                                        return_value=midband,
                                        create=True,
                                    ):
                                        with mock.patch.object(
                                            scan2_barcode,
                                            "decode_cli_sharp_variants",
                                            return_value=[],
                                            create=True,
                                        ):
                                            with mock.patch.object(
                                                scan2,
                                                "decode_small_patch",
                                                side_effect=fake_small_patch,
                                            ):
                                                model, raw = scan2.try_model_from_barcode("model.png")

        self.assertEqual(model, "AP162E")
        self.assertEqual(raw, "AP162E")

    def test_model_barcode_skips_band_fallback_after_direct_hit(self):
        scan2 = _import_scan2()

        with mock.patch.object(
            scan2,
            "_scan_barcode_sources",
            return_value=[{"source": "model", "data": "AP362E"}],
        ):
            with mock.patch.object(scan2, "_scan_model_barcode_band_entries") as fallback:
                model, raw = scan2.try_model_from_barcode("model.png")

        self.assertEqual(model, "AP362E")
        self.assertEqual(raw, "AP362E")
        fallback.assert_not_called()

    def test_model_barcode_skips_heavy_scan_after_fast_zxing_hit(self):
        scan2 = _import_scan2()

        with mock.patch.object(scan2, "_try_model_from_fast_zxing", return_value=("AP362E", "AP362E")):
            with mock.patch.object(scan2, "_scan_barcode_sources") as heavy_scan:
                model, raw = scan2.try_model_from_barcode("model.png")

        self.assertEqual(model, "AP362E")
        self.assertEqual(raw, "AP362E")
        heavy_scan.assert_not_called()

    def test_model_recognition_marks_visual_barcode_match(self):
        scan2 = _import_scan2()

        with mock.patch.object(scan2, "try_model_from_barcode", return_value=("", "")):
            with mock.patch.object(
                scan2,
                "ocr_text_with_details",
                return_value=("Model: AP162E", "Model: AP162E", []),
            ):
                with mock.patch.object(
                    scan2,
                    "_verify_model_barcode_visual",
                    return_value={"text": "AP162E", "score": 0.76},
                ):
                    with mock.patch.object(scan2, "load_for_ocr_color") as color_mock:
                        model, raw, source = scan2.recognize_model("model.png", use_barcode=True)

        self.assertEqual(model, "AP162E")
        self.assertIn("[BARCODE_VISUAL] AP162E", raw)
        self.assertEqual(source, "barcode_visual")
        color_mock.assert_not_called()

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

    def test_sources_are_interleaved_before_exhausting_sn_candidates(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        candidates_by_source = {
            "sn": [
                sn_barcode.CandidateImage(img, "sn", f"sn.{index}", "raw")
                for index in range(5)
            ],
            "label": [sn_barcode.CandidateImage(img, "label", "label.0", "raw")],
        }
        decode_order = []

        def fake_candidates(_image, source, max_candidates=96):
            return candidates_by_source[source]

        def fake_decode(candidate):
            decode_order.append(candidate.source_region)
            if candidate.source == "label":
                return [
                    sn_barcode.DecoderResult(
                        "fake",
                        "SN:4E25A0170000",
                        "label",
                        candidate.source_region,
                    )
                ], []
            return [], []

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", side_effect=fake_candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_decode):
                            report = sn_barcode.scan_sn_barcodes(
                                [("sn", "sn.png"), ("label", "label.png")],
                                max_decoder_attempts=96,
                            )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(report.attempts, 2)
        self.assertEqual(decode_order[:2], ["sn.0", "label.0"])

    def test_label_source_is_not_initialized_when_sn_hits_first(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        read_paths = []

        def fake_read(path):
            read_paths.append(path)
            return img

        def fake_candidates(_image, source, max_candidates=96):
            self.assertEqual(source, "sn")
            return [sn_barcode.CandidateImage(img, "sn", "sn.0", "raw")]

        def fake_decode(candidate):
            return [
                sn_barcode.DecoderResult(
                    "fake",
                    "SN:4E25A0170000",
                    "sn",
                    candidate.source_region,
                )
            ], []

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar"}):
            with mock.patch.object(sn_barcode, "_read_image", side_effect=fake_read):
                with mock.patch.object(sn_barcode, "generate_candidate_images", side_effect=fake_candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_decode):
                            report = sn_barcode.scan_sn_barcodes(
                                [("sn", "sn.png"), ("label", "label.png")],
                                max_decoder_attempts=96,
                            )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(read_paths, ["sn.png"])

    def test_prior_source_fallback_preempts_lower_source_ambiguity(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        candidates_by_source = {
            "sn": [sn_barcode.CandidateImage(img, "sn", "sn.0", "raw")],
            "label": [sn_barcode.CandidateImage(img, "label", "label.0", "raw")],
        }

        def fake_candidates(_image, source, max_candidates=96):
            return candidates_by_source[source]

        def fake_decode(candidate):
            if candidate.source == "label":
                return [
                    sn_barcode.DecoderResult("fake", "SN:4E25A0170001", "label", "label.0"),
                    sn_barcode.DecoderResult("fake", "SN:4E25A0170002", "label", "label.0"),
                ], []
            return [], []

        def fake_fallback(path):
            if path == "sn.png":
                return ["SN:4E25A0170000"]
            return []

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", side_effect=fake_candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_decode):
                            report = sn_barcode.scan_sn_barcodes(
                                [("sn", "sn.png"), ("label", "label.png")],
                                fallback_path_decoder=fake_fallback,
                                max_decoder_attempts=96,
                            )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(report.source, "sn")
        self.assertEqual(report.attempts, 2)

    def test_sn_path_fallback_runs_before_label_source_after_first_sn_miss(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        read_paths = []
        fallback_paths = []

        def fake_read(path):
            read_paths.append(path)
            return img

        def fake_candidates(_image, source, max_candidates=96):
            return [sn_barcode.CandidateImage(img, source, f"{source}.0", "raw")]

        def fake_fallback(path):
            fallback_paths.append(path)
            if path == "sn.png":
                return ["SN:4E25A0170000"]
            return []

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar"}):
            with mock.patch.object(sn_barcode, "_read_image", side_effect=fake_read):
                with mock.patch.object(sn_barcode, "generate_candidate_images", side_effect=fake_candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", return_value=([], [])):
                            report = sn_barcode.scan_sn_barcodes(
                                [("sn", "sn.png"), ("label", "label.png")],
                                fallback_path_decoder=fake_fallback,
                                max_decoder_attempts=96,
                            )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(report.attempts, 2)
        self.assertEqual(read_paths, ["sn.png"])
        self.assertEqual(fallback_paths, ["sn.png"])

    def test_sn_pixel_repair_runs_before_label_source_after_first_sn_miss(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        read_paths = []
        template_specs = []

        def fake_read(path):
            read_paths.append(path)
            return img

        def fake_candidates(_image, source, max_candidates=96):
            return [sn_barcode.CandidateImage(img, source, f"{source}.0", "raw")]

        def fake_fallback(_path):
            return []

        fake_candidate = types.SimpleNamespace(text="4E2630137607", score=0.12)
        fake_repair = types.SimpleNamespace(
            parse_templates=lambda spec: template_specs.append(spec) or [[[]]],
            compile_regex_or_empty=lambda pattern: pattern,
            decode_image=lambda *_args, **_kwargs: ([fake_candidate], [], "template"),
        )

        with mock.patch.dict(
            os.environ,
            {
                "SN_BARCODE_DECODERS": "pyzbar",
                "SN_BARCODE_REPAIR_TEMPLATE": "4E26########",
                "SN_BARCODE_REPAIR_ACCEPT_SCORE": "0.24",
            },
        ):
            with mock.patch.dict(sys.modules, {"linear_barcode_repair": fake_repair}):
                with mock.patch.object(sn_barcode.os.path, "exists", return_value=True):
                    with mock.patch.object(sn_barcode, "_read_image", side_effect=fake_read):
                        with mock.patch.object(sn_barcode, "generate_candidate_images", side_effect=fake_candidates):
                            with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                                with mock.patch.object(sn_barcode, "_decode_pyzbar", return_value=([], [])):
                                    report = sn_barcode.scan_sn_barcodes(
                                        [("sn", "sn.png"), ("label", "label.png")],
                                        fallback_path_decoder=fake_fallback,
                                        max_decoder_attempts=96,
                                    )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E2630137607")
        self.assertEqual(report.decoder_name, "code128_pixel_repair")
        self.assertEqual(report.source_region, "sn.code128_pixel_repair")
        self.assertEqual(read_paths, ["sn.png"])
        self.assertEqual(template_specs, ["4E26########"])

    def test_direct_unknown_sn_defers_pixel_repair_for_ocr_consensus(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        unknown_sn = "21500108452TC1500363"
        direct_result = sn_barcode.DecoderResult(
            "zxingcpp",
            unknown_sn,
            "sn",
            "sn.0",
        )

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(
                    sn_barcode,
                    "generate_candidate_images",
                    return_value=[sn_barcode.CandidateImage(img, "sn", "sn.0", "raw")],
                ):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(
                            sn_barcode,
                            "_decode_pyzbar",
                            return_value=([direct_result], []),
                        ):
                            with mock.patch.object(
                                sn_barcode,
                                "_pixel_repair_decode_path",
                                return_value=(
                                    [
                                        sn_barcode.DecoderResult(
                                            "code128_pixel_repair",
                                            "21500108452ES1500363",
                                            "sn",
                                            "sn.code128_pixel_repair",
                                        )
                                    ],
                                    1,
                                    [],
                                ),
                            ) as repair:
                                report = sn_barcode.scan_sn_barcodes(
                                    [("sn", "sn.png")],
                                    fallback_path_decoder=lambda _path: [],
                                    max_decoder_attempts=96,
                                )

        self.assertEqual(report.status, "parse_failure")
        self.assertIn(unknown_sn, report.failed_payloads)
        self.assertTrue(any(result.raw_text == unknown_sn for result in report.results))
        repair.assert_not_called()

    def test_scan_does_not_materialize_candidates_beyond_attempt_budget(self):
        img = np.full((80, 240, 3), 255, dtype=np.uint8)
        resize_calls = 0

        def fake_resize(image, scale):
            nonlocal resize_calls
            resize_calls += 1
            return image

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(
                    sn_barcode,
                    "_base_candidate_images",
                    return_value=[("sn.rot0.full", img, 0, None)],
                ):
                    with mock.patch.object(sn_barcode, "_resize", side_effect=fake_resize):
                        with mock.patch.object(sn_barcode, "_deskew", side_effect=lambda image, _angle: image):
                            with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                                with mock.patch.object(sn_barcode, "_decode_pyzbar", return_value=([], [])):
                                    report = sn_barcode.scan_sn_barcodes(
                                        [("sn", "sn.png")],
                                        max_candidates=96,
                                        max_decoder_attempts=1,
                                    )

        self.assertEqual(report.status, "decoder_miss")
        self.assertEqual(report.attempts, 1)
        self.assertEqual(resize_calls, 1)

    def test_label_source_with_sn_uses_independent_decoder_attempt_cap(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        candidates_by_source = {
            "sn": [sn_barcode.CandidateImage(img, "sn", "sn.0", "raw")],
            "label": [
                sn_barcode.CandidateImage(img, "label", f"label.{i}", "raw")
                for i in range(10)
            ],
        }
        calls_by_source = {"sn": 0, "label": 0}

        def fake_candidates(_image, source, max_candidates=96):
            return candidates_by_source[source]

        def fake_decode(candidate):
            calls_by_source[candidate.source] += 1
            return [], []

        with mock.patch.dict(
            os.environ,
            {
                "SN_BARCODE_LABEL_MAX_DECODER_ATTEMPTS": "4",
                "SN_BARCODE_DECODERS": "pyzbar,zxingcpp,cli",
            },
        ):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", side_effect=fake_candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_decode):
                            with mock.patch.object(sn_barcode, "_decode_zxingcpp", side_effect=fake_decode):
                                with mock.patch.object(sn_barcode, "_decode_cli", side_effect=fake_decode):
                                    report = sn_barcode.scan_sn_barcodes(
                                        [("sn", "sn.png"), ("label", "label.png")],
                                        max_decoder_attempts=96,
                                    )

        self.assertEqual(report.status, "decoder_miss")
        self.assertEqual(report.attempts, 7)
        self.assertEqual(calls_by_source["sn"], 3)
        self.assertEqual(calls_by_source["label"], 4)

    def test_default_sn_barcode_decoders_skip_cli(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        candidates = [sn_barcode.CandidateImage(img, "sn", "sn.0", "raw")]

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", return_value=candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", return_value=([], [])):
                            with mock.patch.object(sn_barcode, "_decode_zxingcpp", return_value=([], [])):
                                with mock.patch.object(sn_barcode, "_decode_cli", return_value=([], [])) as decode_cli:
                                    report = sn_barcode.scan_sn_barcodes(
                                        [("sn", "sn.png")],
                                        max_decoder_attempts=6,
                                    )

        self.assertEqual(report.status, "decoder_miss")
        self.assertEqual(report.attempts, 2)
        decode_cli.assert_not_called()

    def test_label_source_without_sn_uses_full_decoder_attempt_budget(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        candidates = [
            sn_barcode.CandidateImage(img, "label", f"label.{i}", "raw")
            for i in range(10)
        ]

        with mock.patch.dict(os.environ, {"SN_BARCODE_LABEL_MAX_DECODER_ATTEMPTS": "4"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", return_value=candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", return_value=([], [])):
                            with mock.patch.object(sn_barcode, "_decode_zxingcpp", return_value=([], [])):
                                with mock.patch.object(sn_barcode, "_decode_cli", return_value=([], [])):
                                    report = sn_barcode.scan_sn_barcodes(
                                        [("label", "label.png")],
                                        max_decoder_attempts=7,
                                    )

        self.assertEqual(report.status, "decoder_miss")
        self.assertEqual(report.attempts, 7)

    def test_sn_source_keeps_full_decoder_attempt_budget(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        candidates = [
            sn_barcode.CandidateImage(img, "sn", f"sn.{i}", "raw")
            for i in range(10)
        ]

        with mock.patch.dict(os.environ, {"SN_BARCODE_LABEL_MAX_DECODER_ATTEMPTS": "4"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", return_value=candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", return_value=([], [])):
                            with mock.patch.object(sn_barcode, "_decode_zxingcpp", return_value=([], [])):
                                with mock.patch.object(sn_barcode, "_decode_cli", return_value=([], [])):
                                    report = sn_barcode.scan_sn_barcodes(
                                        [("sn", "sn.png")],
                                        max_decoder_attempts=7,
                                    )

        self.assertEqual(report.status, "decoder_miss")
        self.assertEqual(report.attempts, 7)

class SnBarcodeFailedPayloadSuppressionTest(unittest.TestCase):
    def test_repeated_parse_failure_payload_is_reported_and_suppresses_later_decoders(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        payload = "50087149"
        candidates = [
            sn_barcode.CandidateImage(img, "sn", f"sn.{index}", "raw")
            for index in range(5)
        ]
        calls = []

        def fake_decode(name):
            def decode(candidate):
                calls.append((name, candidate.source_region))
                return [
                    sn_barcode.DecoderResult(name, payload, "sn", candidate.source_region)
                ], []
            return decode

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar,zxingcpp,cli"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", return_value=candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_decode("pyzbar")):
                            with mock.patch.object(sn_barcode, "_decode_zxingcpp", side_effect=fake_decode("zxingcpp")):
                                with mock.patch.object(sn_barcode, "_decode_cli", side_effect=fake_decode("cli")):
                                    report = sn_barcode.scan_sn_barcodes(
                                        [("sn", "sn.png")],
                                        max_decoder_attempts=15,
                                    )

        self.assertEqual(report.status, "parse_failure")
        self.assertEqual(report.failed_payloads, [payload])
        self.assertEqual(report.non_sn_payloads, [payload])
        self.assertEqual(report.attempts, len(calls))
        self.assertLess(len(calls), 15)
        self.assertLessEqual(len(calls), 7)
        self.assertEqual(calls[:3], [("pyzbar", "sn.0"), ("zxingcpp", "sn.0"), ("cli", "sn.0")])
        self.assertEqual([name for name, region in calls if region == "sn.1"], ["pyzbar"])

    def test_known_non_sn_payload_skips_first_candidate_and_preserves_budget_for_sn(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        candidates = [
            sn_barcode.CandidateImage(img, "sn", "sn.0", "raw"),
            sn_barcode.CandidateImage(img, "sn", "sn.1", "raw"),
        ]
        pyzbar_regions = []
        zxing_regions = []

        def fake_pyzbar(candidate):
            pyzbar_regions.append(candidate.source_region)
            raw = "50087149" if candidate.source_region == "sn.0" else "SN:4E25A0170000"
            return [sn_barcode.DecoderResult("pyzbar", raw, "sn", candidate.source_region)], []

        def fake_zxing(candidate):
            zxing_regions.append(candidate.source_region)
            return [sn_barcode.DecoderResult("zxingcpp", "50087149", "sn", candidate.source_region)], []

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar,zxingcpp"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", return_value=candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_pyzbar):
                            with mock.patch.object(sn_barcode, "_decode_zxingcpp", side_effect=fake_zxing):
                                report = sn_barcode.scan_sn_barcodes(
                                    [("sn", "sn.png")],
                                    max_decoder_attempts=2,
                                    known_non_sn_payloads={"50087149"},
                                )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(report.raw_text, "SN:4E25A0170000")
        self.assertEqual(report.failed_payloads, ["50087149"])
        self.assertEqual(report.attempts, 2)
        self.assertEqual(report.decoded_count, 1)
        self.assertEqual(pyzbar_regions, ["sn.0", "sn.1"])
        self.assertEqual(zxing_regions, [])

    def test_valid_sn_payload_is_not_suppressed(self):
        img = np.full((50, 120, 3), 255, dtype=np.uint8)
        candidates = [sn_barcode.CandidateImage(img, "sn", "sn.0", "raw")]
        pyzbar_calls = []
        zxing_calls = []

        def fake_pyzbar(candidate):
            pyzbar_calls.append(candidate.source_region)
            return [
                sn_barcode.DecoderResult("pyzbar", "SN:21500872904ERC000340", "sn", candidate.source_region)
            ], []

        def fake_zxing(candidate):
            zxing_calls.append(candidate.source_region)
            return [], []

        with mock.patch.dict(os.environ, {"SN_BARCODE_DECODERS": "pyzbar,zxingcpp"}):
            with mock.patch.object(sn_barcode, "_read_image", return_value=img):
                with mock.patch.object(sn_barcode, "generate_candidate_images", return_value=candidates):
                    with mock.patch.object(sn_barcode, "diagnose_quality", return_value=[]):
                        with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_pyzbar):
                            with mock.patch.object(sn_barcode, "_decode_zxingcpp", side_effect=fake_zxing):
                                report = sn_barcode.scan_sn_barcodes(
                                    [("sn", "sn.png")],
                                    max_decoder_attempts=2,
                                    known_non_sn_payloads={"50087149"},
                                )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "21500872904ERC000340")
        self.assertEqual(report.failed_payloads, [])
        self.assertEqual(report.attempts, 1)
        self.assertEqual(report.decoded_count, 1)
        self.assertEqual(pyzbar_calls, ["sn.0"])
        self.assertEqual(zxing_calls, [])

class Scan2BarcodeAccountingTest(unittest.TestCase):
    def test_part_no_text_maps_to_model_template(self):
        scan2 = _import_scan2()

        model, part_no = scan2.model_from_part_no_text(
            "[)>06 1P50087147 S2150087147LDS4024590"
        )

        self.assertEqual(part_no, "50087147")
        self.assertEqual(model, "AP362E")

    def test_current_611_part_no_products_map_to_models(self):
        scan2 = _import_scan2()

        cases = {
            "50087144": "AP265E",
            "50087289": "AP162E",
            "50010838": "AR180",
            "50010845": "AR280",
            "98012403": "S110-5T",
            "98012406": "S110-8P1T",
        }

        for part_no, expected_model in cases.items():
            with self.subTest(part_no=part_no):
                model, parsed_part_no = scan2.model_from_part_no_text(f"Part No: {part_no}")
                self.assertEqual(parsed_part_no, part_no)
                self.assertEqual(model, expected_model)

    def test_ocr_fallbacks_are_default_enabled(self):
        scan2 = _import_scan2()

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(scan2.scan_ocr_fallback_enabled())
            self.assertTrue(scan2.part_no_ocr_fallback_enabled())

        with mock.patch.dict(os.environ, {"SCAN2_OCR_FALLBACK": "0"}, clear=True):
            self.assertFalse(scan2.scan_ocr_fallback_enabled())
            self.assertFalse(scan2.part_no_ocr_fallback_enabled())

        with mock.patch.dict(
            os.environ,
            {"SCAN2_OCR_FALLBACK": "0", "SCAN2_PART_NO_OCR_FALLBACK": "1"},
            clear=True,
        ):
            self.assertFalse(scan2.scan_ocr_fallback_enabled())
            self.assertTrue(scan2.part_no_ocr_fallback_enabled())

    def test_main_preserves_unmapped_manifest_part_no_in_output(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            manifest_path = os.path.join(stage2, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(
                    json.dumps(
                        {
                            "label_id": "a__label_1",
                            "part_no": "98099999",
                            "part_no_codes": ["98099999"],
                        }
                    )
                    + "\n"
                )

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.dict(
                os.environ,
                {
                    "SCAN2_PARALLEL": "0",
                    "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
                    "SCAN2_PART_NO_MODEL_MAP_PATH": os.path.join(root, "empty_map.json"),
                },
            ):
                scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=out_jsonl,
                    debug_log=os.path.join(root, "debug.log"),
                )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(rows[0]["part_no"], "98099999")
        self.assertEqual(rows[0]["model"], "")

    def test_part_no_scan_tries_upscaled_candidates_after_miss(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            part_no_crop = os.path.join(root, "part_no.png")
            img = np.full((80, 160, 3), 255, dtype=np.uint8)
            cv2.imwrite(part_no_crop, img)

            with mock.patch.object(scan2, "read_barcodes", return_value=[]) as read_barcodes:
                with mock.patch.object(scan2, "_read_image", return_value=img):
                    with mock.patch.object(scan2.cv2, "copyMakeBorder", return_value=img, create=True):
                        with mock.patch.object(scan2.cv2, "resize", return_value=img, create=True):
                            with mock.patch.object(
                                scan2,
                                "decode_barcodes_with_dbr",
                                return_value=["50087149"],
                            ) as decode_resized:
                                entries = scan2._scan_barcode_sources(
                                    [("part_no", part_no_crop)],
                                    field="PART_NO",
                                )

        self.assertEqual(entries, [{"source": "part_no", "data": "50087149"}])
        read_barcodes.assert_called_once_with(part_no_crop)
        decode_resized.assert_called()

    def test_part_no_scan_uses_stripe_rescue_after_other_decoders_miss(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            part_no_crop = os.path.join(root, "part_no.png")
            img = np.full((80, 160, 3), 255, dtype=np.uint8)
            cv2.imwrite(part_no_crop, img)

            with mock.patch.object(scan2, "read_barcodes", return_value=[]):
                with mock.patch.object(scan2, "_read_image", return_value=img):
                    with mock.patch.object(scan2.cv2, "copyMakeBorder", return_value=img, create=True):
                        with mock.patch.object(scan2.cv2, "resize", return_value=img, create=True):
                            with mock.patch.object(scan2, "decode_barcodes_with_dbr", return_value=[]):
                                with mock.patch.object(scan2, "_part_no_pixel_repair", return_value=[]):
                                    with mock.patch.object(
                                        scan2,
                                        "_part_no_stripe_rescue",
                                        return_value=["50087288"],
                                    ) as stripe_rescue:
                                        codes = scan2.read_part_no_barcodes(part_no_crop)

        self.assertEqual(codes, ["50087288"])
        stripe_rescue.assert_called_once_with(part_no_crop)

    def test_non_part_no_scan_does_not_try_upscaled_candidates(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            model_crop = os.path.join(root, "model.png")
            img = np.full((80, 160, 3), 255, dtype=np.uint8)
            cv2.imwrite(model_crop, img)

            with mock.patch.object(scan2, "read_barcodes", return_value=[]) as read_barcodes:
                entries = scan2._scan_barcode_sources(
                    [("model", model_crop)],
                    field="MODEL",
                )

        self.assertEqual(entries, [])
        read_barcodes.assert_called_once_with(model_crop)

    def test_part_no_crop_ocr_maps_to_model_template(self):
        scan2 = _import_scan2()

        with tempfile.NamedTemporaryFile(suffix=".png") as part_no_crop:
            with mock.patch.object(scan2, "_scan_barcode_sources", return_value=[]):
                with mock.patch.object(
                    scan2,
                    "ocr_text_with_details",
                    return_value=("Part No.: 50087288 Rev:", "PartNo50087288", []),
                ):
                    model, raw, source = scan2.try_model_from_part_no_crop(
                        part_no_crop.name,
                        use_ocr=True,
                    )

        self.assertEqual(model, "AP162E")
        self.assertEqual(raw, "[PART_NO_OCR] 50087288")
        self.assertEqual(source, "part_no_ocr")

    def test_part_no_crop_ocr_can_use_model_text_when_part_no_scan_misses(self):
        scan2 = _import_scan2()

        with tempfile.NamedTemporaryFile(suffix=".png") as part_no_crop:
            with mock.patch.object(scan2, "_scan_barcode_sources", return_value=[]):
                with mock.patch.object(
                    scan2,
                    "ocr_text_with_details",
                    return_value=("Part No.: 50087149 Model: AP162E", "PartNo50087149ModelAP162E", []),
                ):
                    with mock.patch.object(scan2, "model_from_part_no_hint", return_value=("", "")):
                        model, raw, source = scan2.try_model_from_part_no_crop(
                            part_no_crop.name,
                            use_ocr=True,
                        )

        self.assertEqual(model, "AP162E")
        self.assertEqual(raw, "[PART_NO_OCR_MODEL] AP162E")
        self.assertEqual(source, "part_no_ocr_model")

    def test_main_uses_part_no_ocr_after_part_no_barcode_miss(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            part_no_dir = os.path.join(stage2, "part_no")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(part_no_dir)
            os.makedirs(sn_dir)
            label_id = "a__label_1"
            part_no_path = os.path.join(part_no_dir, f"{label_id}__part_no.png")
            open(part_no_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "part_no_path": part_no_path,
                        }
                    )
                    + "\n"
                )

            def fake_part_no(_path, label_id="", use_ocr=False):
                if use_ocr:
                    return "AP265E", "[PART_NO_OCR] 50087144", "part_no_ocr"
                return "", "", "part_no_no_barcode"

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.dict(
                os.environ,
                {
                    "SCAN2_PARALLEL": "0",
                    "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
                    "SCAN2_PART_NO_MODEL_MAP_PATH": os.path.join(root, "empty_map.json"),
                },
            ):
                with mock.patch.object(
                    scan2,
                    "try_model_from_part_no_crop",
                    side_effect=fake_part_no,
                ) as part_no_scan:
                    with mock.patch.object(scan2, "delayed_model_crop_from_label") as delayed_crop:
                        with mock.patch.object(scan2, "recognize_model_barcode") as model_barcode:
                            with mock.patch.object(scan2, "recognize_model_ocr") as model_ocr:
                                stats = scan2.main(
                                    model_dir=model_dir,
                                    sn_dir=sn_dir,
                                    out_jsonl=out_jsonl,
                                    debug_log=os.path.join(root, "debug.log"),
                                )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())

        self.assertEqual(stats["model_success"], 1)
        self.assertEqual(stats["model_part_no_hits"], 1)
        self.assertEqual(row["model"], "AP265E")
        self.assertEqual(row["part_no"], "50087144")
        self.assertEqual(row["model_src"], "part_no_ocr")
        self.assertEqual([call.kwargs["use_ocr"] for call in part_no_scan.call_args_list], [False, True])
        delayed_crop.assert_not_called()
        model_barcode.assert_not_called()
        model_ocr.assert_not_called()

    def test_main_uses_same_label_model_crop_when_part_no_ocr_fails(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            part_no_dir = os.path.join(stage2, "part_no")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(part_no_dir)
            os.makedirs(sn_dir)
            label_id = "a__label_1"
            label_crop = os.path.join(stage2, f"{label_id}.png")
            part_no_path = os.path.join(part_no_dir, f"{label_id}__part_no.png")
            model_path = os.path.join(model_dir, f"{label_id}__model.png")
            open(label_crop, "wb").close()
            open(part_no_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "label_crop": label_crop,
                            "part_no_path": part_no_path,
                        }
                    )
                    + "\n"
                )

            def fake_part_no(_path, label_id="", use_ocr=False):
                return "", "", "part_no_ocr_no_match" if use_ocr else "part_no_no_barcode"

            def fake_delayed_crop(item, requested_label_id):
                self.assertEqual(requested_label_id, label_id)
                open(model_path, "wb").close()
                item["model_path"] = model_path
                return model_path

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.dict(
                os.environ,
                {
                    "SCAN2_PARALLEL": "0",
                    "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
                    "SCAN2_DELAYED_MODEL_CROP": "1",
                    "SCAN2_PART_NO_MODEL_MAP_PATH": os.path.join(root, "empty_map.json"),
                },
            ):
                with mock.patch.object(
                    scan2,
                    "try_model_from_part_no_crop",
                    side_effect=fake_part_no,
                ) as part_no_scan:
                    with mock.patch.object(
                        scan2,
                        "delayed_model_crop_from_label",
                        side_effect=fake_delayed_crop,
                    ) as delayed_crop:
                        with mock.patch.object(
                            scan2,
                            "recognize_model_barcode",
                            return_value=("", "", "barcode_no_match"),
                        ) as model_barcode:
                            with mock.patch.object(
                                scan2,
                                "recognize_model_ocr",
                                return_value=("AP162E", "AP162E", "ocr_file"),
                            ) as model_ocr:
                                stats = scan2.main(
                                    model_dir=model_dir,
                                    sn_dir=sn_dir,
                                    out_jsonl=out_jsonl,
                                    debug_log=os.path.join(root, "debug.log"),
                                )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())

        self.assertEqual(stats["model_success"], 1)
        self.assertEqual(stats["model_deferred_crops"], 1)
        self.assertEqual(stats["model_ocr_recoveries"], 1)
        self.assertEqual(row["model"], "AP162E")
        self.assertEqual(row["model_src"], "ocr_file")
        self.assertEqual([call.kwargs["use_ocr"] for call in part_no_scan.call_args_list], [False, True, True])
        delayed_crop.assert_called_once()
        model_barcode.assert_called_once_with(model_path, label_id=label_id)
        model_ocr.assert_called_once_with(
            model_path,
            label_id=label_id,
            verify_barcode_visual=True,
        )

    def test_model_ocr_garbage_is_not_plausible_model(self):
        scan2 = _import_scan2()

        self.assertEqual(scan2.extract_model_from_text("ModeiZAP1621"), "")
        self.assertEqual(scan2.extract_model_from_text("Made in China CE 86598 34390"), "")
        self.assertEqual(scan2.extract_model_from_text("MAC: E4902A1B73D0"), "")
        self.assertEqual(scan2.extract_model_from_text("Model: AR180Pro"), "AR180Pro")

    def test_main_uses_part_no_crop_before_model_ocr(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            part_no_dir = os.path.join(stage2, "part_no")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(part_no_dir)
            os.makedirs(sn_dir)
            label_id = "a__label_1"
            model_path = os.path.join(model_dir, f"{label_id}__model.png")
            part_no_path = os.path.join(part_no_dir, f"{label_id}__part_no.png")
            open(model_path, "wb").close()
            open(part_no_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "model_path": model_path,
                            "part_no_path": part_no_path,
                        }
                    )
                    + "\n"
                )

            with mock.patch.object(
                scan2,
                "recognize_model_barcode",
                return_value=("", "", "barcode_no_match"),
            ) as model_barcode:
                with mock.patch.object(
                    scan2,
                    "try_model_from_part_no_crop",
                    return_value=("AP362E", "[PART_NO_BARCODE] 50087147", "part_no_barcode"),
                ) as part_no_scan:
                    with mock.patch.object(scan2, "recognize_model_ocr") as model_ocr:
                        stats = scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=os.path.join(root, "out.jsonl"),
                            debug_log=os.path.join(root, "debug.log"),
                        )

        self.assertEqual(stats["model_success"], 1)
        self.assertEqual(stats["model_part_no_hits"], 1)
        part_no_scan.assert_called_once_with(part_no_path, label_id=label_id, use_ocr=False)
        model_barcode.assert_not_called()
        model_ocr.assert_not_called()

    def test_main_falls_back_to_model_barcode_when_part_no_is_unknown(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            part_no_dir = os.path.join(stage2, "part_no")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(part_no_dir)
            os.makedirs(sn_dir)
            label_id = "a__label_1"
            model_path = os.path.join(model_dir, f"{label_id}__model.png")
            part_no_path = os.path.join(part_no_dir, f"{label_id}__part_no.png")
            open(model_path, "wb").close()
            open(part_no_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "model_path": model_path,
                            "part_no_path": part_no_path,
                        }
                    )
                    + "\n"
                )

            with mock.patch.object(
                scan2,
                "try_model_from_part_no_crop",
                return_value=("", "part_no:59999999", "part_no_no_match"),
            ) as part_no_scan:
                with mock.patch.object(
                    scan2,
                    "recognize_model_barcode",
                    return_value=("AP162E", "raw", "barcode"),
                ) as model_barcode:
                    with mock.patch.object(scan2, "recognize_model_ocr") as model_ocr:
                        stats = scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=os.path.join(root, "out.jsonl"),
                            debug_log=os.path.join(root, "debug.log"),
                        )

        self.assertEqual(stats["model_success"], 1)
        self.assertEqual(stats["model_barcode_hits"], 1)
        part_no_scan.assert_called_once_with(part_no_path, label_id=label_id, use_ocr=False)
        model_barcode.assert_called_once_with(model_path, label_id=label_id)
        model_ocr.assert_not_called()

    def test_main_accepts_unknown_model_when_barcode_and_ocr_agree(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            part_no_dir = os.path.join(stage2, "part_no")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(part_no_dir)
            os.makedirs(sn_dir)
            label_id = "xa__label_4"
            model_path = os.path.join(model_dir, f"{label_id}__model.png")
            part_no_path = os.path.join(part_no_dir, f"{label_id}__part_no.png")
            open(model_path, "wb").close()
            open(part_no_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "model_path": model_path,
                            "part_no_path": part_no_path,
                        }
                    )
                    + "\n"
                )

            map_path = os.path.join(root, "part_no_model_map.json")
            learned_path = os.path.join(root, "learned_model_codes.json")
            out_jsonl = os.path.join(root, "out.jsonl")

            with mock.patch.dict(
                os.environ,
                {
                    "SCAN2_PARALLEL": "0",
                    "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
                    "SCAN2_PART_NO_MODEL_MAP_PATH": map_path,
                    "SCAN2_LEARNED_MODEL_CODES_PATH": learned_path,
                },
                clear=False,
            ):
                with mock.patch.object(
                    scan2,
                    "recognize_model_barcode",
                    return_value=("", "model:E3; model_band:E3", "barcode_no_match"),
                ) as model_barcode:
                    with mock.patch.object(
                        scan2,
                        "recognize_model_ocr",
                        return_value=("", "Model: E3", "none"),
                    ) as model_ocr:
                        with mock.patch.object(
                            scan2,
                            "try_model_from_part_no_crop",
                            return_value=("", "part_no:50010842", "part_no_no_match"),
                        ) as part_no_scan:
                            stats = scan2.main(
                                model_dir=model_dir,
                                sn_dir=sn_dir,
                                out_jsonl=out_jsonl,
                                debug_log=os.path.join(root, "debug.log"),
                            )

            self.assertEqual(stats["model_success"], 1)
            # barcode_ocr_consensus required OCR agreement, so it is an OCR
            # recovery and must never inflate the barcode hit rate.
            self.assertEqual(stats["model_barcode_hits"], 0)
            self.assertEqual(stats["model_ocr_recoveries"], 1)
            self.assertEqual(stats["model_consensus_learned"], 1)
            self.assertEqual(stats["model_part_no_learned"], 1)
            self.assertGreaterEqual(model_barcode.call_count, 1)
            self.assertEqual(model_barcode.call_args_list[0], mock.call(model_path, label_id=label_id))
            self.assertGreaterEqual(model_ocr.call_count, 1)
            self.assertEqual(
                model_ocr.call_args_list[0],
                mock.call(
                    model_path,
                    label_id=label_id,
                    verify_barcode_visual=True,
                ),
            )
            part_no_scan.assert_called()

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertEqual(row["model"], "E3")
            self.assertEqual(row["model_src"], "barcode_ocr_consensus")
            self.assertEqual(row["part_no"], "50010842")
            self.assertTrue(row["part_no_model_map_updated"])

            with open(map_path, "r", encoding="utf-8") as f:
                part_no_map = json.load(f)
            self.assertEqual(part_no_map["50010842"]["model"], "E3")

            with open(learned_path, "r", encoding="utf-8") as f:
                learned_models = json.load(f)
            self.assertIn("E3", learned_models)

    def test_main_delays_model_crop_when_part_no_barcode_misses(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            part_no_dir = os.path.join(stage2, "part_no")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(part_no_dir)
            os.makedirs(sn_dir)
            label_id = "a__label_1"
            label_crop = os.path.join(stage2, f"{label_id}.png")
            part_no_path = os.path.join(part_no_dir, f"{label_id}__part_no.png")
            learned_model_path = os.path.join(model_dir, f"{label_id}__model.png")
            open(label_crop, "wb").close()
            open(part_no_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "label_crop": label_crop,
                            "part_no_path": part_no_path,
                        }
                    )
                    + "\n"
                )

            def fake_delayed_crop(item, requested_label_id):
                self.assertEqual(requested_label_id, label_id)
                open(learned_model_path, "wb").close()
                item["model_path"] = learned_model_path
                return learned_model_path

            with mock.patch.dict(
                os.environ,
                {
                    "SCAN2_PARALLEL": "0",
                    "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
                    "SCAN2_DELAYED_MODEL_CROP": "1",
                    "SCAN2_OCR_FALLBACK": "0",
                },
            ):
                with mock.patch.object(
                    scan2,
                    "try_model_from_part_no_crop",
                    return_value=("", "", "part_no_no_barcode"),
                ) as part_no_scan:
                    with mock.patch.object(
                        scan2,
                        "delayed_model_crop_from_label",
                        side_effect=fake_delayed_crop,
                    ) as delayed_crop:
                        with mock.patch.object(
                            scan2,
                            "recognize_model_barcode",
                            return_value=("AP362E", "AP362E", "barcode"),
                        ) as model_barcode:
                            with mock.patch.object(scan2, "recognize_model_ocr") as model_ocr:
                                stats = scan2.main(
                                    model_dir=model_dir,
                                    sn_dir=sn_dir,
                                    out_jsonl=os.path.join(root, "out.jsonl"),
                                    debug_log=os.path.join(root, "debug.log"),
                                )

        self.assertEqual(stats["model_success"], 1)
        self.assertEqual(stats["model_deferred_crops"], 1)
        self.assertEqual(stats["model_barcode_hits"], 1)
        part_no_scan.assert_called_once_with(part_no_path, label_id=label_id, use_ocr=False)
        delayed_crop.assert_called_once()
        model_barcode.assert_called_once_with(learned_model_path, label_id=label_id)
        model_ocr.assert_not_called()

    def test_unknown_part_no_learns_model_once_for_same_batch(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            part_no_dir = os.path.join(stage2, "part_no")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(part_no_dir)
            os.makedirs(sn_dir)
            label_ids = ["a__label_1", "a__label_2"]
            manifest_path = os.path.join(stage2, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                for label_id in label_ids:
                    label_crop = os.path.join(stage2, f"{label_id}.png")
                    part_no_path = os.path.join(part_no_dir, f"{label_id}__part_no.png")
                    open(label_crop, "wb").close()
                    open(part_no_path, "wb").close()
                    manifest.write(
                        json.dumps(
                            {
                                "label_id": label_id,
                                "label_crop": label_crop,
                                "part_no_path": part_no_path,
                            }
                        )
                        + "\n"
                    )

            learned_model_path = os.path.join(model_dir, f"{label_ids[0]}__model.png")
            map_path = os.path.join(root, "part_no_model_map.json")

            def fake_delayed_crop(item, label_id):
                self.assertEqual(label_id, label_ids[0])
                open(learned_model_path, "wb").close()
                item["model_path"] = learned_model_path
                return learned_model_path

            with mock.patch.dict(
                os.environ,
                {
                    "SCAN2_PARALLEL": "0",
                    "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
                    "SCAN2_PART_NO_MODEL_MAP_PATH": map_path,
                    "SCAN2_DELAYED_MODEL_CROP": "1",
                },
            ):
                with mock.patch.object(
                    scan2,
                    "try_model_from_part_no_crop",
                    return_value=("", "Part No: 50099999", "part_no_no_match"),
                ) as part_no_scan:
                    with mock.patch.object(
                        scan2,
                        "delayed_model_crop_from_label",
                        side_effect=fake_delayed_crop,
                    ) as delayed_crop:
                        with mock.patch.object(
                            scan2,
                            "recognize_model_barcode",
                            return_value=("AP162E", "AP162E", "barcode"),
                        ) as model_barcode:
                            with mock.patch.object(scan2, "recognize_model_ocr") as model_ocr:
                                stats = scan2.main(
                                    model_dir=model_dir,
                                    sn_dir=sn_dir,
                                    out_jsonl=os.path.join(root, "out.jsonl"),
                                    debug_log=os.path.join(root, "debug.log"),
                                )

            self.assertEqual(stats["model_success"], 2)
            self.assertEqual(stats["model_part_no_hits"], 2)
            self.assertEqual(stats["model_part_no_learned"], 1)
            self.assertEqual(stats["model_deferred_crops"], 1)
            self.assertEqual(stats["part_no_decoded"], 2)
            self.assertEqual(part_no_scan.call_count, 2)
            delayed_crop.assert_called_once()
            model_barcode.assert_called_once_with(learned_model_path, label_id=label_ids[0])
            model_ocr.assert_not_called()

            with open(map_path, "r", encoding="utf-8") as f:
                part_no_map = json.load(f)
            self.assertEqual(part_no_map["50099999"]["model"], "AP162E")

            with open(os.path.join(root, "out.jsonl"), "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual([row["model"] for row in rows], ["AP162E", "AP162E"])
            self.assertEqual([row["part_no"] for row in rows], ["50099999", "50099999"])
            self.assertEqual([row["model_src"] for row in rows], ["part_no_learned", "part_no_learned"])
            self.assertEqual([row["part_no_model_map_updated"] for row in rows], [True, False])

    def test_recognize_sn_rejects_ocr_prefix_junk_after_barcode_miss(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(status="decoder_miss", attempts=1)

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            with mock.patch.object(scan2, "load_for_ocr_color", return_value=object()):
                with mock.patch.object(
                    scan2,
                    "ocr_text_with_details",
                    side_effect=[
                        (
                            "MSN2150087147LDS4023",
                            "MSN2150087147LDS4023",
                            ["MSN2150087147LDS4023"],
                        ),
                        ("", "", []),
                    ],
                ):
                    with mock.patch.object(scan2, "load_and_preprocess", return_value=object()):
                        with mock.patch.object(scan2, "ocr_sn_top_text", return_value=("", "")):
                            sn, raw, source, meta = scan2.recognize_sn(
                                "sn.png",
                                label_id="a__label_1",
                                label_path="label.png",
                            )

        self.assertEqual(sn, "")
        self.assertEqual(source, "ocr_no_match")
        self.assertTrue(meta["ocr_text_found"])
        self.assertFalse(meta["barcode_found"])
        self.assertNotEqual(sn, "2150087147LDS4023")

    def test_recognize_sn_ocr_does_not_append_waybill_digits(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(status="parse_failure", attempts=1)

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            with mock.patch.object(scan2, "load_for_ocr_color", return_value=object()):
                with mock.patch.object(
                    scan2,
                    "ocr_text_with_details",
                    side_effect=[
                        (
                            "S/N:21500872884ERB011934 966515",
                            "S/N:21500872884ERB011934966515",
                            ["S/N:21500872884ERB011934 966515"],
                        )
                    ],
                ):
                    sn, raw, source, meta = scan2.recognize_sn_ocr_after_barcode(
                        "sn.png",
                        label_id="海南.jpg__label_5",
                        barcode_report=report,
                        meta={"barcode_found": False, "ocr_text_found": False, "barcode_status": "parse_failure"},
                    )

        self.assertEqual(sn, "21500872884ERB011934")
        self.assertEqual(source, "ocr")
        self.assertTrue(meta["ocr_text_found"])
        self.assertEqual(raw, "S/N:21500872884ERB011934 966515")

    def test_recognize_sn_ocr_accepts_unknown_value_when_barcode_and_ocr_agree(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="parse_failure",
            attempts=1,
            decoded_count=1,
            results=[
                sn_barcode.DecoderResult(
                    "fake",
                    "SN: ABC12345XYZ9",
                    "sn",
                    "sn.orig",
                )
            ],
        )

        with mock.patch.object(scan2, "load_for_ocr_color", return_value=object()):
            with mock.patch.object(
                scan2,
                "ocr_text_with_details",
                side_effect=[
                    (
                        "SN: ABC12345XYZ9",
                        "SN: ABC12345XYZ9",
                        ["SN: ABC12345XYZ9"],
                    )
                ],
            ):
                sn, raw, source, meta = scan2.recognize_sn_ocr_after_barcode(
                    "sn.png",
                    label_id="x__label_1",
                    barcode_report=report,
                    meta={"barcode_found": True, "ocr_text_found": False, "barcode_status": "parse_failure"},
                )

        self.assertEqual(sn, "ABC12345XYZ9")
        self.assertEqual(raw, "SN: ABC12345XYZ9")
        self.assertEqual(source, "barcode_ocr_consensus")
        self.assertTrue(meta["ocr_text_found"])

    def test_recognize_sn_barcode_hit_skips_ocr_prefix_junk(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="hit",
            sn="2150010843LDRC000793",
            raw_text="SN:2150010843LDRC000793",
            source="label",
            source_region="label",
            decoder_name="fake",
            attempts=1,
            decoded_count=1,
            results=[
                sn_barcode.DecoderResult(
                    "fake",
                    "SN:2150010843LDRC000793",
                    "label",
                    "label",
                )
            ],
        )

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            with mock.patch.object(scan2, "ocr_text_with_details") as ocr_text:
                sn, raw, source, meta = scan2.recognize_sn(
                    "sn.png",
                    label_id="a__label_1",
                    label_path="label.png",
                )

        self.assertEqual(sn, "2150010843LDRC000793")
        self.assertEqual(source, "barcode")
        self.assertTrue(meta["barcode_found"])
        self.assertIn("BARCODE:label", raw)
        ocr_text.assert_not_called()

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
            report = types.SimpleNamespace(status="decoder_miss", results=[])
            with mock.patch.object(scan2, "_recognize_sn_barcode", return_value=("", "", "barcode_decoder_miss", meta, report)) as recognize_sn:
                with mock.patch.object(scan2, "recognize_sn_ocr_after_barcode", return_value=("", "", "none", meta)):
                    scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=os.path.join(root, "out.jsonl"),
                        debug_log=os.path.join(root, "debug.log"),
                    )

        self.assertEqual(recognize_sn.call_count, 2)
        self.assertEqual([call.kwargs["original_path"] for call in recognize_sn.call_args_list], ["", ""])

    def test_main_uses_label_crop_ocr_when_sn_crop_is_missing(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            label_dir = os.path.join(root, "stage1_labels")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            os.makedirs(label_dir)

            label_id = "photo.jpg__label_1"
            label_path = os.path.join(label_dir, f"{label_id}.png")
            open(label_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": label_id, "label_crop": label_path}) + "\n")

            meta = {"barcode_found": False, "ocr_text_found": False, "barcode_status": "decoder_miss"}
            report = types.SimpleNamespace(status="decoder_miss", results=[])
            with mock.patch.dict(os.environ, {"SCAN2_PARALLEL": "0", "SCAN2_SCAN_LABEL_WITHOUT_SN": "1"}):
                with mock.patch.object(scan2, "_recognize_sn_barcode", return_value=("", "", "barcode_decoder_miss", meta, report)):
                    with mock.patch.object(
                        scan2,
                        "recognize_sn_ocr_after_barcode",
                        return_value=("4E25A0068800", "SN:4E25A0068800", "ocr", meta),
                    ) as sn_ocr:
                        stats = scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=os.path.join(root, "out.jsonl"),
                            debug_log=os.path.join(root, "debug.log"),
                        )

        self.assertEqual(stats["sn_success"], 1)
        sn_ocr.assert_called_once()
        self.assertEqual(sn_ocr.call_args.args[0], label_path)

    def test_main_treats_missing_original_image_as_provenance_only(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            sn_dir = os.path.join(stage2, "sn")
            model_dir = os.path.join(stage2, "model")
            label_dir = os.path.join(root, "stage1_labels")
            os.makedirs(sn_dir)
            os.makedirs(model_dir)
            os.makedirs(label_dir)

            label_id = "photo.jpg__label_1"
            label_path = os.path.join(label_dir, f"{label_id}.png")
            sn_path = os.path.join(sn_dir, f"{label_id}__sn.png")
            open(label_path, "wb").close()
            open(sn_path, "wb").close()

            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "label_id": label_id,
                    "label_crop": label_path,
                    "sn_path": sn_path,
                    "original_image_path": os.path.join(root, "missing-source.jpg"),
                    "image_path": os.path.join(root, "also-missing-source.jpg"),
                }) + "\n")

            meta = {"barcode_found": False, "ocr_text_found": False, "barcode_status": "decoder_miss"}
            report = types.SimpleNamespace(status="decoder_miss", results=[])
            with mock.patch.object(scan2, "_recognize_sn_barcode", return_value=("", "", "barcode_decoder_miss", meta, report)) as recognize_sn:
                with mock.patch.object(scan2, "recognize_sn_ocr_after_barcode", return_value=("", "", "none", meta)):
                    stats = scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=os.path.join(root, "out.jsonl"),
                        debug_log=os.path.join(root, "debug.log"),
                    )

        self.assertEqual(stats["sn_total"], 1)
        recognize_sn.assert_called_once()
        self.assertEqual(recognize_sn.call_args.kwargs["label_path"], label_path)
        self.assertEqual(recognize_sn.call_args.kwargs["original_path"], "")

    def test_recognize_sn_ignores_direct_original_path(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(status="decoder_miss", attempts=1)

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report) as scan_report:
            scan2.recognize_sn(
                "sn.png",
                label_id="a__label_1",
                label_path="label.png",
                original_path="source-photo.jpg",
                allow_ocr=False,
            )

        self.assertEqual(scan_report.call_args.args[0], [("sn", "sn.png"), ("label", "label.png")])

    def test_validation_sources_exclude_original_image(self):
        with tempfile.TemporaryDirectory() as root:
            sn_path = os.path.join(root, "sn.png")
            label_path = os.path.join(root, "label.png")
            original_path = os.path.join(root, "source.jpg")
            for path in (sn_path, label_path, original_path):
                open(path, "wb").close()

            sources = validate_sn_barcodes._sources_for_row(
                {
                    "sn_path": sn_path,
                    "label_crop": label_path,
                    "original_image_path": original_path,
                    "image_path": original_path,
                },
                root,
            )

        self.assertEqual(sources, [("sn", sn_path), ("label", label_path)])

    def test_delayed_model_crop_rejects_unsafe_label_id(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            model_dir = os.path.join(root, "stage2_fields", "model")
            label_path = os.path.join(root, "label.png")
            open(label_path, "wb").close()
            scan2.MODEL_CROP_DIR = model_dir

            crop = types.ModuleType("crop")
            crop.stage2_crop_model_from_label = mock.Mock()
            with mock.patch.dict(sys.modules, {"crop": crop}):
                for unsafe_label_id in (r"..\escape", "../escape", "C:escape"):
                    with self.subTest(label_id=unsafe_label_id):
                        result = scan2.delayed_model_crop_from_label(
                            {"label_crop": label_path},
                            unsafe_label_id,
                        )
                        self.assertEqual(result, "")

        crop.stage2_crop_model_from_label.assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(root, "escape__model.png")))

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
            report = types.SimpleNamespace(status="decoder_miss", results=[])
            with mock.patch.object(scan2, "_recognize_sn_barcode", return_value=("", "", "barcode_decoder_miss", meta, report)):
                with mock.patch.object(scan2, "recognize_sn_ocr_after_barcode", return_value=("4E25A0170000", "ocr raw", "ocr", meta)):
                    with mock.patch.dict(os.environ, {"SCAN2_OCR_FALLBACK": "1"}, clear=False):
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

    def test_main_marks_sn_problem_after_barcode_and_ocr_fail(self):
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

            barcode_meta = {
                "barcode_found": False,
                "ocr_text_found": False,
                "barcode_status": "decoder_miss",
                "barcode_attempts": 2,
                "barcode_decoded_count": 0,
            }
            ocr_meta = dict(barcode_meta)
            ocr_meta["ocr_text_found"] = True
            report = types.SimpleNamespace(status="decoder_miss", results=[])
            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.dict(os.environ, {"SCAN2_PARALLEL": "0"}, clear=False):
                with mock.patch.object(
                    scan2,
                    "_recognize_sn_barcode",
                    return_value=("", "", "barcode_decoder_miss", barcode_meta, report),
                ):
                    with mock.patch.object(
                        scan2,
                        "recognize_sn_ocr_after_barcode",
                        return_value=("", "SN: unreadable", "ocr_no_match", ocr_meta),
                    ) as sn_ocr:
                        stats = scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=out_jsonl,
                            debug_log=os.path.join(root, "debug.log"),
                        )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())

        self.assertEqual(stats["sn_success"], 0)
        self.assertEqual(stats["sn_problem"], 1)
        self.assertEqual(stats["sn_ocr_recoveries"], 0)
        self.assertTrue(row["sn_problem"])
        self.assertEqual(row["sn_problem_reason"], "ocr_no_match")
        self.assertEqual(row["sn_src"], "ocr_no_match")
        sn_ocr.assert_called_once()


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

    def test_validation_rejects_accepted_barcode_row_without_label_local_source(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "source.jpg")
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

            with mock.patch.object(validate_sn_barcodes, "scan_sn_barcodes") as scan:
                summary = validate_sn_barcodes.evaluate_manifest(
                    manifest_path,
                    threshold=0.90,
                    min_accepted=1,
                )

        self.assertFalse(summary.passed)
        self.assertEqual(summary.accepted_barcode_rows, 0)
        self.assertEqual(summary.denominator, 0)
        self.assertEqual(summary.failure_counts["decoder_miss"], 0)
        self.assertIn("label-local source", "\n".join(summary.errors))
        scan.assert_not_called()

    def test_validation_accepts_stale_provenance_when_sn_path_exists(self):
        with tempfile.TemporaryDirectory() as root:
            sn_path = os.path.join(root, "sn.png")
            open(sn_path, "wb").close()
            manifest_path = os.path.join(root, "manifest.jsonl")
            stale_image_path = os.path.join(root, "missing-source.jpg")
            stale_original_path = os.path.join(root, "missing-original.jpg")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "image_path": stale_image_path,
                    "original_image_path": stale_original_path,
                    "sn_path": sn_path,
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
                source="sn",
                source_region="sn",
                decoder_name="fake",
                attempts=1,
                decoded_count=1,
            )
            with mock.patch.object(validate_sn_barcodes, "scan_sn_barcodes", return_value=fake_report) as scan:
                summary = validate_sn_barcodes.evaluate_manifest(
                    manifest_path,
                    threshold=0.90,
                    min_accepted=1,
                )

        self.assertTrue(summary.passed)
        self.assertEqual(summary.errors, [])
        self.assertEqual(summary.accepted_barcode_rows, 1)
        self.assertEqual(summary.denominator, 1)
        scan.assert_called_once()
        self.assertEqual(scan.call_args.args[0], [("sn", sn_path)])

    def test_validation_fails_when_accepted_sample_count_is_too_small(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "label.png")
            open(image_path, "wb").close()
            manifest_path = os.path.join(root, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "image_path": image_path,
                    "label_crop": image_path,
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
                        "label_crop": image_path,
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
                    "label_crop": image_path,
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
