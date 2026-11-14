import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from huaweiocr.barcode.label_codes import LabelCodes, LabelQr, parse_iso15434, read_label_codes
from huaweiocr.barcode.sn import DecoderResult, SnBarcodeReport
from huaweiocr.core import part_no_products as products
from huaweiocr.recognition import pipeline

QR_AP = "[)>\x1e06\x1d1P50087149\x1d\x1d18VLEHWT\x1dS21500871494ERB017591\x1e\x04"
QR_S110 = "[)>\x1e06\x1d1P98013170\x1d18VLEHWT\x1dS21980131704ES6000352\x1e\x04"


def _qr_image(payload, scale):
    import zxingcpp

    barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
    gray = np.array(barcode.to_image(scale=scale))
    return cv2.cvtColor(cv2.copyMakeBorder(gray, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255),
                        cv2.COLOR_GRAY2BGR)


class ParseIso15434Test(unittest.TestCase):
    def test_parses_raw_and_control_picture_payloads(self):
        expected = LabelQr("50087149", "21500871494ERB017591")
        self.assertEqual(parse_iso15434(QR_AP), expected)
        pictures = QR_AP.replace("\x1e", "␞").replace("\x1d", "␝").replace("\x04", "␄")
        self.assertEqual(parse_iso15434(pictures), expected)
        self.assertEqual(parse_iso15434(QR_S110), LabelQr("98013170", "21980131704ES6000352"))
        self.assertEqual(
            parse_iso15434("[)>\x1e06\x1d1P98012125\x1dS4E2690029842\x1e\x04"),
            LabelQr("98012125", "4E2690029842"),
        )

    def test_parses_separator_less_switch_payloads(self):
        self.assertEqual(parse_iso15434("[)>061P9801240418VLEHWTS4E25B0077657"),
                         LabelQr("98012404", "4E25B0077657"))
        self.assertEqual(parse_iso15434("[)>061P5008714918VLEHWTS2150087149LDS5014813"),
                         LabelQr("50087149", "2150087149LDS5014813"))
        self.assertIsNone(parse_iso15434("[)>061P9801240418VLEHWT"))
        self.assertIsNone(parse_iso15434("[)>061P98012404 18VLEHWTSHELLO"))

    def test_accepts_serial_with_short_letter_suffix(self):
        qr = parse_iso15434("[)>061P50087149S21500871494ERB017591A")
        self.assertEqual(qr.sn, "21500871494ERB017591")

    def test_rejects_payloads_without_envelope_or_valid_serial(self):
        self.assertIsNone(parse_iso15434("S21500871494ERB017591"))
        self.assertIsNone(parse_iso15434("MMM={'k5':'SF5168220399968'}"))
        self.assertIsNone(parse_iso15434("[)>\x1e06\x1d1P50087149\x1dSHELLO\x1e\x04"))
        self.assertEqual(parse_iso15434("[)>\x1e06\x1d1P5008\x1dS21500871494ERB017591\x1e").part_no, "")


class LabelCodesTest(unittest.TestCase):
    def test_product_codes_drop_crops_with_neighbour_labels(self):
        self.assertEqual(
            LabelCodes(products=["6901443456956", "0886598343893", "6901443456956"]).label_product_codes(),
            ["6901443456956", "0886598343893"],
        )
        self.assertEqual(LabelCodes(products=["6901443456956", "6901443456963"]).label_product_codes(), [])

    def test_sole_qr_needs_one_payload_with_part_no(self):
        one = LabelQr("50087149", "A")
        self.assertEqual(LabelCodes(qr=[one, one]).sole_qr(), one)
        self.assertIsNone(LabelCodes(qr=[one, LabelQr("50087149", "B")]).sole_qr())
        self.assertIsNone(LabelCodes(qr=[LabelQr("", "A")]).sole_qr())
        self.assertIsNone(LabelCodes().sole_qr())

    def test_qr_sns_follow_label_part_no(self):
        codes = LabelCodes(qr=[LabelQr("50087149", "A"), LabelQr("98013170", "B"), LabelQr("", "C")])
        self.assertEqual(codes.qr_sns_for_part_no("50087149"), ["A"])
        self.assertEqual(codes.qr_sns_for_part_no(""), ["A", "B", "C"])

    def test_reads_label_qr_and_retries_small_codes_only_when_thorough(self):
        with tempfile.TemporaryDirectory() as root:
            big = os.path.join(root, "big.png")
            cv2.imwrite(big, _qr_image(QR_AP, 4))
            self.assertEqual(read_label_codes(big).qr, [LabelQr("50087149", "21500871494ERB017591")])

            small = os.path.join(root, "small.png")
            # 2 px/module shrunk and softened: unreadable as-is, readable upscaled.
            img = cv2.GaussianBlur(
                cv2.resize(_qr_image(QR_AP, 2), None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA),
                (0, 0), 0.8,
            )
            cv2.imwrite(small, img)
            plain = read_label_codes(small)
            thorough = read_label_codes(small, thorough=True)
            if plain.qr:
                self.skipTest("synthetic QR is readable without retries on this zxing-cpp build")
            self.assertEqual([q.sn for q in thorough.qr], ["21500871494ERB017591"])
            self.assertNotEqual(thorough.qr[0].variant, "plain")

    def test_unreadable_path_returns_empty_codes(self):
        codes = read_label_codes(os.path.join(tempfile.gettempdir(), "missing-label-crop.png"), thorough=True)
        self.assertEqual((codes.qr, codes.products), ([], []))


class ProductPartNoTest(unittest.TestCase):
    def test_conflict_needs_repeated_dominant_evidence_for_one_other_part_no(self):
        owners = {"6901443456956": {"50087149": 3}, "0886598343893": {"50087149": 1}}
        conflict = products.product_part_no_conflict
        self.assertEqual(conflict("50087288", ["6901443456956"], owners), "50087149")
        self.assertEqual(conflict("50087149", ["6901443456956"], owners), "")
        self.assertEqual(conflict("50087288", ["0886598343893"], owners), "")
        self.assertEqual(conflict("50087288", ["6901443000000"], owners), "")
        self.assertEqual(conflict("", ["6901443456956"], owners), "")
        # A second, never-seen code does not cancel the first one's conflict.
        self.assertEqual(conflict("50087288", ["6901443456956", "0886598343909"], owners), "50087149")
        # One earlier wrong pairing does not exempt the PartNo forever...
        self.assertEqual(conflict("50087288", ["6901443456956"],
                                  {"6901443456956": {"50087149": 3, "50087288": 1}}), "50087149")
        # Two sightings for another PartNo are enough when this one has none.
        self.assertEqual(conflict("50087288", ["6901443456956"], {"6901443456956": {"50087149": 2}}),
                         "50087149")
        # Own support counted separately: unverified batch pairings do not count.
        self.assertEqual(conflict("50087288", ["6901443456956"],
                                  {"6901443456956": {"50087149": 5, "50087288": 2}},
                                  {"6901443456956": {"50087149": 5}}), "50087149")
        # ...but comparable evidence for both is left alone.
        self.assertEqual(conflict("50087288", ["6901443456956"],
                                  {"6901443456956": {"50087149": 3, "50087288": 2}}), "")

    def test_unreadable_store_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "products.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"part_no_products": {"50087149": {"6901443456956": 9}},')
            with mock.patch.dict(os.environ, {"SCAN2_PART_NO_PRODUCTS_PATH": path}):
                self.assertEqual(products.record_part_no_products([("50087149", "6901443456956")]), 0)
            with open(path, encoding="utf-8") as f:
                self.assertTrue(f.read().endswith(","))

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "products.json")
            with mock.patch.dict(os.environ, {"SCAN2_PART_NO_PRODUCTS_PATH": path}):
                self.assertEqual(products.record_part_no_products(
                    [("50087149", "6901443456956"), ("50087149", "6901443456956"),
                     ("20261113", "6901443456956"), ("50087149", "123")]), 2)
                self.assertEqual(products.load_product_owners(), {"6901443456956": {"50087149": 2}})


def _report(sn, decoder, status="hit"):
    return SnBarcodeReport(
        status=status, sn=sn, raw_text=sn, source="sn", source_region=f"sn.{decoder}",
        decoder_name=decoder, decoded_count=1 if sn else 0,
        results=[DecoderResult(decoder, sn, "sn", f"sn.{decoder}")] if sn else [],
    )


class PipelineLabelCodeCheckTest(unittest.TestCase):
    def test_qr_rescues_weak_sns_and_flags_genuine_mismatches(self):
        cases = {
            # key: (current sn, decoder, label part no, QR codes, product codes)
            "forged_repair": ("21500871493ERB017592", "code128_pixel_repair", "50087149",
                              [LabelQr("50087149", "21500871494ERB017591", "x1.5")], []),
            "agreeing_repair": ("21500871494ERB017600", "code128_pixel_repair", "50087149",
                                [LabelQr("50087149", "21500871494ERB017600")], []),
            "no_sn": ("", "", "50087149", [LabelQr("50087149", "2150087149LDS5015390")], []),
            "genuine_mismatch": ("2150087149LDS5015385", "zxingcpp", "50087149",
                                 [LabelQr("50087149", "2150087149LDS5015363")], []),
            "genuine_ok": ("21980131704ES6000352", "pyzbar", "98013170",
                           [LabelQr("98013170", "21980131704ES6000352")],
                           ["6901443479078", "0886598353649"]),
            "neighbour_qr": ("", "", "50087149", [LabelQr("98013170", "21980131704ES6000352")], []),
            "unknown_part_no": ("", "", "", [LabelQr("98013170", "21980131704ES6000797")], []),
            "product_conflict": ("2150087288LDS0000001", "pyzbar", "50087288", [], ["6901443456956"]),
            "partless_qr": ("", "", "50087149", [LabelQr("", "2150087149LDS5015001")], []),
            "shared_a": ("", "", "50087149", [LabelQr("50087149", "2150087149LDS5015002")], []),
            "shared_b": ("21500871494ERB000001", "code128_pixel_repair", "50087149",
                         [LabelQr("50087149", "2150087149LDS5015002")], []),
            "neighbour_of_genuine": ("21500871494ERB000002", "code128_pixel_repair", "50087149",
                                     [LabelQr("50087149", "2150087149LDS5015385")], []),
        }
        keys = list(cases)
        records = {key: {"label_crop": f"{key}.png", "sn_path": f"{key}_sn.png",
                         "original_image_path": "shelf.jpg"} for key in keys}
        sn_reports, sn_results = {}, {}
        for key, (sn, decoder, *_rest) in cases.items():
            report = _report(sn, decoder) if sn else _report("", "pyzbar", status="decoder_miss")
            sn_reports[key] = report
            sn_results[key] = (sn, f"[BARCODE] {sn}", "barcode" if sn else "barcode_decoder_miss",
                               report.to_meta())
        part_no_by_key = {key: c[2] for key, c in cases.items() if c[2]}
        part_no_src_by_key = {key: "part_no_barcode" for key in part_no_by_key}
        ocr_jobs = [("sn", key, records[key], sn_reports[key])
                    for key in ("no_sn", "neighbour_qr", "unknown_part_no", "partless_qr", "shared_a")]
        ocr_jobs.append(("model", "no_sn", records["no_sn"], None))
        miss = {"unknown_part_no"}
        conflicts = {}
        reads = {}

        def fake_read(path, thorough=False):
            key = path[:-4]
            reads[key] = thorough
            return LabelCodes(qr=list(cases[key][3]), products=list(cases[key][4]))

        owners = {"6901443456956": {"50087149": 5}}
        with mock.patch.object(pipeline, "read_label_codes", fake_read), \
                mock.patch.object(pipeline, "load_product_owners", return_value=owners):
            observations = pipeline._cross_check_with_label_codes(
                keys, records, sn_results, sn_reports, part_no_by_key, part_no_src_by_key,
                ocr_jobs, conflicts, miss, workers=2,
            )

        self.assertTrue(reads["forged_repair"] and reads["no_sn"])
        self.assertFalse(reads["genuine_ok"] or reads["genuine_mismatch"])

        self.assertEqual(sn_results["forged_repair"][:3],
                         ("21500871494ERB017591", "[LABEL_QR] 21500871494ERB017591", "barcode_label_qr"))
        self.assertEqual(sn_results["forged_repair"][3]["label_qr_replaced"], "21500871493ERB017592")
        self.assertEqual(sn_reports["forged_repair"].decoder_name, "label_qr")
        self.assertEqual(sn_results["no_sn"][0], "2150087149LDS5015390")
        self.assertEqual(sn_results["agreeing_repair"][2], "barcode")
        self.assertEqual(sn_results["agreeing_repair"][3]["label_qr_check"], "confirmed")

        self.assertEqual(sn_results["genuine_mismatch"][0], "2150087149LDS5015385")
        self.assertEqual(sn_results["genuine_mismatch"][2], "barcode_qr_conflict")
        self.assertEqual(sn_results["genuine_mismatch"][3]["label_qr_sn"], "2150087149LDS5015363")
        self.assertEqual(sn_results["genuine_ok"][3]["label_qr_check"], "confirmed")

        self.assertEqual(sn_results["neighbour_qr"][0], "")
        self.assertEqual(sn_results["unknown_part_no"][0], "21980131704ES6000797")
        self.assertEqual(part_no_by_key["unknown_part_no"], "98013170")
        self.assertEqual(part_no_src_by_key["unknown_part_no"], "label_qr")
        self.assertEqual(miss, set())

        # A QR without PartNo, or one two labels both read, never rescues; a
        # QR equal to another label's genuine SN is that label's, not ours.
        self.assertEqual(sn_results["partless_qr"][0], "")
        self.assertEqual(sn_results["shared_a"][0], "")
        self.assertEqual(sn_results["shared_b"][0], "21500871494ERB000001")
        self.assertEqual(sn_results["shared_b"][3]["label_qr_check"], "claimed_by_other_label")
        self.assertEqual(sn_results["neighbour_of_genuine"][0], "21500871494ERB000002")
        self.assertEqual(sn_results["genuine_mismatch"][3].get("label_qr_check"), "conflict")
        self.assertEqual(
            [(job[0], job[1]) for job in ocr_jobs],
            [("sn", "neighbour_qr"), ("sn", "partless_qr"), ("sn", "shared_a"), ("model", "no_sn")],
        )
        self.assertEqual(conflicts, {"product_conflict": "50087149"})
        self.assertEqual(observations, [("98013170", "6901443479078"), ("98013170", "0886598353649")])

    def test_same_label_in_two_photos_is_not_a_neighbour(self):
        keys = ["photo1_label", "photo2_label"]
        records = {
            "photo1_label": {"label_crop": "photo1_label.png", "original_image_path": "a.jpg"},
            "photo2_label": {"label_crop": "photo2_label.png", "original_image_path": "b.jpg"},
        }
        sn = "21500871474ERB042721"
        sn_reports = {"photo1_label": _report(sn, "pyzbar"),
                      "photo2_label": _report("21500871474ERB042720", "code128_pixel_repair")}
        sn_results = {key: (r.sn, f"[BARCODE] {r.sn}", "barcode", r.to_meta()) for key, r in sn_reports.items()}
        with mock.patch.object(pipeline, "read_label_codes",
                               lambda path, thorough=False: LabelCodes(qr=[LabelQr("50087147", sn)])),                 mock.patch.object(pipeline, "load_product_owners", return_value={}):
            pipeline._cross_check_with_label_codes(
                keys, records, sn_results, sn_reports, {k: "50087147" for k in keys},
                {k: "part_no_barcode" for k in keys}, [], {},
            )
        self.assertEqual(sn_results["photo1_label"][3]["label_qr_check"], "confirmed")
        self.assertEqual(sn_results["photo2_label"][:3], (sn, f"[LABEL_QR] {sn}", "barcode_label_qr"))

    def test_batch_pairings_cannot_vouch_for_each_other(self):
        keys = ["w1", "w2", "w3", "ok1", "ok2"]
        records = {key: {"label_crop": f"{key}.png"} for key in keys}
        part_no_by_key = {"w1": "50087288", "w2": "50087288", "w3": "50087288",
                          "ok1": "98013170", "ok2": "98013170"}
        part_no_src_by_key = {key: "part_no_barcode" for key in keys}
        product = {"w1": "6901443456956", "w2": "6901443456956", "w3": "6901443456956",
                   "ok1": "6901443479078", "ok2": "6901443479078"}
        conflicts = {}
        with mock.patch.object(pipeline, "read_label_codes",
                               lambda path, thorough=False: LabelCodes(products=[product[path[:-4]]])),                 mock.patch.object(pipeline, "load_product_owners",
                                  return_value={"6901443456956": {"50087149": 5}}):
            observations = pipeline._cross_check_with_label_codes(
                keys, records, {}, {}, part_no_by_key, part_no_src_by_key, [], conflicts,
            )
        self.assertEqual(conflicts, {"w1": "50087149", "w2": "50087149", "w3": "50087149"})
        self.assertEqual(observations, [("98013170", "6901443479078")] * 2)

    def test_qr_first_takes_sn_and_part_no_from_a_trusted_label_qr(self):
        qr = {
            "match": LabelQr("50087149", "2150087149LDS5014813"),
            "other_part": LabelQr("50087149", "2150087149LDS5014814"),
            "no_manifest": LabelQr("98013170", "21980131704ES6000352"),
            "unknown": LabelQr("12345678", "2112345678ES0000001"),
            "dup1": LabelQr("50087149", "2150087149LDS5015367"),
            "dup2": LabelQr("50087149", "2150087149LDS5015367"),
            "two": LabelQr("50087149", "2150087149LDS5015363"),
        }
        codes = {key: LabelCodes(qr=[value]) for key, value in qr.items()}
        codes["two"].qr.append(LabelQr("50087149", "2150087149LDS5015481"))
        records = {key: {"label_crop": key, "original_image_path": key} for key in qr}
        records["match"]["part_no"] = "50087149"
        records["other_part"]["part_no"] = "50087288"
        records["dup1"]["original_image_path"] = records["dup2"]["original_image_path"] = "photo.jpg"
        known = {"50087149": "AP162E", "98013170": "S110-4P1T-BU-V2"}
        sn_results, sn_reports, part_no_by_key, part_no_src, models = {}, {}, {}, {}, {}
        with mock.patch.object(pipeline, "model_from_part_no_text",
                               lambda text: (known.get(text, ""), text)):
            handled = pipeline._apply_label_qr_first(
                sorted(qr), records, codes, sn_results, sn_reports, part_no_by_key, part_no_src, models,
            )
        self.assertEqual(handled, {"match", "no_manifest"})
        self.assertEqual(sn_results["match"][0], "2150087149LDS5014813")
        self.assertEqual(sn_results["match"][2], "barcode_label_qr")
        self.assertEqual(sn_results["match"][3]["label_qr_check"], "qr_first")
        self.assertEqual(sn_reports["match"].decoder_name, "label_qr")
        self.assertNotIn("match", part_no_by_key)
        self.assertEqual(part_no_by_key["no_manifest"], "98013170")
        self.assertEqual(part_no_src["no_manifest"], "label_qr")
        self.assertEqual(models["no_manifest"][0], "S110-4P1T-BU-V2")
        self.assertEqual(models["no_manifest"][2], "part_no_label_qr")

    def test_crop_part_no_skips_slow_decode_when_label_qr_known(self):
        import crop
        import scan2

        img = np.full((40, 200, 3), 255, dtype=np.uint8)
        with mock.patch.object(scan2, "_read_part_no_fast", return_value=[]) as fast, \
                mock.patch.object(crop, "_decode_raw_part_no_crop_once", return_value=["slow"]) as slow:
            crop._STAGE2_LABEL_QR.part_no = "50087149"
            try:
                self.assertEqual(crop.decode_raw_part_no_crop(img), [])
            finally:
                crop._STAGE2_LABEL_QR.part_no = ""
            self.assertEqual(crop.decode_raw_part_no_crop(img), ["slow"])
        fast.assert_called_once()
        slow.assert_called_once()

    def test_label_qr_sources_count_as_barcode_hits(self):
        from huaweiocr.recognition.metrics import new_stats, record_sn_metrics

        stats = new_stats()
        record_sn_metrics(stats, True, "21500871494ERB017591", "barcode_label_qr",
                          {"barcode_status": "hit", "label_qr_check": "rescued"})
        record_sn_metrics(stats, True, "2150087149LDS5015385", "barcode_qr_conflict", {"barcode_status": "hit"})
        record_sn_metrics(stats, True, "2150087149LDS5014813", "barcode_label_qr",
                          {"barcode_status": "hit", "label_qr_check": "qr_first"})
        self.assertEqual(stats["sn_barcode_hits"], 3)
        self.assertEqual(stats["sn_label_qr_rescues"], 1)
        self.assertEqual(stats["sn_label_qr_first"], 1)
        self.assertEqual(stats["sn_label_qr_conflicts"], 1)
        self.assertEqual(stats["sn_ocr_recoveries"], 0)

    def test_sources_have_display_names(self):
        from huaweiocr.core.sources import display_source

        self.assertEqual(display_source("barcode_label_qr"), "标签二维码")
        self.assertIn("需复核", display_source("barcode_qr_conflict"))
        self.assertIn("需复核", display_source("part_no_product_conflict"))
        self.assertIn("需复核", display_source("ocr_repair_conflict"))


class RepairOcrCheckTest(unittest.TestCase):
    def _run(self, repaired, ocr_sn, part_no="50087149", kinds=(), ocr_src="ocr", alternatives=()):
        sn_results = {"k": (repaired, "[BARCODE]", "barcode", {"barcode_status": "hit"})}
        pipeline._apply_repair_ocr_check(
            "k", (ocr_sn, "", ocr_src, {}), sn_results, part_no, kinds, alternatives,
        )
        return sn_results["k"]

    def test_default_template_allows_2025_dated_4e_serials(self):
        from huaweiocr.barcode import linear_repair
        from huaweiocr.barcode.sn import DEFAULT_PIXEL_REPAIR_TEMPLATE

        templates = linear_repair.parse_templates(DEFAULT_PIXEL_REPAIR_TEMPLATE)
        four_e = [t for t in templates if len(t) == 12]
        self.assertEqual(len(four_e), 1)
        for sn in ("4E25C0085603", "4E25A0068800", "4E2640085582"):
            self.assertTrue(linear_repair.text_matches_template(sn, four_e[0]), sn)

    def test_agreement_confirms_repair(self):
        sn, _raw, src, meta = self._run("21500871494ES4051146", "21500871494ES4051146")
        self.assertEqual((sn, src, meta["sn_repair_check"]), ("21500871494ES4051146", "barcode", "ocr_confirmed"))

    def test_ocr_fills_only_positions_the_repair_is_unsure_about(self):
        alts = ("21500871494ES4051145", "21500871494ES4051278", "21500871494ES4051178")
        sn, _raw, src, meta = self._run("21500871494ES4051145", "21500871494ES4051146", alternatives=alts)
        self.assertEqual((sn, src), ("21500871494ES4051146", "ocr_repair_conflict"))
        self.assertEqual(meta["sn_repair_check"], "ocr_conflict")
        self.assertEqual(meta["sn_repair_other"], "21500871494ES4051145")

    def test_confident_repair_beats_ocr_misread(self):
        # Zhangjiakou label_5: all close candidates read 4E25A, OCR read 4E26A.
        alts = ("4E25A0068800",)
        sn, _raw, src, meta = self._run("4E25A0068800", "4E26A0068800", alternatives=alts)
        self.assertEqual((sn, src), ("4E25A0068800", "barcode"))
        self.assertEqual(meta["sn_repair_check"], "ocr_conflict")
        self.assertEqual(meta["sn_repair_other"], "4E26A0068800")
        sn, _raw, src, _meta = self._run("21500871494ES4051145", "21500871494ES4051146")
        self.assertEqual((sn, src), ("21500871494ES4051145", "barcode"))

    def test_ocr_confusable_mismatch_keeps_repair(self):
        alts = ("21500872884ERB006106", "21500872884ES3006106")
        sn, _raw, src, meta = self._run(
            "21500872884ERB006106", "21500872884ER8006106", part_no="50087288", alternatives=alts,
        )
        self.assertEqual((sn, src), ("21500872884ERB006106", "barcode"))
        self.assertEqual(meta["sn_repair_other"], "21500872884ER8006106")

    def test_fused_value_contradicting_part_no_keeps_repair(self):
        alts = ("21500871494ES4051145", "21500871504ES4051145")
        sn, _raw, src, _meta = self._run("21500871494ES4051145", "21500871504ES4051146", alternatives=alts)
        self.assertEqual((sn, src), ("21500871494ES4051145", "barcode"))

    def test_no_ocr_read_marks_repair_unverified(self):
        sn, _raw, src, meta = self._run("4E2640085582", "", ocr_src="none")
        self.assertEqual((sn, src, meta["sn_repair_check"]), ("4E2640085582", "barcode", "unverified"))
        sn, _raw, src, meta = self._run("4E2640085582", "4E2640085583", ocr_src="barcode_ocr_consensus")
        self.assertEqual(meta["sn_repair_check"], "unverified")

    def test_only_unconfirmed_repairs_are_rechecked(self):
        reports = {
            "repair": _report("4E2640085582", "code128_pixel_repair"),
            "qr_confirmed": _report("4E2640085583", "code128_pixel_repair"),
            "genuine": _report("4E2640085584", "pyzbar"),
            "rejected": _report("", "code128_pixel_repair", status="part_no_conflict"),
        }
        results = {key: (r.sn, "", "barcode", {}) for key, r in reports.items()}
        results["qr_confirmed"][3]["label_qr_check"] = "confirmed"
        self.assertEqual(pipeline._unverified_repair_keys(list(reports), results, reports), ["repair"])

    def test_ocr_override_counts_as_ocr_recovery_not_barcode_hit(self):
        from huaweiocr.recognition.metrics import new_stats, record_sn_metrics

        stats = new_stats()
        meta = {"barcode_status": "hit", "sn_repair_check": "ocr_conflict"}
        record_sn_metrics(stats, True, "21500871494ES4051146", "ocr_repair_conflict", meta)
        record_sn_metrics(stats, True, "21500872884ERB006106", "barcode", dict(meta))
        self.assertEqual(stats["sn_barcode_hits"], 1)
        self.assertEqual(stats["sn_ocr_recoveries"], 1)
        self.assertEqual(stats["sn_repair_ocr_conflicts"], 2)


if __name__ == "__main__":
    unittest.main()
