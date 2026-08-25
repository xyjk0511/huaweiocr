import os
import json
import tempfile
import unittest
from unittest import mock

import scan2
from huaweiocr.barcode import sn as sn_barcode
from huaweiocr.barcode import sn_rules


class SnRuleLearningTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ,
            {
                "HUAWEIOCR_LEARNED_SN_FAMILIES_PATH": os.path.join(
                    self.temp_dir.name, "families.json"
                ),
                "HUAWEIOCR_SN_FAMILY_CANDIDATES_PATH": os.path.join(
                    self.temp_dir.name, "candidates.json"
                ),
            },
        )
        self.env.start()
        sn_rules.reload_sn_rules()

    def tearDown(self):
        sn_rules.reload_sn_rules()
        self.env.stop()
        self.temp_dir.cleanup()

    def test_promotes_standard_family_only_after_two_physical_sources(self):
        first = "21500108452TC1500363"
        second = "21500108453TC1500364"

        result = sn_rules.commit_sn_observations(
            [
                {
                    "value": first,
                    "label_id": "photo_a__label_1",
                    "original_image_path": "C:/photos/a.jpg",
                },
                {
                    "value": second,
                    "label_id": "photo_a__label_2",
                    "original_image_path": "C:/photos/a.jpg",
                },
            ]
        )
        self.assertEqual(result["promoted"], [])
        self.assertEqual(sn_rules.match_learned_sn(first), "")

        result = sn_rules.commit_sn_observations(
            [
                {
                    "value": second,
                    "label_id": "photo_b__label_1",
                    "original_image_path": "C:/photos/b.jpg",
                }
            ]
        )
        self.assertEqual(len(result["promoted"]), 1)
        self.assertEqual(sn_rules.match_learned_sn(first), first)
        self.assertEqual(sn_rules.match_learned_sn(second), second)

    def test_rejects_nonstandard_shape(self):
        result = sn_rules.commit_sn_observations(
            [
                {
                    "value": "ABC12345XYZ9",
                    "label_id": "a__label_1",
                    "original_image_path": "C:/photos/a.jpg",
                }
            ]
        )
        self.assertEqual(result["promoted"], [])
        self.assertEqual(result["rejected"][0]["reason"], "nonstandard_or_ambiguous_envelope")

    def test_does_not_promote_generic_ss_family_from_unverified_prefix(self):
        result = sn_rules.commit_sn_observations(
            [
                {
                    "value": "21500772777SS4049560",
                    "label_id": "a__label_1",
                    "original_image_path": "C:/photos/a.jpg",
                },
                {
                    "value": "21500772778SS4049561",
                    "label_id": "b__label_1",
                    "original_image_path": "C:/photos/b.jpg",
                },
            ]
        )
        self.assertEqual(result["promoted"], [])
        self.assertEqual(len(result["rejected"]), 2)

    def test_strict_sn_field_marks_learning_evidence(self):
        value = "21500108452TC1500363"
        report = sn_barcode.SnBarcodeReport(
            status="parse_failure",
            attempts=1,
            decoded_count=1,
            results=[sn_barcode.DecoderResult("zxingcpp", value, "sn", "sn.0")],
        )
        meta = report.to_meta()
        meta["ocr_text_found"] = False
        with mock.patch.object(scan2, "load_for_ocr_color", return_value=object()):
            with mock.patch.object(
                scan2,
                "ocr_text_with_details",
                return_value=(f"S/N: {value}", f"S/N:{value}", []),
            ):
                sn, _raw, source, result_meta = scan2.recognize_sn_ocr_after_barcode(
                    "sn.png", barcode_report=report, meta=meta
                )
        self.assertEqual(sn, value)
        self.assertEqual(source, "barcode_ocr_consensus")
        self.assertEqual(result_meta["sn_learning_candidate"], value)

    def test_main_promotes_two_source_sn_family_after_rows_are_emitted(self):
        stage2 = os.path.join(self.temp_dir.name, "stage2_fields")
        model_dir = os.path.join(stage2, "model")
        sn_dir = os.path.join(stage2, "sn")
        os.makedirs(model_dir)
        os.makedirs(sn_dir)
        values = {
            "photo_a__label_1": "21500108452TC1500363",
            "photo_b__label_1": "21500108453TC1500364",
        }
        with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
            for label_id, value in values.items():
                sn_path = os.path.join(sn_dir, f"{label_id}__sn.png")
                open(sn_path, "wb").close()
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "sn_path": sn_path,
                            "original_image_path": os.path.join(
                                self.temp_dir.name, label_id.split("__", 1)[0] + ".jpg"
                            ),
                        }
                    )
                    + "\n"
                )

        def fake_barcode(_sn_path, label_id="", **_kwargs):
            value = values[label_id]
            report = sn_barcode.SnBarcodeReport(
                status="parse_failure",
                attempts=1,
                decoded_count=1,
                results=[sn_barcode.DecoderResult("zxingcpp", value, "sn", "sn.0")],
            )
            meta = report.to_meta()
            meta["ocr_text_found"] = False
            return "", value, "barcode_parse_fail", meta, report

        def fake_ocr(_sn_path, label_id="", barcode_report=None, meta=None):
            value = values[label_id]
            result_meta = dict(meta or {})
            result_meta["ocr_text_found"] = True
            result_meta["sn_learning_candidate"] = value
            return value, f"S/N: {value}", "barcode_ocr_consensus", result_meta

        with mock.patch.dict(
            os.environ,
            {
                "SCAN2_PARALLEL": "0",
                "SCAN2_SCAN_LABEL_WITH_SN": "0",
                "SCAN2_OCR_FALLBACK": "1",
            },
            clear=False,
        ):
            with mock.patch.object(scan2, "_recognize_sn_barcode", side_effect=fake_barcode):
                with mock.patch.object(
                    scan2, "recognize_sn_ocr_after_barcode", side_effect=fake_ocr
                ):
                    stats = scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=os.path.join(self.temp_dir.name, "out.jsonl"),
                        debug_log=os.path.join(self.temp_dir.name, "debug.log"),
                    )

        self.assertEqual(stats["sn_families_learned"], 1)
        self.assertEqual(sn_rules.match_learned_sn(values["photo_a__label_1"]), values["photo_a__label_1"])


class ModelLearningTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ,
            {
                "SCAN2_MODEL_LEARNING_CANDIDATES_PATH": os.path.join(
                    self.temp_dir.name, "model_candidates.json"
                )
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp_dir.cleanup()

    def test_strict_model_field_extraction(self):
        self.assertEqual(scan2.extract_anchored_unknown_model("Model: AR999\n"), "AR999")
        self.assertEqual(scan2.extract_anchored_unknown_model("description AR999"), "")
        self.assertEqual(scan2.extract_anchored_unknown_model("MODEL: DESCRIPTION"), "")

    def test_model_pair_promotes_after_two_physical_sources(self):
        from huaweiocr.core.model_learning import commit_model_observations

        first = commit_model_observations(
            [
                {
                    "part_no": "50099999",
                    "model": "AR999",
                    "label_id": "a__label_1",
                    "original_image_path": "C:/photos/a.jpg",
                },
                {
                    "part_no": "50099999",
                    "model": "AR999",
                    "label_id": "a__label_2",
                    "original_image_path": "C:/photos/a.jpg",
                },
            ]
        )
        self.assertEqual(first["promoted"], [])

        second = commit_model_observations(
            [
                {
                    "part_no": "50099999",
                    "model": "AR999",
                    "label_id": "b__label_1",
                    "original_image_path": "C:/photos/b.jpg",
                }
            ]
        )
        self.assertEqual(
            second["promoted"],
            [{"part_no": "50099999", "model": "AR999"}],
        )

    def test_conflicting_models_do_not_promote(self):
        from huaweiocr.core.model_learning import commit_model_observations

        result = commit_model_observations(
            [
                {
                    "part_no": "50099999",
                    "model": "AR999",
                    "label_id": "a__label_1",
                    "original_image_path": "C:/photos/a.jpg",
                },
                {
                    "part_no": "50099999",
                    "model": "AR998",
                    "label_id": "b__label_1",
                    "original_image_path": "C:/photos/b.jpg",
                },
            ]
        )
        self.assertEqual(result["promoted"], [])
        self.assertTrue(result["conflicts"])

    def test_main_outputs_strong_unknown_model_candidate_without_same_batch_promotion(self):
        stage2 = os.path.join(self.temp_dir.name, "stage2_fields")
        model_dir = os.path.join(stage2, "model")
        sn_dir = os.path.join(stage2, "sn")
        os.makedirs(model_dir)
        os.makedirs(sn_dir)
        label_id = "photo_a__label_1"
        model_path = os.path.join(model_dir, f"{label_id}__model.png")
        original_path = os.path.join(self.temp_dir.name, "photo_a.jpg")
        open(model_path, "wb").close()
        with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "label_id": label_id,
                        "model_path": model_path,
                        "part_no": "50099999",
                        "original_image_path": original_path,
                    }
                )
                + "\n"
            )
        out_path = os.path.join(self.temp_dir.name, "out.jsonl")
        env = {
            "SCAN2_PARALLEL": "0",
            "SCAN2_MODEL_BARCODE": "0",
            "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
            "SCAN2_PART_NO_MODEL_MAP_PATH": os.path.join(self.temp_dir.name, "map.json"),
            "SCAN2_LEARNED_MODEL_CODES_PATH": os.path.join(self.temp_dir.name, "models.json"),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(
                scan2,
                "recognize_model_ocr",
                return_value=("", "Model: AR999\n", "none"),
            ):
                stats = scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=out_path,
                    debug_log=os.path.join(self.temp_dir.name, "debug.log"),
                )

        with open(out_path, "r", encoding="utf-8") as f:
            row = json.loads(f.readline())
        self.assertEqual(row["model"], "AR999")
        self.assertEqual(row["model_src"], "ocr_candidate")
        self.assertEqual(row["part_no"], "50099999")
        self.assertEqual(stats["model_consensus_learned"], 0)
        self.assertFalse(os.path.exists(env["SCAN2_LEARNED_MODEL_CODES_PATH"]))

    def test_two_source_model_learning_is_used_on_next_batch(self):
        stage2 = os.path.join(self.temp_dir.name, "stage2_fields")
        model_dir = os.path.join(stage2, "model")
        sn_dir = os.path.join(stage2, "sn")
        os.makedirs(model_dir)
        os.makedirs(sn_dir)
        manifest_path = os.path.join(stage2, "manifest.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as manifest:
            for stem in ("photo_a", "photo_b"):
                label_id = f"{stem}__label_1"
                model_path = os.path.join(model_dir, f"{label_id}__model.png")
                open(model_path, "wb").close()
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "model_path": model_path,
                            "part_no": "50099999",
                            "original_image_path": os.path.join(
                                self.temp_dir.name, f"{stem}.jpg"
                            ),
                        }
                    )
                    + "\n"
                )

        map_path = os.path.join(self.temp_dir.name, "map.json")
        learned_path = os.path.join(self.temp_dir.name, "models.json")
        env = {
            "SCAN2_PARALLEL": "0",
            "SCAN2_MODEL_BARCODE": "0",
            "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
            "SCAN2_PART_NO_MODEL_MAP_PATH": map_path,
            "SCAN2_LEARNED_MODEL_CODES_PATH": learned_path,
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(
                scan2,
                "recognize_model_ocr",
                return_value=("", "Model: AR999\n", "none"),
            ):
                first_stats = scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=os.path.join(self.temp_dir.name, "first.jsonl"),
                    debug_log=os.path.join(self.temp_dir.name, "debug.log"),
                )

            self.assertEqual(first_stats["model_consensus_learned"], 1)
            self.assertEqual(first_stats["model_part_no_learned"], 1)
            with open(learned_path, "r", encoding="utf-8") as f:
                self.assertIn("AR999", json.load(f))
            with open(map_path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["50099999"]["model"], "AR999")

            with mock.patch.object(scan2, "recognize_model_ocr") as model_ocr:
                scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=os.path.join(self.temp_dir.name, "second.jsonl"),
                    debug_log=os.path.join(self.temp_dir.name, "debug2.log"),
                )
            model_ocr.assert_not_called()

        with open(os.path.join(self.temp_dir.name, "second.jsonl"), "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual([row["model"] for row in rows], ["AR999", "AR999"])
        self.assertEqual(
            [row["model_src"] for row in rows],
            ["part_no_barcode", "part_no_barcode"],
        )

    def test_two_source_model_barcode_candidate_learns_without_ocr(self):
        stage2 = os.path.join(self.temp_dir.name, "stage2_fields")
        model_dir = os.path.join(stage2, "model")
        sn_dir = os.path.join(stage2, "sn")
        os.makedirs(model_dir)
        os.makedirs(sn_dir)
        with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
            for stem in ("photo_a", "photo_b"):
                label_id = f"{stem}__label_1"
                model_path = os.path.join(model_dir, f"{label_id}__model.png")
                open(model_path, "wb").close()
                manifest.write(
                    json.dumps(
                        {
                            "label_id": label_id,
                            "model_path": model_path,
                            "part_no": "50099999",
                            "original_image_path": os.path.join(
                                self.temp_dir.name, f"{stem}.jpg"
                            ),
                        }
                    )
                    + "\n"
                )

        map_path = os.path.join(self.temp_dir.name, "map.json")
        learned_path = os.path.join(self.temp_dir.name, "models.json")
        out_path = os.path.join(self.temp_dir.name, "out.jsonl")
        env = {
            "SCAN2_PARALLEL": "0",
            "SCAN2_MODEL_BARCODE": "1",
            "SCAN2_SCAN_LABEL_WITHOUT_SN": "0",
            "SCAN2_PART_NO_MODEL_MAP_PATH": map_path,
            "SCAN2_LEARNED_MODEL_CODES_PATH": learned_path,
        }
        barcode_result = ("", "model:AR999; model_band:AR999", "barcode_no_match")
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(
                scan2, "recognize_model_barcode", return_value=barcode_result
            ):
                with mock.patch.object(
                    scan2,
                    "recognize_model_ocr",
                    side_effect=AssertionError("model barcode candidate should skip OCR"),
                ):
                    stats = scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=out_path,
                        debug_log=os.path.join(self.temp_dir.name, "debug.log"),
                    )

        with open(out_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual([row["model"] for row in rows], ["AR999", "AR999"])
        self.assertEqual(
            [row["model_src"] for row in rows],
            ["barcode_candidate", "barcode_candidate"],
        )
        self.assertEqual(stats["model_barcode_hits"], 2)
        self.assertEqual(stats["model_ocr_recoveries"], 0)
        self.assertEqual(stats["model_consensus_learned"], 1)
        self.assertEqual(stats["model_part_no_learned"], 1)
        with open(learned_path, "r", encoding="utf-8") as f:
            self.assertIn("AR999", json.load(f))
        with open(map_path, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["50099999"]["model"], "AR999")


if __name__ == "__main__":
    unittest.main()
