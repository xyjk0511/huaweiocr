import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


fake_inference_sdk = types.ModuleType("inference_sdk")
fake_inference_sdk.InferenceHTTPClient = object
sys.modules.setdefault("inference_sdk", fake_inference_sdk)

import crop  # noqa: E402  (fake inference_sdk must be registered before importing crop)


def draw_1d_barcode(img, x1, y1, x2, y2, step=4, bar_width=2):
    for x in range(x1, x2, step):
        img[y1:y2, x:min(x + bar_width, x2)] = 0


def draw_top_right_product_mark(img):
    rects = [
        (338, 24, 382, 58),
        (398, 28, 430, 54),
        (350, 78, 386, 106),
        (405, 82, 444, 116),
        (366, 138, 410, 170),
    ]
    for x1, y1, x2, y2 in rects:
        img[y1:y2, x1:x2] = 0


def draw_product_field_structure(img):
    h, w = img.shape[:2]
    draw_1d_barcode(
        img,
        max(12, int(w * 0.08)),
        max(24, int(h * 0.34)),
        max(120, int(w * 0.64)),
        max(48, int(h * 0.46)),
        step=4,
        bar_width=2,
    )


class CropFieldPaddingTests(unittest.TestCase):
    def test_stage1_raw_label_mode_defaults_enabled_for_local_backend(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CROP_INFERENCE_BACKEND", None)
            os.environ.pop("CROP_STAGE1_USE_RAW_LABEL_DETECTIONS", None)
            os.environ.pop("CROP_STAGE1_KEEP_ALL_CROPS", None)
            self.assertTrue(crop.stage1_raw_label_mode_enabled())
            self.assertTrue(crop.stage1_keep_all_crops_enabled())

    def test_stage1_collect_raw_label_crops_keeps_all_non_conflicting_boxes(self):
        img = np.full((300, 500, 3), 255, dtype=np.uint8)
        preds = [
            {"x": 120, "y": 90, "width": 180, "height": 70, "class": crop.MODEL1_LABEL_CLASS, "confidence": 0.92},
            {"x": 380, "y": 210, "width": 160, "height": 68, "class": crop.MODEL1_LABEL_CLASS, "confidence": 0.88},
        ]

        crops, dropped = crop._stage1_collect_raw_label_crops(img, preds)

        self.assertEqual(dropped, 0)
        self.assertEqual(len(crops), 2)
        self.assertTrue(all(crop_img is not None and crop_img.size > 0 for crop_img in crops))

    def test_stage1_collect_raw_label_crops_rejects_edge_near_square_box(self):
        img = np.full((300, 500, 3), 255, dtype=np.uint8)
        preds = [
            {"x": 120, "y": 90, "width": 180, "height": 70, "class": crop.MODEL1_LABEL_CLASS, "confidence": 0.92},
            {"x": 470, "y": 262, "width": 58, "height": 56, "class": crop.MODEL1_LABEL_CLASS, "confidence": 0.98},
        ]

        crops, dropped = crop._stage1_collect_raw_label_crops(img, preds)

        self.assertEqual(dropped, 0)
        self.assertEqual(len(crops), 1)
        self.assertTrue(crops[0] is not None and crops[0].size > 0)

    def test_stage1_save_preview_image_writes_overlay(self):
        img = np.full((240, 360, 3), 255, dtype=np.uint8)
        entries = [
            {"box": (20, 30, 180, 110), "confidence": 0.91},
            {"box": (190, 120, 330, 210), "confidence": 0.88},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = os.path.join(tmpdir, "previews")
            with mock.patch.object(crop, "STAGE1_PREVIEW_DIR", preview_dir):
                out = crop._stage1_save_preview_image(
                    img,
                    r"F:\dummy\sample.jpg",
                    entries,
                )
                self.assertIsNotNone(out)
                self.assertTrue(os.path.exists(out))

    def test_infer_with_resize_prefers_local_original_path_for_stage1_pt_model(self):
        img = np.full((200, 300, 3), 255, dtype=np.uint8)

        class StubClient:
            def supports_original_path_inference(self, model_id=None):
                return model_id == crop.MODEL1_ID

            def infer_original_path(self, image_path, model_id=None):
                return {"predictions": [{"class": crop.MODEL1_LABEL_CLASS, "confidence": 0.44}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "sample.jpg")
            crop.save_png_required(img_path, img, "sample image")
            with mock.patch.object(crop, "inference_backend", return_value="local"):
                with mock.patch.object(crop, "get_inference_client", return_value=StubClient()):
                    preds = crop.infer_with_resize(img, img_path, model_id=crop.MODEL1_ID)

        self.assertEqual(preds, [{"class": crop.MODEL1_LABEL_CLASS, "confidence": 0.44}])

    def test_asymmetric_crop_expands_toward_adjacent_barcode(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        pred = {"x": 100, "y": 50, "width": 40, "height": 20}

        out = crop.crop_from_pred_asym(
            img,
            pred,
            pad_x_ratio=0.25,
            pad_top_ratio=1.80,
            pad_bottom_ratio=0.35,
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.shape[:2], (63, 60))

    def test_model_crop_without_lower_barcode_keeps_model_line_only(self):
        img = np.full((140, 260, 3), 255, dtype=np.uint8)
        pred = {"x": 90, "y": 50, "width": 100, "height": 30}

        out = crop.crop_model_field(img, pred)

        self.assertIsNotNone(out)
        self.assertEqual(out.shape[1], 140)
        self.assertGreaterEqual(out.shape[0], 40)
        self.assertLessEqual(out.shape[0], 52)

    def test_model_crop_without_barcode_keeps_full_text_line(self):
        img = np.full((160, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Model: S380-S8P2T",
            (35, 60),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.putText(
            img,
            "Desc: S380-S8P2T, 2*GE WAN",
            (35, 105),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            crop.cv2.LINE_AA,
        )
        pred = {"x": 150, "y": 55, "width": 210, "height": 52}

        out = crop.crop_model_field(img, pred)

        self.assertIsNotNone(out)
        self.assertGreaterEqual(out.shape[0], 28)
        self.assertLess(out.shape[0], 65)
        dark_rows = (out[:, :, 0] < 128).mean(axis=1)
        active_rows = np.where(dark_rows > 0.01)[0]
        self.assertGreater(active_rows[0], 1)
        self.assertLess(active_rows[-1], out.shape[0] - 2)

    def test_fallback_model_crop_from_sn_keeps_text_and_barcode(self):
        img = np.full((600, 1000, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Model: AR180Pro",
            (110, 170),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        for x in range(110, 430, 8):
            img[190:240, x:x + 3] = 0
        crop.cv2.putText(
            img,
            "Desc: AR180Pro",
            (110, 280),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            crop.cv2.LINE_AA,
        )
        sn_pred = {"x": 350, "y": 395, "width": 500, "height": 100}

        out = crop.fallback_model_crop_from_sn(img, sn_pred)

        self.assertIsNotNone(out)
        self.assertGreaterEqual(out.shape[0], 90)
        self.assertLess(out.shape[0], 155)
        dark_rows = (out[:, :, 0] < 128).mean(axis=1)
        self.assertGreater(np.count_nonzero(dark_rows > 0.12), 45)

    def test_fallback_model_crop_from_sn_keeps_full_model_barcode_band(self):
        img = np.full((620, 1000, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Model: AP362E",
            (110, 170),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        for x in range(110, 440, 8):
            img[188:246, x:x + 3] = 0
        crop.cv2.putText(
            img,
            "Desc: AP362E(11ax indoor,2+2 dual-band)",
            (110, 282),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            crop.cv2.LINE_AA,
        )
        sn_pred = {"x": 360, "y": 405, "width": 520, "height": 95}

        out = crop.fallback_model_crop_from_sn(img, sn_pred)

        self.assertIsNotNone(out)
        self.assertGreaterEqual(out.shape[0], 105)
        dark_rows = (out[:, :, 0] < 128).mean(axis=1)
        self.assertGreater(np.count_nonzero(dark_rows > 0.18), 55)

    def test_fallback_model_crop_from_part_no_keeps_text_and_barcode(self):
        img = np.full((300, 520, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087290",
            (24, 42),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 26, 56, 234, 90, step=4, bar_width=2)
        crop.cv2.putText(
            img,
            "Model: AP162E",
            (24, 126),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 26, 140, 234, 176, step=4, bar_width=2)

        with mock.patch.object(crop, "_stage2_part_no_box_from_pred", return_value=(18, 18, 248, 102)):
            out = crop.fallback_model_crop_from_part_no(img, {"x": 120, "y": 60, "width": 180, "height": 66})

        self.assertIsNotNone(out)
        self.assertGreaterEqual(out.shape[0], 55)
        self.assertGreaterEqual(out.shape[1], 210)
        dark_rows = (out[:, :, 0] < 128).mean(axis=1)
        self.assertGreater(np.count_nonzero(dark_rows > 0.10), 28)

    def test_model_crop_with_lower_barcode_includes_barcode_band(self):
        img = np.full((140, 260, 3), 255, dtype=np.uint8)
        for x in range(35, 175, 6):
            img[76:108, x:x + 3] = 0
        pred = {"x": 90, "y": 50, "width": 100, "height": 30}

        with mock.patch.object(crop, "model_box_decodes_as_barcode", return_value=True):
            out = crop.crop_model_field(img, pred)

        self.assertIsNotNone(out)
        self.assertGreaterEqual(out.shape[0], 70)

    def test_model_crop_with_inside_barcode_drops_upper_neighbor_fragment(self):
        img = np.full((160, 300, 3), 255, dtype=np.uint8)
        for x in range(45, 190, 6):
            img[30:40, x:x + 3] = 0
            img[86:128, x:x + 3] = 0
        img[62:72, 45:130] = 0
        pred = {"x": 112, "y": 76, "width": 140, "height": 96}

        with mock.patch.object(crop, "model_box_decodes_as_barcode", return_value=True):
            out = crop.crop_model_field(img, pred)

        self.assertIsNotNone(out)
        self.assertLess(out.shape[0], 100)
        self.assertGreaterEqual(out.shape[0], 70)
        dark_rows = (out[:, :, 0] < 128).mean(axis=1)
        self.assertGreater(np.count_nonzero(dark_rows > 0.12), 35)

    def test_model_crop_with_inside_barcode_keeps_slanted_model_text_margin(self):
        img = np.full((180, 360, 3), 255, dtype=np.uint8)
        text_layer = np.full_like(img, 255)
        crop.cv2.putText(
            text_layer,
            "Model: S380-S8P2T",
            (48, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        matrix = crop.cv2.getRotationMatrix2D((160, 70), -6, 1.0)
        tilted = crop.cv2.warpAffine(
            text_layer,
            matrix,
            (360, 180),
            flags=crop.cv2.INTER_LINEAR,
            borderValue=(255, 255, 255),
        )
        img = np.minimum(img, tilted)
        for x in range(54, 260, 7):
            img[100:132, x:x + 3] = 0
        pred = {"x": 156, "y": 88, "width": 220, "height": 104}

        with mock.patch.object(crop, "model_box_decodes_as_barcode", return_value=True):
            out = crop.crop_model_field(img, pred)

        self.assertIsNotNone(out)
        dark_rows = (out[:, :, 0] < 128).mean(axis=1)
        active_rows = np.where(dark_rows > 0.05)[0]
        self.assertGreaterEqual(active_rows[0], 5)
        self.assertLessEqual(active_rows[-1], out.shape[0] - 2)

    def test_model_crop_satisfies_target_requires_nearby_barcode(self):
        img = np.full((220, 500, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Model: AP362E",
            (55, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        for x in range(55, 265, 7):
            img[98:132, x:x + 3] = 0
        pred = {"x": 150, "y": 70, "width": 190, "height": 42}
        text_only = img[38:88, 45:255]

        with mock.patch.object(crop, "model_box_decodes_as_barcode", return_value=True):
            full = crop.crop_model_field(img, pred)
            self.assertTrue(crop.model_pred_has_lower_barcode(img, pred))
            with mock.patch.object(crop, "model_crop_has_decodable_barcode", return_value=True):
                self.assertFalse(crop.model_crop_satisfies_target(text_only, img, pred))
                self.assertTrue(crop.model_crop_satisfies_target(full, img, pred))
        self.assertTrue(crop.crop_has_complete_1d_barcode(full))

    def test_model_crop_satisfies_target_rejects_unscannable_lower_barcode(self):
        img = np.full((220, 500, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Model: AP362E",
            (55, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        for x in range(55, 265, 7):
            img[98:132, x:x + 3] = 0
        pred = {"x": 150, "y": 70, "width": 190, "height": 42}
        with mock.patch.object(crop, "model_box_decodes_as_barcode", return_value=True):
            full = crop.crop_model_field(img, pred)

            with mock.patch.object(crop, "model_crop_has_decodable_barcode", return_value=False):
                self.assertFalse(crop.model_crop_satisfies_target(full, img, pred))

    def test_model_crop_when_detector_hits_barcode_keeps_text_above_it(self):
        img = np.full((220, 520, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Model: AP362E",
            (55, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        for x in range(55, 310, 7):
            img[98:132, x:x + 3] = 0
        pred = {"x": 182, "y": 115, "width": 250, "height": 48}

        with mock.patch.object(crop, "model_box_decodes_as_barcode", return_value=True):
            out = crop.crop_model_field(img, pred)

            self.assertTrue(crop.model_pred_has_lower_barcode(img, pred))
            self.assertTrue(crop.model_crop_has_complete_1d_barcode(out))
            self.assertTrue(crop.model_crop_has_text_above_barcode(out))
            with mock.patch.object(crop, "model_crop_has_decodable_barcode", return_value=True):
                self.assertTrue(crop.model_crop_satisfies_target(out, img, pred))

    def test_model_crop_satisfies_target_accepts_text_only_when_no_lower_barcode(self):
        img = np.full((140, 360, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Model: S380-S8P2T",
            (35, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        pred = {"x": 150, "y": 70, "width": 210, "height": 42}
        model_line = crop.crop_model_field(img, pred)

        self.assertFalse(crop.model_pred_has_lower_barcode(img, pred))
        with mock.patch.object(crop, "model_crop_has_decodable_barcode") as scan:
            self.assertTrue(crop.model_crop_satisfies_target(model_line, img, pred))
            scan.assert_not_called()

    def test_model_text_only_crop_trims_next_line_fragment(self):
        img = np.full((150, 420, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Model: S380-S8P2T",
            (25, 62),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.putText(
            img,
            "Desc.: S380-S8P2T",
            (25, 112),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        pred = {"x": 160, "y": 76, "width": 300, "height": 88}

        out = crop.crop_model_field(img, pred)

        self.assertIsNotNone(out)
        self.assertLess(out.shape[0], 58)
        lower_band = out[int(out.shape[0] * 0.82):, :, 0]
        self.assertLess((lower_band < 128).mean(), 0.08)

    def test_model_crop_ignores_distant_lower_barcode(self):
        img = np.full((180, 260, 3), 255, dtype=np.uint8)
        for x in range(35, 200, 6):
            img[120:150, x:x + 3] = 0
        pred = {"x": 90, "y": 50, "width": 100, "height": 30}

        out = crop.crop_model_field(img, pred)

        self.assertIsNotNone(out)
        self.assertEqual(out.shape[1], 140)
        self.assertGreaterEqual(out.shape[0], 40)
        self.assertLessEqual(out.shape[0], 52)
        self.assertFalse(crop.model_pred_has_lower_barcode(img, pred))

    def test_crop_has_complete_1d_barcode_rejects_edge_cut_barcodes(self):
        complete = np.full((90, 240, 3), 255, dtype=np.uint8)
        left_cut = np.full((90, 240, 3), 255, dtype=np.uint8)
        right_cut = np.full((90, 240, 3), 255, dtype=np.uint8)

        draw_1d_barcode(complete, 30, 24, 210, 64)
        draw_1d_barcode(left_cut, 0, 24, 180, 64)
        draw_1d_barcode(right_cut, 60, 24, 240, 64)

        self.assertTrue(crop.crop_has_complete_1d_barcode(complete))
        self.assertFalse(crop.crop_has_complete_1d_barcode(left_cut))
        self.assertFalse(crop.crop_has_complete_1d_barcode(right_cut))

    def test_normalize_part_no_codes_accepts_common_prefixes(self):
        self.assertEqual(
            crop.normalize_part_no_codes(
                [
                    "1P50087147",
                    "PN 98012125",
                    "P/N: 50087288",
                    "Part No.: 98012403",
                    "AP162E",
                    "50087147",
                ]
            ),
            ["50087147", "98012125", "50087288", "98012403"],
        )

    def test_part_no_crop_includes_part_no_barcode_band(self):
        img = np.full((420, 720, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Make SME Network Easier and Smarter",
            (120, 58),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.rectangle(img, (95, 105), (625, 335), (0, 0, 0), 2)
        crop.cv2.putText(
            img,
            "Part No.: 50087147",
            (118, 138),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 118, 148, 330, 178, step=3, bar_width=1)
        crop.cv2.putText(
            img,
            "Model: AP362E",
            (118, 214),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 118, 226, 360, 266, step=5, bar_width=2)

        out = crop.crop_part_no_field(img)

        self.assertIsNotNone(out)
        self.assertTrue(crop.crop_contains_1d_barcode(out, min_span_ratio=0.18))
        self.assertLess(out.shape[0], 230)

    def test_stage2_part_no_detector_crop_includes_complete_barcode(self):
        img = np.full((180, 360, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087147",
            (30, 40),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 34, 52, 190, 84, step=4, bar_width=2)
        pred = {
            "x": 112,
            "y": 57,
            "width": 168,
            "height": 58,
            "class": "partno",
            "confidence": 0.93,
        }

        out, source, ok = crop._stage2_crop_part_no(img, pred)

        self.assertEqual(source, "detector")
        self.assertTrue(ok)
        self.assertTrue(crop.crop_contains_1d_barcode(out, min_span_ratio=0.16))
        self.assertLess(out.shape[0], 140)

    def test_stage2_part_no_detector_crop_keeps_text_left_of_barcode(self):
        img = np.full((180, 360, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087147",
            (18, 40),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 92, 52, 226, 84, step=4, bar_width=2)
        pred = {
            "x": 159,
            "y": 68,
            "width": 134,
            "height": 36,
            "class": "partno",
            "confidence": 0.91,
        }

        out, source, ok = crop._stage2_crop_part_no(img, pred)

        self.assertEqual(source, "detector")
        self.assertTrue(ok)
        self.assertIsNotNone(out)
        self.assertGreaterEqual(out.shape[1], 210)
        self.assertTrue(crop.crop_has_complete_1d_barcode(out, min_span_ratio=0.18))

    def test_stage2_part_no_detector_crop_keeps_text_and_avoids_model_barcode(self):
        img = np.full((220, 520, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087149",
            (36, 42),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 40, 56, 250, 92, step=4, bar_width=2)
        crop.cv2.putText(
            img,
            "Model: AP162E",
            (36, 136),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 40, 150, 250, 188, step=4, bar_width=2)
        pred = {
            "x": 148,
            "y": 66,
            "width": 235,
            "height": 72,
            "class": "partno",
            "confidence": 0.93,
        }

        out, source, ok = crop._stage2_crop_part_no(img, pred)

        self.assertEqual(source, "detector")
        self.assertTrue(ok)
        self.assertIsNotNone(out)
        self.assertTrue(crop.crop_has_complete_1d_barcode(out, min_span_ratio=0.18))
        self.assertLessEqual(out.shape[0], 105)

    def test_part_no_crop_pads_bottom_quiet_zone(self):
        img = np.full((86, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087149",
            (18, 28),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 20, 38, 250, 80, step=4, bar_width=2)

        out = crop.pad_part_no_crop_quiet_zone(img)

        self.assertGreater(out.shape[0], img.shape[0])
        barcode_box = crop.crop_1d_barcode_box(
            out,
            min_span_ratio=0.18,
            row_trans_threshold=0.07,
            active_threshold=0.25,
        )
        self.assertIsNotNone(barcode_box)
        self.assertGreaterEqual(out.shape[0] - barcode_box[3], 16)
        self.assertLess((out[-8:, :, 0] < 128).mean(), 0.01)

    def test_part_no_crop_pads_side_quiet_zones(self):
        img = np.full((86, 286, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087149",
            (8, 28),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 3, 38, 276, 80, step=4, bar_width=2)

        out = crop.pad_part_no_crop_quiet_zone(img)

        self.assertGreater(out.shape[1], img.shape[1])
        barcode_box = crop.crop_1d_barcode_box(
            out,
            min_span_ratio=0.18,
            row_trans_threshold=0.07,
            active_threshold=0.25,
        )
        self.assertIsNotNone(barcode_box)
        self.assertGreaterEqual(barcode_box[0], 14)
        self.assertGreaterEqual(out.shape[1] - barcode_box[2], 14)
        self.assertLess((out[:, :8, 0] < 128).mean(), 0.01)
        self.assertLess((out[:, -8:, 0] < 128).mean(), 0.01)

    def test_part_no_polished_crop_accepts_white_bottom_padding(self):
        img = np.full((86, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087149",
            (18, 28),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 20, 38, 250, 80, step=4, bar_width=2)
        padded = crop.pad_part_no_crop_quiet_zone(img)
        dirty = padded.copy()
        dirty[-6:, 40:110] = 0

        self.assertTrue(crop.part_no_polished_crop_is_safe(img, padded))
        self.assertFalse(crop.part_no_polished_crop_is_safe(img, dirty))

    def test_part_no_polished_crop_rejects_barcode_only_padding(self):
        img = np.full((58, 320, 3), 255, dtype=np.uint8)
        draw_1d_barcode(img, 20, 8, 250, 50, step=4, bar_width=2)
        padded = crop.pad_part_no_crop_quiet_zone(img)

        self.assertFalse(crop.part_no_polished_crop_is_safe(img, padded))

    def test_part_no_polished_crop_rejects_edge_clipped_barcode_padding(self):
        img = np.full((86, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087290",
            (18, 28),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 1, 38, 250, 80, step=4, bar_width=2)
        padded = crop.pad_part_no_crop_quiet_zone(img)

        self.assertTrue(crop.part_no_crop_has_edge_clipped_barcode(img))
        self.assertFalse(crop.part_no_polished_crop_is_safe(img, padded))

    def test_stage2_build_candidate_saves_safe_padded_part_no_on_scan_miss(self):
        img = np.full((160, 380, 3), 255, dtype=np.uint8)
        part_no_crop = np.full((86, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            part_no_crop,
            "Part No.: 50087149",
            (18, 28),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(part_no_crop, 20, 38, 250, 80, step=4, bar_width=2)
        padded = crop.pad_part_no_crop_quiet_zone(part_no_crop)

        with mock.patch.object(crop, "stage2_save_model_crops_enabled", return_value=False):
            with mock.patch.object(crop, "_stage2_crop_part_no", return_value=(part_no_crop, "detector", True)):
                with mock.patch.object(crop, "polish_part_no_crop_for_scan_miss", return_value=padded):
                    with mock.patch.object(crop, "_stage2_crop_sn", return_value=None):
                        with mock.patch.object(crop, "decode_raw_part_no_crop", side_effect=[[], []]):
                            candidate = crop._stage2_build_candidate(img, [], rotation=0)

        self.assertEqual(candidate["part_no_codes"], [])
        self.assertGreater(candidate["part_no_crop"].shape[0], part_no_crop.shape[0])
        self.assertLess((candidate["part_no_crop"][-8:, :, 0] < 128).mean(), 0.01)

    def test_stage2_infer_skips_high_res_pass_when_required_fields_are_found(self):
        img = np.full((160, 380, 3), 255, dtype=np.uint8)
        preds1 = [
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 120, "y": 50, "width": 160, "height": 42},
            {"class": crop.MODEL2_SN_CLASS, "x": 135, "y": 120, "width": 190, "height": 44},
        ]

        with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
            with mock.patch.object(crop, "infer_with_resize", return_value=preds1) as infer_mock:
                out = crop._stage2_infer_field_preds(img, "label.png")

        self.assertEqual(out, preds1)
        self.assertEqual(infer_mock.call_count, 1)
        self.assertEqual(infer_mock.call_args.kwargs["max_side"], 1600)

    def test_stage2_infer_keeps_high_res_pass_when_model_crop_is_required(self):
        img = np.full((160, 380, 3), 255, dtype=np.uint8)
        preds1 = [
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 120, "y": 50, "width": 160, "height": 42},
            {"class": crop.MODEL2_SN_CLASS, "x": 135, "y": 120, "width": 190, "height": 44},
        ]
        preds2 = [
            {"class": crop.MODEL2_MODEL_CLASS, "x": 125, "y": 86, "width": 180, "height": 38},
        ]

        with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "1"}):
            with mock.patch.object(crop, "infer_with_resize", side_effect=[preds1, preds2]) as infer_mock:
                out = crop._stage2_infer_field_preds(img, "label.png")

        self.assertEqual(out, preds1 + preds2)
        self.assertEqual(infer_mock.call_count, 2)
        self.assertEqual(infer_mock.call_args_list[0].kwargs["max_side"], 1600)
        self.assertEqual(infer_mock.call_args_list[1].kwargs["max_side"], 2048)

    def test_stage2_part_no_detector_tries_wider_when_direct_barcode_is_cut(self):
        img = np.full((150, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087147",
            (20, 42),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 20, 58, 220, 96, step=4, bar_width=2)
        pred = {"x": 120, "y": 72, "width": 160, "height": 58}

        with mock.patch.object(crop, "_stage2_part_no_box_from_pred", return_value=(20, 24, 178, 108)):
            with mock.patch.object(
                crop,
                "box_from_pred_asym",
                side_effect=[(8, 18, 238, 116), (0, 10, 260, 126)],
            ):
                out = crop._stage2_crop_part_no_from_pred(img, pred)

        self.assertIsNotNone(out)
        self.assertTrue(crop.part_no_crop_has_complete_1d_barcode(out, min_span_ratio=0.18))
        self.assertGreaterEqual(out.shape[1], 220)

    def test_stage2_part_no_prefers_clean_direct_crop_over_wide_model_neighbor(self):
        img = np.full((220, 520, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 98012403",
            (36, 42),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 40, 56, 250, 92, step=4, bar_width=2)
        crop.cv2.putText(
            img,
            "Model: S110-5T",
            (36, 136),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 40, 150, 250, 188, step=4, bar_width=2)
        pred = {"x": 148, "y": 66, "width": 235, "height": 72}

        with mock.patch.object(crop, "_stage2_part_no_box_from_pred", return_value=(28, 28, 246, 104)):
            with mock.patch.object(
                crop,
                "box_from_pred_asym",
                side_effect=[(20, 18, 320, 170), (0, 10, 350, 200)],
            ):
                out = crop._stage2_crop_part_no_from_pred(img, pred)

        self.assertIsNotNone(out)
        self.assertLess(out.shape[0], 120)
        self.assertTrue(crop.part_no_crop_contains_1d_barcode(out, min_span_ratio=0.16))
        self.assertFalse(crop.part_no_crop_has_lower_neighbor_content(out))
        self.assertLess((out[-8:, :, 0] < 128).mean(), 0.01)

    def test_part_no_scan_miss_polish_does_not_pull_model_neighbor(self):
        img = np.full((180, 420, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087149",
            (24, 34),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 26, 46, 230, 92, step=4, bar_width=2)
        crop.cv2.putText(
            img,
            "Model: AP162E",
            (24, 124),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 26, 136, 230, 168, step=4, bar_width=2)
        pred = {"x": 130, "y": 64, "width": 220, "height": 76}

        raw, source, ok = crop._stage2_crop_part_no(img, pred)
        out = crop.polish_part_no_crop_for_scan_miss(raw)

        self.assertEqual(source, "detector")
        self.assertTrue(ok)
        self.assertIsNotNone(out)
        self.assertTrue(crop.part_no_crop_has_complete_1d_barcode(out, min_span_ratio=0.18))
        self.assertFalse(crop.part_no_crop_has_lower_neighbor_content(out))
        self.assertLess(out.shape[0], 130)
        self.assertLess((out[-8:, :, 0] < 128).mean(), 0.01)

    def test_trim_part_no_crop_removes_model_when_part_no_band_is_merged(self):
        img = np.full((240, 520, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 98012403",
            (24, 62),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.76,
            (0, 0, 0),
            3,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 24, 78, 455, 142, step=4, bar_width=2)
        crop.cv2.putText(
            img,
            "Model: S110-5T",
            (26, 202),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (0, 0, 0),
            3,
            crop.cv2.LINE_AA,
        )

        out = crop.trim_part_no_crop_before_lower_neighbor(img)

        self.assertLess(out.shape[0], 200)
        self.assertTrue(crop.part_no_crop_contains_1d_barcode(out, min_span_ratio=0.16))
        self.assertFalse(crop.part_no_crop_has_lower_neighbor_content(out))

    def test_trim_part_no_crop_removes_separate_model_barcode_neighbor(self):
        img = np.full((150, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087149",
            (18, 34),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 20, 46, 230, 68, step=4, bar_width=2)
        crop.cv2.putText(
            img,
            "Model: AP162E",
            (18, 104),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 22, 116, 210, 140, step=4, bar_width=2)

        out = crop.trim_part_no_crop_before_lower_neighbor(img)

        self.assertLess(out.shape[0], 100)
        self.assertTrue(crop.part_no_crop_contains_1d_barcode(out, min_span_ratio=0.16))
        self.assertFalse(crop.part_no_crop_has_lower_neighbor_content(out))

    def test_trim_part_no_crop_keeps_barcode_below_header_band(self):
        img = np.full((140, 540, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Part No.: 50087147",
            (24, 34),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.putText(
            img,
            "Rev:",
            (290, 34),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 372, 16, 455, 42, step=4, bar_width=2)
        draw_1d_barcode(img, 26, 58, 430, 108, step=4, bar_width=2)

        out = crop.trim_part_no_crop_before_lower_neighbor(img)

        self.assertGreater(out.shape[0], 108)
        self.assertTrue(crop.part_no_crop_contains_1d_barcode(out, min_span_ratio=0.16))

    def test_stage2_sn_crop_includes_lower_barcode(self):
        img = np.full((220, 520, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "S/N: 2150087147LDS4024590",
            (55, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 65, 88, 365, 128)
        pred = {"x": 215, "y": 70, "width": 320, "height": 42}

        out = crop._stage2_crop_sn(img, pred)

        self.assertIsNotNone(out)
        self.assertGreaterEqual(out.shape[0], 90)
        self.assertTrue(
            crop.crop_has_complete_1d_barcode(
                out,
                min_span_ratio=0.30,
                row_trans_threshold=0.08,
                active_threshold=0.18,
            )
        )

    def test_stage2_sn_crop_rejects_text_only_without_barcode(self):
        img = np.full((180, 460, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "S/N: 2150087147LDS4024590",
            (45, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        pred = {"x": 205, "y": 70, "width": 320, "height": 42}

        self.assertIsNone(crop._stage2_crop_sn(img, pred))

    def test_stage2_sn_crop_keeps_edge_barcode_with_extra_margin(self):
        img = np.full((220, 520, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "S/N: 2150087147LDS4024590",
            (55, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 0, 88, 300, 128)
        pred = {"x": 170, "y": 70, "width": 320, "height": 42}

        self.assertIsNotNone(crop.sn_barcode_box_near_pred(img, pred))
        out = crop._stage2_crop_sn(img, pred)
        self.assertIsNotNone(out)
        self.assertTrue(
            crop.crop_contains_1d_barcode(
                out,
                min_span_ratio=0.18,
                row_trans_threshold=0.08,
                active_threshold=0.18,
            )
        )

    def test_stage2_sn_crop_expands_when_barcode_extends_beyond_detector(self):
        img = np.full((220, 640, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "SN: 4E2640069549",
            (85, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 215, 88, 535, 128)
        pred = {"x": 250, "y": 70, "width": 230, "height": 42}

        out = crop._stage2_crop_sn(img, pred)

        self.assertIsNotNone(out)
        self.assertTrue(
            crop.crop_has_complete_1d_barcode(
                out,
                min_span_ratio=0.18,
                edge_guard_px=4,
                row_trans_threshold=0.08,
                active_threshold=0.18,
            )
        )
        barcode_box = crop.crop_1d_barcode_box(
            out,
            min_span_ratio=0.18,
            row_trans_threshold=0.08,
            active_threshold=0.18,
        )
        self.assertIsNotNone(barcode_box)
        self.assertLess(barcode_box[2], out.shape[1] - 4)

    def test_stage2_sn_crop_prefers_sn_barcode_over_lower_ean_neighbor(self):
        img = np.full((260, 620, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "S/N: 2150087147LDS4024590",
            (55, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 65, 88, 365, 128)
        crop.cv2.putText(
            img,
            "EAN:",
            (55, 156),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 115, 162, 520, 210, step=3, bar_width=1)
        pred = {"x": 215, "y": 92, "width": 330, "height": 96}

        out = crop._stage2_crop_sn(img, pred)

        self.assertIsNotNone(out)
        self.assertLess(out.shape[0], 531)
        self.assertTrue(
            crop.crop_has_complete_1d_barcode(
                out,
                min_span_ratio=0.18,
                row_trans_threshold=0.08,
                active_threshold=0.18,
            )
        )

    def test_stage2_sn_crop_limits_same_row_neighbor_barcode(self):
        img = np.full((220, 680, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "S/N: 2150087147LDS4024590",
            (55, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 65, 88, 365, 128)
        crop.cv2.putText(
            img,
            "MAC:",
            (420, 72),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 455, 88, 630, 128, step=3, bar_width=1)
        pred = {"x": 255, "y": 70, "width": 430, "height": 42}

        out = crop._stage2_crop_sn(img, pred)

        self.assertIsNotNone(out)
        self.assertLess(out.shape[1], 1500)
        self.assertTrue(
            crop.crop_has_complete_1d_barcode(
                out,
                min_span_ratio=0.18,
                row_trans_threshold=0.08,
                active_threshold=0.18,
            )
        )

    def test_stage2_can_save_sn_and_part_no_without_model_crop(self):
        img = np.full((120, 320, 3), 255, dtype=np.uint8)
        sn_crop = img[20:72, 30:250].copy()
        part_no_crop = img[5:55, 20:180].copy()

        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "input__label_1.png")
            crop.cv2.imwrite(label_path, img)
            crop.configure_paths(input_dir=root, out_dir=root)
            crop.ensure_dirs(clean=True)
            candidate = {
                "score": 100.0,
                "rotation": 0,
                "model_crop": None,
                "model_kind": "",
                "model_conf": None,
                "sn_crop": sn_crop,
                "sn_conf": 0.91,
                "part_no_crop": part_no_crop,
                "part_no_conf": 0.93,
                "part_no_kind": "detector",
                "part_no_ok": True,
                "part_no_codes": ["50087147"],
                "model_required": False,
            }
            with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
                with mock.patch.object(crop, "_stage2_infer_field_preds", return_value=[]):
                    with mock.patch.object(crop, "_stage2_build_candidate", return_value=candidate):
                        with mock.patch.object(crop, "_stage2_should_retry_rot180", return_value=False):
                            out = crop.stage2_crop_fields(label_path)

            self.assertIsNone(out["model_path"])
            self.assertTrue(os.path.isfile(out["sn_path"]))
            self.assertIsNotNone(out["part_no_path"])
            self.assertIsNotNone(out["part_no_path"])
            self.assertTrue(os.path.isfile(out["part_no_path"]))

    def test_stage2_saves_part_no_crop_when_barcode_decodes(self):
        img = np.full((120, 320, 3), 255, dtype=np.uint8)
        sn_crop = img[20:72, 30:250].copy()
        part_no_crop = img[5:55, 20:180].copy()

        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "input__label_1.png")
            crop.cv2.imwrite(label_path, img)
            crop.configure_paths(input_dir=root, out_dir=root)
            crop.ensure_dirs(clean=True)
            candidate = {
                "score": 100.0,
                "rotation": 0,
                "model_crop": None,
                "model_kind": "",
                "model_conf": None,
                "sn_crop": sn_crop,
                "sn_conf": 0.91,
                "part_no_crop": part_no_crop,
                "part_no_conf": 0.93,
                "part_no_kind": "detector",
                "part_no_ok": False,
                "part_no_codes": [],
                "model_required": False,
            }
            with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
                with mock.patch.object(crop, "_stage2_infer_field_preds", return_value=[]):
                    with mock.patch.object(crop, "_stage2_build_candidate", return_value=candidate):
                        with mock.patch.object(crop, "_stage2_should_retry_rot180", return_value=False):
                            with mock.patch.object(crop, "decode_raw_part_no_crop", return_value=["50087147"], create=True):
                                out = crop.stage2_crop_fields(label_path)

            self.assertIsNotNone(out["part_no_path"])
            self.assertTrue(os.path.isfile(out["part_no_path"]))
            self.assertEqual(out["part_no_codes"], ["50087147"])
            self.assertEqual(out["part_no"], "50087147")

    def test_decode_raw_part_no_crop_enables_repair_fallback(self):
        part_no_crop = np.full((40, 160, 3), 255, dtype=np.uint8)
        fake_scan2 = types.SimpleNamespace(
            read_part_no_barcodes=mock.Mock(return_value=["50087288"])
        )

        with mock.patch.dict(sys.modules, {"scan2": fake_scan2}):
            codes = crop.decode_raw_part_no_crop(part_no_crop, label_id="unit")

        self.assertEqual(codes, ["50087288"])
        args, kwargs = fake_scan2.read_part_no_barcodes.call_args
        self.assertTrue(args)
        self.assertTrue(kwargs["allow_pixel_repair"])

    def test_stage2_saves_part_no_crop_when_barcode_scan_misses(self):
        img = np.full((120, 320, 3), 255, dtype=np.uint8)
        sn_crop = img[20:72, 30:250].copy()
        part_no_crop = np.full((86, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            part_no_crop,
            "Part No.: 50087147",
            (18, 28),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(part_no_crop, 20, 38, 250, 80, step=4, bar_width=2)

        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "input__label_1.png")
            crop.cv2.imwrite(label_path, img)
            crop.configure_paths(input_dir=root, out_dir=root)
            crop.ensure_dirs(clean=True)
            candidate = {
                "score": 100.0,
                "rotation": 0,
                "model_crop": None,
                "model_kind": "",
                "model_conf": None,
                "sn_crop": sn_crop,
                "sn_conf": 0.91,
                "part_no_crop": part_no_crop,
                "part_no_conf": 0.93,
                "part_no_kind": "detector",
                "part_no_ok": False,
                "part_no_codes": [],
                "model_required": False,
            }
            with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
                with mock.patch.object(crop, "_stage2_infer_field_preds", return_value=[]):
                    with mock.patch.object(crop, "_stage2_build_candidate", return_value=candidate):
                        with mock.patch.object(crop, "_stage2_should_retry_rot180", return_value=False):
                            with mock.patch.object(crop, "decode_raw_part_no_crop", return_value=[], create=True):
                                out = crop.stage2_crop_fields(label_path)

            self.assertIsNotNone(out["part_no_path"])
            self.assertTrue(os.path.isfile(out["part_no_path"]))
            self.assertEqual(out["part_no_codes"], [])
            self.assertEqual(out["part_no"], "")
            self.assertTrue(os.path.isfile(out["sn_path"]))

    def test_stage2_does_not_save_invalid_heuristic_part_no_scan_miss(self):
        img = np.full((120, 320, 3), 255, dtype=np.uint8)
        sn_crop = img[20:72, 30:250].copy()
        part_no_crop = np.full((64, 320, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            part_no_crop,
            "AP162E(11ax indoor,2+2 dual bands)",
            (6, 34),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )

        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "input__label_1.png")
            crop.cv2.imwrite(label_path, img)
            crop.configure_paths(input_dir=root, out_dir=root)
            crop.ensure_dirs(clean=True)
            candidate = {
                "score": 100.0,
                "rotation": 0,
                "model_crop": None,
                "model_kind": "",
                "model_conf": None,
                "sn_crop": sn_crop,
                "sn_conf": 0.91,
                "part_no_crop": part_no_crop,
                "part_no_conf": 0.21,
                "part_no_kind": "heuristic",
                "part_no_ok": False,
                "part_no_codes": [],
                "model_required": False,
            }
            with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
                with mock.patch.object(crop, "_stage2_infer_field_preds", return_value=[]):
                    with mock.patch.object(crop, "_stage2_build_candidate", return_value=candidate):
                        with mock.patch.object(crop, "_stage2_should_retry_rot180", return_value=False):
                            with mock.patch.object(crop, "decode_raw_part_no_crop", return_value=[], create=True):
                                out = crop.stage2_crop_fields(label_path)

            self.assertIsNone(out["part_no_path"])
            self.assertEqual(out["part_no_codes"], [])
            self.assertEqual(out["part_no"], "")
            self.assertTrue(os.path.isfile(out["sn_path"]))

    def test_stage2_drops_candidate_without_any_field_crop(self):
        img = np.full((120, 320, 3), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "input__label_1.png")
            crop.cv2.imwrite(label_path, img)
            crop.configure_paths(input_dir=root, out_dir=root)
            crop.ensure_dirs(clean=True)
            candidate = {
                "score": 100.0,
                "rotation": 0,
                "model_crop": None,
                "model_kind": "",
                "model_conf": None,
                "sn_crop": None,
                "sn_conf": None,
                "part_no_crop": None,
                "part_no_conf": None,
                "part_no_kind": "",
                "part_no_ok": False,
                "part_no_codes": [],
                "model_required": False,
            }
            with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
                with mock.patch.object(crop, "_stage2_infer_field_preds", return_value=[]):
                    with mock.patch.object(crop, "_stage2_build_candidate", return_value=candidate):
                        with mock.patch.object(crop, "_stage2_should_retry_rot180", return_value=False):
                            out = crop.stage2_crop_fields(label_path)

            self.assertIsNone(out)

    def test_stage2_drops_part_no_scan_miss_without_sn_or_model_crop(self):
        img = np.full((120, 320, 3), 255, dtype=np.uint8)
        part_no_crop = img[5:55, 20:180].copy()

        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "input__label_1.png")
            crop.cv2.imwrite(label_path, img)
            crop.configure_paths(input_dir=root, out_dir=root)
            crop.ensure_dirs(clean=True)
            candidate = {
                "score": 100.0,
                "rotation": 0,
                "model_crop": None,
                "model_kind": "",
                "model_conf": None,
                "sn_crop": None,
                "sn_conf": None,
                "part_no_crop": part_no_crop,
                "part_no_conf": 0.42,
                "part_no_kind": "detector",
                "part_no_ok": False,
                "part_no_codes": [],
                "model_required": False,
            }
            with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
                with mock.patch.object(crop, "_stage2_infer_field_preds", return_value=[]):
                    with mock.patch.object(crop, "_stage2_build_candidate", return_value=candidate):
                        with mock.patch.object(crop, "_stage2_should_retry_rot180", return_value=False):
                            with mock.patch.object(crop, "decode_raw_part_no_crop", return_value=[]):
                                out = crop.stage2_crop_fields(label_path)

            self.assertIsNone(out)
            self.assertFalse(os.path.exists(os.path.join(crop.OUT_PART_NO_DIR, "input__label_1__part_no.png")))

    def test_stage2_recovers_part_no_from_original_context_when_label_top_is_cut(self):
        original = np.full((320, 560, 3), 255, dtype=np.uint8)
        crop.cv2.rectangle(original, (30, 48), (520, 285), (0, 0, 0), 2)
        crop.cv2.putText(
            original,
            "Part No.: 50087149",
            (54, 86),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(original, 58, 100, 265, 135, step=4, bar_width=2)
        crop.cv2.putText(
            original,
            "QTY: 1 PCS",
            (370, 86),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(original, 390, 100, 485, 135, step=4, bar_width=2)
        crop.cv2.putText(
            original,
            "Model: AP162E",
            (54, 176),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(original, 58, 188, 270, 226, step=4, bar_width=2)
        label_crop = original[145:285, 30:520].copy()

        def fake_part_no_decode(crop_img, label_id=""):
            top_dark = (crop_img[:42, :, 0] < 128).mean()
            if top_dark > 0.015 and crop_img.shape[1] < 360:
                return ["50087149"]
            return []

        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "input")
            out_dir = os.path.join(root, "out")
            os.makedirs(input_dir)
            crop.configure_paths(input_dir=input_dir, out_dir=out_dir)
            crop.ensure_dirs(clean=True)
            original_path = os.path.join(input_dir, "photo.jpg")
            label_path = os.path.join(crop.STAGE1_DIR, "photo.jpg__label_1.png")
            crop.cv2.imwrite(original_path, original)
            crop.cv2.imwrite(label_path, label_crop)
            sn_crop = label_crop[20:72, 30:250].copy()
            candidate = {
                "score": 100.0,
                "rotation": 0,
                "model_crop": None,
                "model_kind": "",
                "model_conf": None,
                "sn_crop": sn_crop,
                "sn_conf": 0.91,
                "part_no_crop": None,
                "part_no_conf": None,
                "part_no_kind": "",
                "part_no_ok": False,
                "part_no_codes": [],
                "model_required": False,
            }
            with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
                with mock.patch.object(crop, "_stage2_infer_field_preds", return_value=[]):
                    with mock.patch.object(crop, "_stage2_build_candidate", return_value=candidate):
                        with mock.patch.object(crop, "_stage2_should_retry_rot180", return_value=False):
                            with mock.patch.object(crop, "decode_raw_part_no_crop", side_effect=fake_part_no_decode):
                                out = crop.stage2_crop_fields(label_path)

            self.assertEqual(out["part_no_codes"], ["50087149"])
            self.assertEqual(out["part_no"], "50087149")
            self.assertEqual(out["part_no_crop_source"], "original_context")
            self.assertTrue(os.path.isfile(out["part_no_path"]))
            saved = crop.read_image(out["part_no_path"])
            self.assertIsNotNone(saved)
            self.assertLess(saved.shape[1], 360)
            self.assertGreater((saved[:42, :, 0] < 128).mean(), 0.015)

    def test_stage2_rejects_part_no_crop_when_scan_finds_model_code(self):
        img = np.full((120, 320, 3), 255, dtype=np.uint8)
        sn_crop = img[20:72, 30:250].copy()
        part_no_crop = img[5:55, 20:180].copy()

        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "input__label_1.png")
            crop.cv2.imwrite(label_path, img)
            crop.configure_paths(input_dir=root, out_dir=root)
            crop.ensure_dirs(clean=True)
            candidate = {
                "score": 100.0,
                "rotation": 0,
                "model_crop": None,
                "model_kind": "",
                "model_conf": None,
                "sn_crop": sn_crop,
                "sn_conf": 0.91,
                "part_no_crop": part_no_crop,
                "part_no_conf": 0.27,
                "part_no_kind": "detector",
                "part_no_ok": False,
                "part_no_codes": [],
                "model_required": False,
            }
            with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_SAVE_MODEL": "0"}):
                with mock.patch.object(crop, "_stage2_infer_field_preds", return_value=[]):
                    with mock.patch.object(crop, "_stage2_build_candidate", return_value=candidate):
                        with mock.patch.object(crop, "_stage2_should_retry_rot180", return_value=False):
                            with mock.patch.object(crop, "decode_raw_part_no_crop", return_value=["AP162E"], create=True):
                                out = crop.stage2_crop_fields(label_path)

            self.assertIsNone(out["part_no_path"])
            self.assertEqual(out["part_no_codes"], [])
            self.assertEqual(out["part_no"], "")
            self.assertTrue(os.path.isfile(out["sn_path"]))

    def test_stage2_retries_rot180_when_part_no_is_incomplete(self):
        candidate = {
            "model_required": False,
            "part_no_ok": False,
            "sn_crop": np.full((32, 140, 3), 255, dtype=np.uint8),
            "sn_conf": 0.95,
        }

        with mock.patch.dict(crop.os.environ, {"CROP_STAGE2_ROTATION_RETRY": "1"}):
            self.assertTrue(crop._stage2_should_retry_rot180(candidate))

    def test_stage1_filter_keeps_product_label_with_corner_mark(self):
        img = np.full((260, 420, 3), 255, dtype=np.uint8)
        img[25:58, 338:385] = 0
        draw_product_field_structure(img)

        self.assertTrue(crop.stage1_is_product_label_crop(img))

    def test_stage1_filter_rejects_corner_only_crop_without_field_structure(self):
        img = np.full((260, 420, 3), 255, dtype=np.uint8)
        img[25:58, 338:385] = 0

        self.assertFalse(crop.stage1_is_product_label_crop(img))

    def test_stage1_filter_rejects_red_shipping_or_brand_crop(self):
        img = np.full((260, 420, 3), 255, dtype=np.uint8)
        img[25:58, 338:385] = 0
        draw_product_field_structure(img)
        img[40:150, 20:135] = (0, 0, 220)

        self.assertFalse(crop.stage1_is_product_label_crop(img))

    def test_stage1_filter_rejects_dark_edge_false_positive(self):
        img = np.full((260, 500, 3), 255, dtype=np.uint8)
        img[:120, :] = 55
        img[:, 455:] = 35
        img[150:195, 360:420] = 0

        self.assertFalse(crop.stage1_is_product_label_crop(img))

    def test_stage1_filter_rejects_half_width_product_label(self):
        img = np.full((580, 721, 3), 255, dtype=np.uint8)
        img[30:95, 560:650] = 0

        self.assertFalse(crop.stage1_is_product_label_crop(img))

    def test_stage1_collect_rejects_blurry_small_background_label(self):
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        large_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        small_crop = np.full((120, 180, 3), 255, dtype=np.uint8)
        large_box = (220, 140, 700, 400)
        small_box = (18, 180, 198, 300)
        large_pred = {
            "x": 460,
            "y": 270,
            "width": 480,
            "height": 260,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }
        small_pred = {
            "x": 108,
            "y": 240,
            "width": 180,
            "height": 120,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.31,
        }

        def fake_box_from_pred(_img, pred, *_args, **_kwargs):
            return large_box if pred is large_pred else small_box

        def fake_crop_from_pred(_img, pred, *_args, **_kwargs):
            return large_crop if pred is large_pred else small_crop

        with mock.patch.object(crop, "box_from_pred", side_effect=fake_box_from_pred):
            with mock.patch.object(crop, "crop_from_pred", side_effect=fake_crop_from_pred):
                with mock.patch.object(
                    crop,
                    "_stage1_prepare_product_label_crop",
                    side_effect=lambda candidate, **_kwargs: candidate,
                ):
                    with mock.patch.object(
                        crop,
                        "_stage1_maybe_retry_upward_crop",
                        side_effect=lambda _img, _pred, box, candidate, **_kwargs: (box, candidate),
                    ):
                        with mock.patch.object(
                            crop,
                            "_stage1_maybe_retry_edge_recovery_crop",
                            side_effect=lambda _img, _pred, box, candidate, **_kwargs: (box, candidate),
                        ):
                            with mock.patch.object(crop, "_stage1_has_single_label_field_evidence", return_value=True):
                                with mock.patch.object(
                                    crop,
                                    "_stage1_focus_score",
                                    side_effect=lambda candidate: 1800.0 if candidate is large_crop else 120.0,
                                ):
                                    out, dropped = crop._stage1_collect_product_label_entries(
                                        img,
                                        [small_pred, large_pred],
                                        require_single_label_field_evidence=True,
                                    )

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(out[0]["box"], large_box)

    def test_stage1_collect_keeps_small_blurry_high_conf_label(self):
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        small_crop = np.full((120, 180, 3), 255, dtype=np.uint8)
        small_box = (420, 720, 600, 840)
        pred = {
            "x": 510,
            "y": 780,
            "width": 180,
            "height": 120,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.94,
        }

        with mock.patch.object(crop, "box_from_pred", return_value=small_box):
            with mock.patch.object(crop, "crop_from_pred", return_value=small_crop):
                with mock.patch.object(
                    crop,
                    "_stage1_prepare_product_label_crop",
                    side_effect=lambda candidate, **_kwargs: candidate,
                ):
                    with mock.patch.object(
                        crop,
                        "_stage1_maybe_retry_upward_crop",
                        side_effect=lambda _img, _pred, box, candidate, **_kwargs: (box, candidate),
                    ):
                        with mock.patch.object(
                            crop,
                            "_stage1_maybe_retry_edge_recovery_crop",
                            side_effect=lambda _img, _pred, box, candidate, **_kwargs: (box, candidate),
                        ):
                            with mock.patch.object(crop, "_stage1_has_single_label_field_evidence", return_value=True):
                                with mock.patch.object(crop, "_stage1_focus_score", return_value=120.0):
                                    out, dropped = crop._stage1_collect_product_label_entries(
                                        img,
                                        [pred],
                                        require_single_label_field_evidence=True,
                                    )

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)
        self.assertEqual(out[0]["box"], small_box)

    def test_stage1_collect_keeps_small_sharp_label(self):
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        small_crop = np.full((120, 180, 3), 255, dtype=np.uint8)
        small_box = (420, 720, 600, 840)
        pred = {
            "x": 510,
            "y": 780,
            "width": 180,
            "height": 120,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.94,
        }

        with mock.patch.object(crop, "box_from_pred", return_value=small_box):
            with mock.patch.object(crop, "crop_from_pred", return_value=small_crop):
                with mock.patch.object(
                    crop,
                    "_stage1_prepare_product_label_crop",
                    side_effect=lambda candidate, **_kwargs: candidate,
                ):
                    with mock.patch.object(
                        crop,
                        "_stage1_maybe_retry_upward_crop",
                        side_effect=lambda _img, _pred, box, candidate, **_kwargs: (box, candidate),
                    ):
                        with mock.patch.object(
                            crop,
                            "_stage1_maybe_retry_edge_recovery_crop",
                            side_effect=lambda _img, _pred, box, candidate, **_kwargs: (box, candidate),
                        ):
                            with mock.patch.object(crop, "_stage1_has_single_label_field_evidence", return_value=True):
                                with mock.patch.object(crop, "_stage1_focus_score", return_value=1800.0):
                                    out, dropped = crop._stage1_collect_product_label_entries(
                                        img,
                                        [pred],
                                        require_single_label_field_evidence=True,
                                    )

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)
        self.assertEqual(out[0]["box"], small_box)

    def test_stage1_filter_accepts_partial_product_label_with_field_structure(self):
        img = np.full((340, 290, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(img)
        draw_product_field_structure(img)
        crop.cv2.putText(
            img,
            "Model: AP162E",
            (22, 128),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )

        self.assertLess(img.shape[1] / float(img.shape[0]), crop.STAGE1_MIN_ASPECT)
        self.assertTrue(crop.stage1_is_partial_product_label_crop(img))
        self.assertTrue(crop.stage1_is_product_label_crop(img))

    def test_stage1_filter_accepts_small_red_overrun_when_field_structure_is_strong(self):
        img = np.full((260, 420, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(img)
        draw_product_field_structure(img)
        img[34:76, 18:88] = (0, 0, 215)

        red_ratio = crop.stage1_red_pixel_ratio(img)
        self.assertGreater(red_ratio, crop.STAGE1_MAX_RED_RATIO)
        self.assertLess(red_ratio, crop.STAGE1_PARTIAL_MAX_RED_RATIO)
        self.assertTrue(crop.stage1_is_partial_product_label_crop(img))
        self.assertTrue(crop.stage1_is_product_label_crop(img))

    def test_stage1_filter_accepts_moderate_red_overrun_when_corner_and_field_are_clear(self):
        img = np.full((260, 420, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(img)
        draw_product_field_structure(img)
        img[34:76, 18:104] = (0, 0, 215)

        red_ratio = crop.stage1_red_pixel_ratio(img)
        self.assertGreater(red_ratio, 0.03)
        self.assertLess(red_ratio, crop.STAGE1_PARTIAL_MAX_RED_RATIO)
        self.assertTrue(crop.stage1_is_partial_product_label_crop(img))
        self.assertTrue(crop.stage1_is_product_label_crop(img))

    def test_stage1_edge_crop_rejects_false_single_label_layout(self):
        img = np.full((900, 1200, 3), 255, dtype=np.uint8)
        crop_img = np.full((635, 1497, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(crop_img)
        draw_product_field_structure(crop_img)

        pred = {
            "x": 540,
            "y": 140,
            "width": 860,
            "height": 420,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.97,
        }
        field_preds = [
            {
                "class": crop.MODEL2_PART_NO_CLASS,
                "x": int(crop_img.shape[1] * 0.564),
                "y": int(crop_img.shape[0] * 0.034),
                "width": int(crop_img.shape[1] * 0.117),
                "height": int(crop_img.shape[0] * 0.068),
            },
            {
                "class": crop.MODEL2_SN_CLASS,
                "x": int(crop_img.shape[1] * 0.043),
                "y": int(crop_img.shape[0] * 0.044),
                "width": int(crop_img.shape[1] * 0.085),
                "height": int(crop_img.shape[0] * 0.088),
            },
            {
                "class": crop.MODEL2_SN_CLASS,
                "x": int(crop_img.shape[1] * 0.849),
                "y": int(crop_img.shape[0] * 0.776),
                "width": int(crop_img.shape[1] * 0.303),
                "height": int(crop_img.shape[0] * 0.315),
            },
        ]

        with mock.patch.object(crop, "box_from_pred", return_value=(0, 0, 860, 420)):
            with mock.patch.object(crop, "crop_from_pred", return_value=crop_img):
                with mock.patch.object(crop, "_stage1_prepare_product_label_crop", return_value=crop_img):
                    with mock.patch.object(crop, "infer_with_resize", return_value=field_preds):
                        out, dropped = crop._stage1_collect_product_label_entries(img, [pred])

        self.assertEqual(len(out), 0)
        self.assertEqual(dropped, 1)

    def test_stage1_edge_crop_accepts_plausible_single_label_layout(self):
        img = np.full((900, 1200, 3), 255, dtype=np.uint8)
        crop_img = np.full((420, 760, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(crop_img)
        draw_product_field_structure(crop_img)

        pred = {
            "x": 600,
            "y": 180,
            "width": 900,
            "height": 420,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.97,
        }
        field_preds = [
            {
                "class": crop.MODEL2_MODEL_CLASS,
                "x": int(crop_img.shape[1] * 0.12),
                "y": int(crop_img.shape[0] * 0.30),
                "width": int(crop_img.shape[1] * 0.19),
                "height": int(crop_img.shape[0] * 0.12),
            },
            {
                "class": crop.MODEL2_PART_NO_CLASS,
                "x": int(crop_img.shape[1] * 0.14),
                "y": int(crop_img.shape[0] * 0.17),
                "width": int(crop_img.shape[1] * 0.22),
                "height": int(crop_img.shape[0] * 0.12),
            },
            {
                "class": crop.MODEL2_SN_CLASS,
                "x": int(crop_img.shape[1] * 0.25),
                "y": int(crop_img.shape[0] * 0.60),
                "width": int(crop_img.shape[1] * 0.42),
                "height": int(crop_img.shape[0] * 0.14),
            },
        ]

        with mock.patch.object(crop, "box_from_pred", return_value=(0, 0, 900, 420)):
            with mock.patch.object(crop, "crop_from_pred", return_value=crop_img):
                with mock.patch.object(crop, "_stage1_prepare_product_label_crop", return_value=crop_img):
                    with mock.patch.object(crop, "infer_with_resize", return_value=field_preds):
                        out, dropped = crop._stage1_collect_product_label_entries(img, [pred])

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)

    def test_stage1_relaxed_single_field_evidence_accepts_top_clipped_layout(self):
        crop_img = np.full((300, 480, 3), 255, dtype=np.uint8)
        preds = [
            {
                "class": crop.MODEL2_PART_NO_CLASS,
                "x": int(crop_img.shape[1] * 0.16),
                "y": int(crop_img.shape[0] * 0.035),
                "width": int(crop_img.shape[1] * 0.23),
                "height": int(crop_img.shape[0] * 0.11),
            },
            {
                "class": crop.MODEL2_SN_CLASS,
                "x": int(crop_img.shape[1] * 0.29),
                "y": int(crop_img.shape[0] * 0.58),
                "width": int(crop_img.shape[1] * 0.44),
                "height": int(crop_img.shape[0] * 0.14),
            },
        ]

        with mock.patch.object(crop, "infer_with_resize", return_value=preds):
            self.assertTrue(crop._stage1_has_relaxed_single_label_field_evidence(crop_img))

    def test_stage1_relaxed_single_field_evidence_rejects_false_edge_layout(self):
        crop_img = np.full((635, 1497, 3), 255, dtype=np.uint8)
        preds = [
            {
                "class": crop.MODEL2_PART_NO_CLASS,
                "x": int(crop_img.shape[1] * 0.564),
                "y": int(crop_img.shape[0] * 0.034),
                "width": int(crop_img.shape[1] * 0.117),
                "height": int(crop_img.shape[0] * 0.068),
            },
            {
                "class": crop.MODEL2_SN_CLASS,
                "x": int(crop_img.shape[1] * 0.043),
                "y": int(crop_img.shape[0] * 0.044),
                "width": int(crop_img.shape[1] * 0.085),
                "height": int(crop_img.shape[0] * 0.088),
            },
            {
                "class": crop.MODEL2_SN_CLASS,
                "x": int(crop_img.shape[1] * 0.849),
                "y": int(crop_img.shape[0] * 0.776),
                "width": int(crop_img.shape[1] * 0.303),
                "height": int(crop_img.shape[0] * 0.315),
            },
        ]

        with mock.patch.object(crop, "infer_with_resize", return_value=preds):
            self.assertFalse(crop._stage1_has_relaxed_single_label_field_evidence(crop_img))

    def test_stage1_filter_rejects_vertical_and_superwide_crops(self):
        vertical = np.full((420, 120, 3), 255, dtype=np.uint8)
        vertical[20:55, 80:115] = 0
        superwide = np.full((120, 480, 3), 255, dtype=np.uint8)
        superwide[20:55, 360:410] = 0

        self.assertFalse(crop.stage1_is_product_label_crop(vertical))
        self.assertFalse(crop.stage1_is_product_label_crop(superwide))

    def test_stage1_collect_uses_validated_low_confidence_fallback(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        product_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(product_crop)
        draw_product_field_structure(product_crop)
        field_preds = [
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 110, "y": 45, "width": 160, "height": 40},
            {"class": crop.MODEL2_SN_CLASS, "x": 300, "y": 84, "width": 190, "height": 44},
        ]
        pred = {
            "x": 500,
            "y": 400,
            "width": 520,
            "height": 320,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": crop.MIN_CONF_1 - 0.10,
        }

        with mock.patch.object(crop, "crop_from_pred", return_value=product_crop):
            with mock.patch.object(crop, "infer_with_resize", return_value=field_preds):
                out, dropped = crop._stage1_collect_label_crops(img, [pred])

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))

    def test_stage1_collect_rejects_low_confidence_candidate_without_field_evidence(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        product_like_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(product_like_crop)
        draw_product_field_structure(product_like_crop)
        pred = {
            "x": 500,
            "y": 400,
            "width": 520,
            "height": 320,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": crop.MIN_CONF_1 - 0.10,
        }

        with mock.patch.object(crop, "crop_from_pred", return_value=product_like_crop):
            with mock.patch.object(crop, "infer_with_resize", return_value=[]):
                out, _dropped = crop._stage1_collect_label_crops(img, [pred])

        self.assertEqual(out, [])

    def test_stage1_collect_rotates_vertical_fallback_candidate(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        product_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(product_crop)
        draw_product_field_structure(product_crop)
        vertical_crop = crop.rotate_image(product_crop, 90)
        field_preds = [
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 110, "y": 45, "width": 160, "height": 40},
            {"class": crop.MODEL2_SN_CLASS, "x": 300, "y": 84, "width": 190, "height": 44},
        ]
        pred = {
            "x": 500,
            "y": 400,
            "width": 320,
            "height": 520,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.95,
        }

        self.assertFalse(crop.stage1_is_product_label_crop(vertical_crop))
        with mock.patch.object(crop, "crop_from_pred", return_value=vertical_crop):
            with mock.patch.object(crop, "infer_with_resize", return_value=field_preds):
                out, dropped = crop._stage1_collect_label_crops(img, [pred])

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)
        self.assertGreater(out[0].shape[1] / float(out[0].shape[0]), crop.STAGE1_MIN_ASPECT)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))

    def test_stage1_collect_rejects_invalid_low_confidence_fallback(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        invalid_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        pred = {
            "x": 500,
            "y": 400,
            "width": 520,
            "height": 320,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": crop.MIN_CONF_1 - 0.10,
        }

        with mock.patch.object(crop, "crop_from_pred", return_value=invalid_crop):
            out, _dropped = crop._stage1_collect_label_crops(img, [pred])

        self.assertEqual(out, [])

    def test_stage1_prepare_keeps_crop_that_only_becomes_valid_after_tighten(self):
        raw = np.full((260, 420, 3), 255, dtype=np.uint8)
        tightened = np.full((220, 380, 3), 255, dtype=np.uint8)

        with mock.patch.object(crop, "stage1_tighten_label_crop", return_value=tightened):
            with mock.patch.object(
                crop,
                "stage1_is_product_label_crop",
                side_effect=lambda img: img is tightened,
            ):
                with mock.patch.object(
                    crop,
                    "stage1_normalize_label_orientation",
                    side_effect=lambda img: img,
                ):
                    out = crop._stage1_prepare_product_label_crop(raw)

        self.assertIs(out, tightened)

    def test_stage1_collect_supplements_existing_labels_with_validated_partial_low_confidence_candidate(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        full_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(full_crop)
        draw_product_field_structure(full_crop)

        partial_crop = np.full((320, 250, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(partial_crop)
        draw_product_field_structure(partial_crop)

        high_pred = {
            "x": 280,
            "y": 350,
            "width": 420,
            "height": 240,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }
        low_pred = {
            "x": 760,
            "y": 360,
            "width": 260,
            "height": 280,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.18,
        }
        field_preds = [  # noqa: F841  (documents the fixture; pre-existing)
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 85, "y": 45, "width": 150, "height": 36},
            {"class": crop.MODEL2_SN_CLASS, "x": 180, "y": 90, "width": 170, "height": 42},
        ]

        def fake_crop_from_pred(_img, pred, *_args, **_kwargs):
            return full_crop if pred["confidence"] > 0.5 else partial_crop

        with mock.patch.object(crop, "crop_from_pred", side_effect=fake_crop_from_pred):
            with mock.patch.object(
                crop,
                "_stage1_has_single_label_field_evidence",
                side_effect=lambda img: img is full_crop or img is partial_crop,
            ):
                out, dropped = crop._stage1_collect_label_crops(img, [high_pred, low_pred])

        self.assertEqual(len(out), 2)
        self.assertEqual(dropped, 0)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))
        self.assertTrue(crop.stage1_is_product_label_crop(out[1]))

    def test_stage1_collect_supplement_keeps_lower_confidence_real_candidate_after_invalid_overlap(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        full_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(full_crop)
        draw_product_field_structure(full_crop)

        invalid_crop = np.full((320, 250, 3), 255, dtype=np.uint8)

        partial_crop = np.full((320, 250, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(partial_crop)
        draw_product_field_structure(partial_crop)

        high_pred = {
            "x": 240,
            "y": 350,
            "width": 420,
            "height": 240,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }
        invalid_overlap_pred = {
            "x": 700,
            "y": 360,
            "width": 280,
            "height": 300,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.24,
        }
        valid_overlap_pred = {
            "x": 708,
            "y": 364,
            "width": 270,
            "height": 294,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.17,
        }
        single_label_field_preds = [
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 85, "y": 45, "width": 150, "height": 36},
            {"class": crop.MODEL2_SN_CLASS, "x": 180, "y": 90, "width": 170, "height": 42},
        ]

        def fake_crop_from_pred(_img, pred, *_args, **_kwargs):
            if pred["confidence"] > 0.5:
                return full_crop
            if pred["confidence"] > 0.2:
                return invalid_crop
            return partial_crop

        def fake_infer_with_resize(crop_img, *_args, **_kwargs):
            if crop_img is partial_crop:
                return single_label_field_preds
            return []

        with mock.patch.object(crop, "crop_from_pred", side_effect=fake_crop_from_pred):
            with mock.patch.object(
                crop,
                "_stage1_has_single_label_field_evidence",
                side_effect=lambda img: img is full_crop or img is partial_crop,
            ):
                out, dropped = crop._stage1_collect_label_crops(
                    img,
                    [high_pred, invalid_overlap_pred, valid_overlap_pred],
                )

        self.assertEqual(len(out), 2)
        self.assertEqual(dropped, 1)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))
        self.assertTrue(crop.stage1_is_product_label_crop(out[1]))

    def test_stage1_collect_supplement_rejects_larger_near_contained_single_label_box(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        full_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(full_crop)
        draw_product_field_structure(full_crop)

        merged_crop = np.full((430, 700, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(merged_crop)
        draw_product_field_structure(merged_crop)

        high_pred = {
            "x": 320,
            "y": 420,
            "width": 420,
            "height": 240,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }
        low_pred = {
            "x": 420,
            "y": 350,
            "width": 640,
            "height": 420,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.19,
        }
        single_label_field_preds = [
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 120, "y": 45, "width": 180, "height": 36},
            {"class": crop.MODEL2_SN_CLASS, "x": 260, "y": 90, "width": 190, "height": 42},
        ]

        def fake_crop_from_pred(_img, pred, *_args, **_kwargs):
            if pred["confidence"] > 0.5:
                return full_crop
            return merged_crop

        def fake_box_from_pred(_img, pred, *_args, **_kwargs):
            if pred["confidence"] > 0.5:
                return (140, 300, 560, 540)
            return (170, 120, 810, 540)

        def fake_infer_with_resize(image_or_path, *_args, model_id=None, **_kwargs):
            if model_id == crop.MODEL2_ID and image_or_path is merged_crop:
                return single_label_field_preds
            return []

        with mock.patch.object(crop, "crop_from_pred", side_effect=fake_crop_from_pred):
            with mock.patch.object(crop, "box_from_pred", side_effect=fake_box_from_pred):
                with mock.patch.object(crop, "infer_with_resize", side_effect=fake_infer_with_resize):
                    with mock.patch.object(crop, "_stage1_hardcase_model_available", return_value=False):
                        out, dropped = crop._stage1_collect_label_crops(img, [high_pred, low_pred])

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))

    def test_stage1_collect_supplement_rejects_multi_label_candidate(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        full_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(full_crop)
        draw_product_field_structure(full_crop)

        multi_label_crop = np.full((360, 420, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(multi_label_crop)
        draw_product_field_structure(multi_label_crop)

        high_pred = {
            "x": 280,
            "y": 350,
            "width": 420,
            "height": 240,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }
        low_pred = {
            "x": 760,
            "y": 360,
            "width": 340,
            "height": 320,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.18,
        }
        multi_label_field_preds = [
            {"class": crop.MODEL2_MODEL_CLASS, "x": 80, "y": 40, "width": 120, "height": 30},
            {"class": crop.MODEL2_MODEL_CLASS, "x": 240, "y": 40, "width": 120, "height": 30},
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 70, "y": 70, "width": 140, "height": 32},
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 230, "y": 70, "width": 140, "height": 32},
            {"class": crop.MODEL2_SN_CLASS, "x": 70, "y": 120, "width": 150, "height": 34},
            {"class": crop.MODEL2_SN_CLASS, "x": 230, "y": 120, "width": 150, "height": 34},
        ]

        def fake_crop_from_pred(_img, pred, *_args, **_kwargs):
            return full_crop if pred["confidence"] > 0.5 else multi_label_crop

        def fake_infer_with_resize(crop_img, *_args, **_kwargs):
            if crop_img is multi_label_crop:
                return multi_label_field_preds
            return []

        with mock.patch.object(crop, "crop_from_pred", side_effect=fake_crop_from_pred):
            with mock.patch.object(crop, "infer_with_resize", side_effect=fake_infer_with_resize):
                out, dropped = crop._stage1_collect_label_crops(img, [high_pred, low_pred])

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 1)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))

    def test_stage1_collect_supplement_rejects_merged_box_covering_two_existing_labels(self):
        img = np.full((800, 1200, 3), 255, dtype=np.uint8)
        full_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(full_crop)
        draw_product_field_structure(full_crop)

        merged_crop = np.full((320, 820, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(merged_crop)
        draw_product_field_structure(merged_crop)

        high_pred_left = {
            "x": 260,
            "y": 340,
            "width": 420,
            "height": 240,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.99,
        }
        high_pred_right = {
            "x": 760,
            "y": 340,
            "width": 420,
            "height": 240,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }
        low_pred_merged = {
            "x": 520,
            "y": 320,
            "width": 940,
            "height": 300,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.18,
        }
        single_label_field_preds = [
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 120, "y": 45, "width": 180, "height": 36},
            {"class": crop.MODEL2_SN_CLASS, "x": 260, "y": 90, "width": 190, "height": 42},
        ]

        def fake_crop_from_pred(_img, pred, *_args, **_kwargs):
            if pred["confidence"] >= 0.98:
                return full_crop
            return merged_crop

        def fake_infer_with_resize(image_or_path, *_args, model_id=None, **_kwargs):
            if model_id == crop.MODEL2_ID and image_or_path is merged_crop:
                return single_label_field_preds
            return []

        with mock.patch.object(crop, "crop_from_pred", side_effect=fake_crop_from_pred):
            with mock.patch.object(crop, "_stage1_hardcase_model_available", return_value=False):
                with mock.patch.object(crop, "infer_with_resize", side_effect=fake_infer_with_resize):
                    out, dropped = crop._stage1_collect_label_crops(
                        img,
                        [high_pred_left, high_pred_right, low_pred_merged],
                    )

        self.assertEqual(len(out), 2)
        self.assertEqual(dropped, 0)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))
        self.assertTrue(crop.stage1_is_product_label_crop(out[1]))

    def test_stage1_box_conflicts_rejects_large_near_contained_overlap(self):
        existing_box = (274, 731, 2330, 1856)
        oversized_partial_box = (834, 0, 2908, 1740)
        self.assertTrue(crop._stage1_box_conflicts_with_existing(oversized_partial_box, [existing_box]))

    def test_stage1_box_conflicts_rejects_medium_overlap_large_oversized_box(self):
        existing_box = (496, 1004, 2254, 1774)
        oversized_partial_box = (156, 0, 2310, 1403)
        self.assertTrue(crop._stage1_box_conflicts_with_existing(oversized_partial_box, [existing_box]))

    def test_stage1_collect_rejects_tiny_main_box_even_if_crop_looks_label_like(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        tiny_crop = np.full((320, 250, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(tiny_crop)
        draw_product_field_structure(tiny_crop)

        tiny_pred = {
            "x": 80,
            "y": 220,
            "width": 100,
            "height": 80,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.99,
        }

        with mock.patch.object(crop, "crop_from_pred", return_value=tiny_crop):
            out, dropped = crop._stage1_collect_label_crops(img, [tiny_pred])

        self.assertEqual(out, [])
        self.assertEqual(dropped, 1)

    def test_stage1_collect_accepts_near_threshold_valid_main_box(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        valid_crop = np.full((320, 250, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(valid_crop)
        draw_product_field_structure(valid_crop)
        relaxed_box = (120, 150, 280, 245)
        pred = {
            "x": 200,
            "y": 200,
            "width": 160,
            "height": 95,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.99,
        }

        with mock.patch.object(crop, "box_from_pred", return_value=relaxed_box):
            with mock.patch.object(crop, "crop_from_pred", return_value=valid_crop):
                with mock.patch.object(crop, "_stage1_prepare_product_label_crop", return_value=valid_crop):
                    with mock.patch.object(crop, "_stage1_has_single_label_field_evidence", return_value=True):
                        out, dropped = crop._stage1_collect_product_label_entries(
                            img,
                            [pred],
                            enforce_relaxed_area_evidence=True,
                        )

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)

    def test_stage1_collect_accepts_near_threshold_valid_main_box_with_fallback_field_evidence(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        valid_crop = np.full((320, 250, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(valid_crop)
        draw_product_field_structure(valid_crop)
        relaxed_box = (120, 150, 280, 245)
        pred = {
            "x": 200,
            "y": 200,
            "width": 160,
            "height": 95,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.99,
        }

        with mock.patch.object(crop, "box_from_pred", return_value=relaxed_box):
            with mock.patch.object(crop, "crop_from_pred", return_value=valid_crop):
                with mock.patch.object(crop, "_stage1_prepare_product_label_crop", return_value=valid_crop):
                    with mock.patch.object(crop, "_stage1_has_single_label_field_evidence", return_value=False):
                        with mock.patch.object(crop, "_stage1_has_fallback_field_evidence", return_value=True):
                            out, dropped = crop._stage1_collect_product_label_entries(
                                img,
                                [pred],
                                enforce_relaxed_area_evidence=True,
                            )

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)

    def test_stage1_collect_uses_upward_expanded_crop_when_it_recovers_single_label(self):
        img = np.full((900, 1200, 3), 255, dtype=np.uint8)
        low_crop = np.full((250, 500, 3), 255, dtype=np.uint8)
        expanded_crop = np.full((320, 500, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(expanded_crop)
        draw_product_field_structure(expanded_crop)
        base_box = (180, 220, 700, 430)
        expanded_box = (180, 140, 700, 430)
        pred = {
            "x": 440,
            "y": 320,
            "width": 520,
            "height": 210,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }

        with mock.patch.object(crop, "box_from_pred", return_value=base_box):
            with mock.patch.object(crop, "crop_from_pred", return_value=low_crop):
                with mock.patch.object(
                    crop,
                    "_stage1_prepare_product_label_crop",
                    side_effect=[low_crop, expanded_crop],
                ):
                    with mock.patch.object(crop, "box_from_pred_asym", return_value=expanded_box):
                        with mock.patch.object(crop, "crop_from_box", return_value=expanded_crop):
                            with mock.patch.object(
                                crop,
                                "_stage1_has_single_label_field_evidence",
                                side_effect=lambda candidate: candidate is expanded_crop,
                            ):
                                out, dropped = crop._stage1_collect_product_label_entries(
                                    img,
                                    [pred],
                                    require_single_label_field_evidence=True,
                                )

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)
        self.assertEqual(out[0]["box"], expanded_box)
        self.assertIs(out[0]["crop"], expanded_crop)

    def test_stage1_collect_uses_edge_recovered_crop_when_left_expansion_recovers_single_label(self):
        img = np.full((900, 1200, 3), 255, dtype=np.uint8)
        base_crop = np.full((280, 460, 3), 255, dtype=np.uint8)
        expanded_crop = np.full((280, 540, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(expanded_crop)
        draw_product_field_structure(expanded_crop)
        base_box = (220, 180, 700, 430)
        expanded_box = (150, 180, 710, 430)
        pred = {
            "x": 460,
            "y": 305,
            "width": 480,
            "height": 220,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.97,
        }

        with mock.patch.object(crop, "box_from_pred", return_value=base_box):
            with mock.patch.object(crop, "crop_from_pred", return_value=base_crop):
                with mock.patch.object(crop, "_stage1_prepare_product_label_crop", side_effect=[base_crop, expanded_crop]):
                    with mock.patch.object(crop, "_stage1_maybe_retry_upward_crop", return_value=(base_box, base_crop)):
                        with mock.patch.object(crop, "_stage1_label_crop_edge_ink_ratios", return_value={"top": 0.0, "left": 0.18, "right": 0.0}):
                            with mock.patch.object(crop, "box_from_pred_xy_asym", return_value=expanded_box):
                                with mock.patch.object(crop, "crop_from_box", return_value=expanded_crop):
                                    with mock.patch.object(
                                        crop,
                                        "_stage1_has_single_label_field_evidence",
                                        side_effect=lambda candidate: candidate is expanded_crop,
                                    ):
                                        out, dropped = crop._stage1_collect_product_label_entries(
                                            img,
                                            [pred],
                                            require_single_label_field_evidence=True,
                                        )

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)
        self.assertEqual(out[0]["box"], expanded_box)
        self.assertIs(out[0]["crop"], expanded_crop)

    def test_stage1_collect_accepts_top_recovery_when_relaxed_field_evidence_recovers(self):
        img = np.full((900, 1200, 3), 255, dtype=np.uint8)
        base_crop = np.full((240, 470, 3), 255, dtype=np.uint8)
        expanded_crop = np.full((312, 490, 3), 255, dtype=np.uint8)
        base_box = (220, 210, 700, 430)
        expanded_box = (220, 150, 700, 430)
        pred = {
            "x": 460,
            "y": 320,
            "width": 480,
            "height": 220,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.96,
        }

        with mock.patch.object(crop, "box_from_pred", return_value=base_box):
            with mock.patch.object(crop, "crop_from_pred", return_value=base_crop):
                with mock.patch.object(crop, "_stage1_prepare_product_label_crop", side_effect=[base_crop, expanded_crop]):
                    with mock.patch.object(crop, "_stage1_maybe_retry_upward_crop", return_value=(base_box, base_crop)):
                        with mock.patch.object(crop, "_stage1_label_crop_edge_ink_ratios", return_value={"top": 0.08, "left": 0.0, "right": 0.0}):
                            with mock.patch.object(crop, "box_from_pred_xy_asym", return_value=expanded_box):
                                with mock.patch.object(crop, "crop_from_box", return_value=expanded_crop):
                                    with mock.patch.object(crop, "_stage1_has_single_label_field_evidence", return_value=False):
                                        with mock.patch.object(
                                            crop,
                                            "_stage1_has_relaxed_single_label_field_evidence",
                                            side_effect=lambda candidate: candidate is expanded_crop,
                                        ):
                                            with mock.patch.object(
                                                crop,
                                                "stage1_is_partial_product_label_crop",
                                                side_effect=lambda candidate: candidate is expanded_crop,
                                            ):
                                                out, dropped = crop._stage1_collect_product_label_entries(
                                                    img,
                                                    [pred],
                                                    require_single_label_field_evidence=True,
                                                )

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)
        self.assertEqual(out[0]["box"], expanded_box)
        self.assertIs(out[0]["crop"], expanded_crop)

    def test_stage1_collect_rejects_conflicting_duplicate_main_candidate(self):
        img = np.full((900, 1200, 3), 255, dtype=np.uint8)
        full_crop = np.full((320, 520, 3), 255, dtype=np.uint8)
        partial_crop = np.full((260, 500, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(full_crop)
        draw_product_field_structure(full_crop)
        draw_top_right_product_mark(partial_crop)
        draw_product_field_structure(partial_crop)
        pred_a = {
            "x": 420,
            "y": 310,
            "width": 520,
            "height": 220,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.99,
        }
        pred_b = {
            "x": 430,
            "y": 332,
            "width": 500,
            "height": 215,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.93,
        }
        boxes = [(160, 150, 700, 430), (178, 196, 688, 430)]
        crops = [full_crop, partial_crop]

        with mock.patch.object(crop, "box_from_pred", side_effect=boxes):
            with mock.patch.object(crop, "crop_from_pred", side_effect=crops):
                with mock.patch.object(crop, "_stage1_prepare_product_label_crop", side_effect=crops):
                    with mock.patch.object(crop, "box_from_pred_asym", side_effect=boxes):
                        with mock.patch.object(
                            crop,
                            "_stage1_has_single_label_field_evidence",
                            return_value=True,
                        ):
                            out, dropped = crop._stage1_collect_product_label_entries(img, [pred_b, pred_a])

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(out[0]["box"], boxes[0])

    def test_stage2_crop_part_no_returns_padded_heuristic_crop_for_edge_clipped_fallback(self):
        img = np.full((260, 520, 3), 255, dtype=np.uint8)
        fallback = np.full((70, 220, 3), 255, dtype=np.uint8)
        padded = np.full((88, 248, 3), 255, dtype=np.uint8)

        with mock.patch.object(crop, "_stage2_crop_part_no_from_pred", return_value=None):
            with mock.patch.object(
                crop,
                "trim_part_no_crop_before_lower_neighbor",
                side_effect=lambda candidate, trim_text_neighbor=False: candidate,
            ):
                with mock.patch.object(crop, "crop_part_no_field", return_value=fallback):
                    with mock.patch.object(crop, "part_no_crop_has_complete_1d_barcode", side_effect=lambda candidate, min_span_ratio=0.18: candidate is padded):
                        with mock.patch.object(crop, "part_no_crop_is_structurally_valid", return_value=False):
                            with mock.patch.object(crop, "part_no_crop_has_text_above_barcode", return_value=True):
                                with mock.patch.object(crop, "part_no_crop_has_lower_neighbor_content", return_value=False):
                                    with mock.patch.object(crop, "part_no_crop_contains_1d_barcode", return_value=True):
                                        with mock.patch.object(crop, "pad_part_no_crop_quiet_zone", return_value=padded):
                                            out, kind, ok = crop._stage2_crop_part_no(img, None)

        self.assertIs(out, padded)
        self.assertEqual(kind, "heuristic_padded")
        self.assertTrue(ok)

    def test_stage2_crop_part_no_prefers_heuristic_when_detector_padded_is_not_complete(self):
        img = np.full((260, 520, 3), 255, dtype=np.uint8)
        detected = np.full((62, 188, 3), 255, dtype=np.uint8)
        padded_detected = np.full((76, 214, 3), 255, dtype=np.uint8)
        heuristic = np.full((96, 250, 3), 255, dtype=np.uint8)
        pred = {"x": 180, "y": 80, "width": 110, "height": 42}

        with mock.patch.object(crop, "_stage2_crop_part_no_from_pred", return_value=detected):
            with mock.patch.object(
                crop,
                "trim_part_no_crop_before_lower_neighbor",
                side_effect=lambda candidate, trim_text_neighbor=False: candidate,
            ):
                with mock.patch.object(crop, "crop_part_no_field", return_value=heuristic):
                    with mock.patch.object(
                        crop,
                        "part_no_crop_has_complete_1d_barcode",
                        side_effect=lambda candidate, min_span_ratio=0.18: False,
                    ):
                        with mock.patch.object(
                            crop,
                            "part_no_crop_is_structurally_valid",
                            side_effect=lambda candidate, allow_partial=False: candidate is heuristic,
                        ):
                            with mock.patch.object(crop, "part_no_crop_has_text_above_barcode", return_value=True):
                                with mock.patch.object(crop, "part_no_crop_has_lower_neighbor_content", return_value=False):
                                    with mock.patch.object(crop, "part_no_crop_contains_1d_barcode", return_value=True):
                                        with mock.patch.object(crop, "part_no_crop_has_edge_clipped_barcode", return_value=False):
                                            with mock.patch.object(crop, "part_no_crop_has_partial_1d_barcode", return_value=False):
                                                with mock.patch.object(crop, "pad_part_no_crop_quiet_zone", return_value=padded_detected):
                                                    out, kind, ok = crop._stage2_crop_part_no(img, pred)

        self.assertIs(out, heuristic)
        self.assertEqual(kind, "heuristic")
        self.assertFalse(ok)

    def test_stage1_collect_supplements_from_secondary_hardcase_model(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        full_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(full_crop)
        draw_product_field_structure(full_crop)

        partial_crop = np.full((320, 250, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(partial_crop)
        draw_product_field_structure(partial_crop)

        high_pred = {
            "x": 240,
            "y": 350,
            "width": 420,
            "height": 240,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }
        secondary_pred = {
            "x": 720,
            "y": 700,
            "width": 260,
            "height": 280,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.81,
        }
        single_label_field_preds = [  # noqa: F841  (documents the fixture; pre-existing)
            {"class": crop.MODEL2_PART_NO_CLASS, "x": 85, "y": 45, "width": 150, "height": 36},
            {"class": crop.MODEL2_SN_CLASS, "x": 180, "y": 90, "width": 170, "height": 42},
        ]

        def fake_crop_from_pred(_img, pred, *_args, **_kwargs):
            return full_crop if pred["confidence"] > 0.9 else partial_crop

        def fake_infer_with_resize(image_or_path, _path, model_id=None, **_kwargs):
            if model_id == crop.MODEL1_HARDCASE_SUPPLEMENT_ID:
                return [secondary_pred]
            return []

        with mock.patch.object(crop, "crop_from_pred", side_effect=fake_crop_from_pred):
            with mock.patch.object(crop, "_stage1_hardcase_model_available", return_value=True):
                with mock.patch.object(crop, "infer_with_resize", side_effect=fake_infer_with_resize):
                    with mock.patch.object(
                        crop,
                        "_stage1_has_single_label_field_evidence",
                        side_effect=lambda img: img is full_crop or img is partial_crop,
                    ):
                        out, dropped = crop._stage1_collect_label_crops(img, [high_pred])

        self.assertEqual(len(out), 2)
        self.assertEqual(dropped, 0)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))
        self.assertTrue(crop.stage1_is_product_label_crop(out[1]))

    def test_stage1_collect_rejects_portrait_secondary_hardcase_supplement(self):
        img = np.full((1600, 2200, 3), 255, dtype=np.uint8)
        full_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(full_crop)
        draw_product_field_structure(full_crop)

        portrait_crop = np.full((900, 520, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(portrait_crop)
        draw_product_field_structure(portrait_crop)

        high_pred = {
            "x": 900,
            "y": 520,
            "width": 780,
            "height": 500,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.98,
        }
        secondary_pred = {
            "x": 320,
            "y": 540,
            "width": 520,
            "height": 980,
            "class": crop.MODEL1_LABEL_CLASS,
            "confidence": 0.81,
        }

        def fake_crop_from_pred(_img, pred, *_args, **_kwargs):
            return full_crop if pred["confidence"] > 0.9 else portrait_crop

        def fake_infer_with_resize(image_or_path, _path, model_id=None, **_kwargs):
            if model_id == crop.MODEL1_HARDCASE_SUPPLEMENT_ID:
                return [secondary_pred]
            return []

        with mock.patch.object(crop, "crop_from_pred", side_effect=fake_crop_from_pred):
            with mock.patch.object(crop, "_stage1_hardcase_model_available", return_value=True):
                with mock.patch.object(crop, "infer_with_resize", side_effect=fake_infer_with_resize):
                    with mock.patch.object(
                        crop,
                        "_stage1_has_single_label_field_evidence",
                        side_effect=lambda img: img is full_crop or img is portrait_crop,
                    ):
                        out, dropped = crop._stage1_collect_label_crops(img, [high_pred])

        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 1)
        self.assertTrue(crop.stage1_is_product_label_crop(out[0]))

    def test_stage1_hardcase_supplement_is_disabled_when_primary_already_uses_same_model(self):
        fake_local_yolo = types.SimpleNamespace(
            DEFAULT_MODEL_SPECS={
                crop.MODEL1_ID: types.SimpleNamespace(path="same.onnx"),
                crop.MODEL1_HARDCASE_SUPPLEMENT_ID: types.SimpleNamespace(path="same.onnx"),
            },
            model_ids_share_same_path=lambda *_args, **_kwargs: True,
        )

        with mock.patch.dict(sys.modules, {"local_yolo": fake_local_yolo}):
            self.assertFalse(crop._stage1_hardcase_model_available())

    def test_stage1_orientation_keeps_normal_top_right_product_mark(self):
        img = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(img)

        with mock.patch.dict(
            crop.os.environ,
            {"CROP_STAGE1_ORIENTATION_NORMALIZE": "1"},
        ):
            with mock.patch.object(crop, "_stage1_qr_orientation_decision", return_value="upright"):
                self.assertFalse(crop.stage1_should_rotate_180_label(img))
                self.assertIs(crop.stage1_normalize_label_orientation(img), img)

    def test_stage1_orientation_uses_qr_position_when_available(self):
        img = np.full((260, 480, 3), 255, dtype=np.uint8)

        with mock.patch.dict(
            crop.os.environ,
            {"CROP_STAGE1_ORIENTATION_NORMALIZE": "1"},
        ):
            with mock.patch.object(crop, "_stage1_qr_orientation_decision", return_value="rotate_180"):
                self.assertTrue(crop.stage1_should_rotate_180_label(img))

    def test_stage1_qr_orientation_ignores_mid_left_false_positive(self):
        img = np.full((286, 717, 3), 255, dtype=np.uint8)

        with mock.patch.object(crop, "_stage1_qr_centers", return_value=[(105, 131)]):
            self.assertEqual(crop._stage1_qr_orientation_decision(img), "")

        with mock.patch.object(crop, "_stage1_qr_centers", return_value=[(105, 175)]):
            self.assertEqual(crop._stage1_qr_orientation_decision(img), "rotate_180")

    def test_stage1_orientation_does_not_flip_when_right_mark_is_stronger(self):
        img = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(img)
        img[175:220, 35:115] = 0

        with mock.patch.dict(
            crop.os.environ,
            {"CROP_STAGE1_ORIENTATION_NORMALIZE": "1"},
        ):
            with mock.patch.object(crop, "_stage1_qr_orientation_decision", return_value=""):
                self.assertFalse(crop.stage1_should_rotate_180_label(img))

    def test_stage1_orientation_keeps_strong_upright_corner_despite_left_noise(self):
        img = np.full((260, 480, 3), 255, dtype=np.uint8)

        with mock.patch.dict(
            crop.os.environ,
            {"CROP_STAGE1_ORIENTATION_NORMALIZE": "1"},
        ):
            with mock.patch.object(crop, "_stage1_qr_orientation_decision", return_value=""):
                with mock.patch.object(crop, "_stage1_left_bottom_product_mark_score", return_value=0.60):
                    with mock.patch.object(crop, "_stage1_right_product_mark_score", return_value=0.20):
                        with mock.patch.object(
                            crop,
                            "_stage1_corner_mark_score",
                            side_effect=[0.50, 0.30, 0.10, 0.05],
                        ):
                            self.assertFalse(crop.stage1_should_rotate_180_label(img))

    def test_stage1_orientation_rotates_inverted_bottom_left_product_mark(self):
        upright = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(upright)
        inverted = crop.cv2.rotate(upright, crop.cv2.ROTATE_180)

        with mock.patch.dict(
            crop.os.environ,
            {"CROP_STAGE1_ORIENTATION_NORMALIZE": "1"},
        ):
            with mock.patch.object(crop, "_stage1_qr_orientation_decision", return_value=""):
                self.assertTrue(crop.stage1_should_rotate_180_label(inverted))
                out = crop.stage1_normalize_label_orientation(inverted)

        self.assertTrue(np.array_equal(out, upright))

    def test_stage1_retries_rotated_source_when_original_has_no_product_label(self):
        img = np.full((800, 1000, 3), 255, dtype=np.uint8)
        product_crop = np.full((260, 480, 3), 255, dtype=np.uint8)
        draw_top_right_product_mark(product_crop)
        draw_product_field_structure(product_crop)
        pred = {"x": 500, "y": 400, "width": 520, "height": 320, "class": crop.MODEL1_LABEL_CLASS}
        calls = {"count": 0}

        def infer_side_effect(_img, _path, model_id, **_kwargs):
            if model_id == crop.MODEL1_ID:
                calls["count"] += 1
                return [] if calls["count"] == 1 else [pred]
            return []

        with mock.patch.object(crop, "read_image", return_value=img):
            with mock.patch.object(crop, "infer_with_resize", side_effect=infer_side_effect) as infer_mock:
                with mock.patch.object(crop, "crop_from_pred", return_value=product_crop):
                    with mock.patch.object(crop, "save_png_required", side_effect=lambda path, _img, _ctx: path):
                        with mock.patch.object(crop, "_stage1_has_single_label_field_evidence", return_value=True):
                            with mock.patch.dict(
                                crop.os.environ,
                                {
                                    "CROP_STAGE1_ROTATION_RETRY": "1",
                                    "CROP_STAGE1_HARDCASE_MODEL_SUPPLEMENT": "0",
                                },
                            ):
                                out = crop.stage1_crop_labels("rotated.jpg")

        self.assertEqual(len(out), 1)
        self.assertEqual(infer_mock.call_count, 2)
        self.assertEqual(os.path.basename(out[0]), "rotated.jpg__label_1.png")

    def test_stage1_rotation_retry_defaults_to_180_only(self):
        with mock.patch.dict(crop.os.environ, {}, clear=True):
            self.assertEqual(crop.stage1_rotation_retry_angles(), (180,))

    def test_stage1_rotation_retry_allows_explicit_all_angles(self):
        with mock.patch.dict(crop.os.environ, {"CROP_STAGE1_ROTATION_RETRY": "all"}, clear=True):
            self.assertEqual(crop.stage1_rotation_retry_angles(), (90, 270, 180))

    def test_stage1_rotation_retry_can_be_disabled(self):
        with mock.patch.dict(crop.os.environ, {"CROP_STAGE1_ROTATION_RETRY": "0"}, clear=True):
            self.assertEqual(crop.stage1_rotation_retry_angles(), ())

    def test_stage1_tighten_removes_outer_box_margin(self):
        img = np.full((420, 720, 3), 255, dtype=np.uint8)
        crop.cv2.putText(
            img,
            "Make SME Network Easier and Smarter",
            (130, 55),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.rectangle(img, (95, 110), (625, 335), (0, 0, 0), 2)
        crop.cv2.putText(
            img,
            "Part No.: 50087149",
            (118, 145),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.putText(
            img,
            "Model: AP162E",
            (118, 185),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        for x in range(118, 360, 8):
            img[205:250, x:x + 3] = 0
        img[130:170, 555:610] = 0
        crop.cv2.line(img, (95, 270), (625, 270), (0, 0, 0), 2)

        out = crop.stage1_tighten_label_crop(img)

        self.assertIsNotNone(out)
        self.assertLess(out.shape[0], 280)
        self.assertLess(out.shape[1], 590)
        self.assertTrue(crop.stage1_is_product_label_crop(out))

    def test_stage1_tighten_keeps_bottom_text_margin(self):
        img = np.full((420, 720, 3), 235, dtype=np.uint8)
        img[65:365, 80:640] = 255
        crop.cv2.rectangle(img, (105, 100), (610, 292), (0, 0, 0), 2)
        crop.cv2.putText(
            img,
            "Part No.: 50087147",
            (125, 138),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.putText(
            img,
            "Model: AP362E",
            (125, 178),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        for x in range(125, 395, 9):
            img[205:252, x:x + 3] = 0
        img[120:170, 548:598] = 0
        crop.cv2.putText(
            img,
            "EAN: 6901443456949",
            (125, 326),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.putText(
            img,
            "Remark:",
            (125, 352),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )

        out = crop.stage1_tighten_label_crop(img)

        self.assertIsNotNone(out)
        self.assertLess(out.shape[0], 360)
        self.assertLess(out.shape[1], 590)
        self.assertGreater(np.count_nonzero(out[-52:, :, :] < 80), 80)

    def test_stage1_tighten_preserves_side_field_context(self):
        img = np.full((340, 1000, 3), 245, dtype=np.uint8)
        img[20:320, 20:960] = 255
        crop.cv2.rectangle(img, (25, 32), (515, 300), (0, 0, 0), 2)
        crop.cv2.putText(
            img,
            "Part No.: 98012123",
            (48, 70),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        crop.cv2.putText(
            img,
            "Model: S380-L4P1T",
            (48, 132),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        draw_1d_barcode(img, 48, 84, 350, 120, step=8, bar_width=3)
        draw_1d_barcode(img, 540, 112, 810, 158, step=8, bar_width=3)
        crop.cv2.putText(
            img,
            "SN: 4E25A0068800",
            (540, 92),
            crop.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            crop.cv2.LINE_AA,
        )
        for x1, y1, x2, y2 in (
            (825, 38, 878, 72),
            (890, 40, 930, 72),
            (835, 90, 878, 122),
            (895, 92, 942, 126),
        ):
            img[y1:y2, x1:x2] = 0

        self.assertTrue(crop.stage1_is_product_label_crop(img))

        out = crop.stage1_tighten_label_crop(img)

        self.assertIsNotNone(out)
        self.assertGreater(out.shape[1], 850)
        self.assertTrue(crop.stage1_is_product_label_crop(out))


if __name__ == "__main__":
    unittest.main()
