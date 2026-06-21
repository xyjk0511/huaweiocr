import importlib
import importlib
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

import local_yolo


class LocalYoloInferenceTests(unittest.TestCase):
    def test_default_field_detector_names_include_partno(self):
        self.assertEqual(
            local_yolo.DEFAULT_MODEL_SPECS["sn_model/9"].names,
            ("model", "partno", "sn"),
        )

    def test_decode_yolov8_output_returns_roboflow_style_predictions(self):
        output = np.zeros((1, 6, 10), dtype=np.float32)
        output[0, 0, 0] = 100.0
        output[0, 1, 0] = 80.0
        output[0, 2, 0] = 40.0
        output[0, 3, 0] = 20.0
        output[0, 4, 0] = 0.9
        output[0, 5, 0] = 0.1

        output[0, 0, 1] = 300.0
        output[0, 1, 1] = 200.0
        output[0, 2, 1] = 60.0
        output[0, 3, 1] = 30.0
        output[0, 4, 1] = 0.2
        output[0, 5, 1] = 0.8

        preds = local_yolo.decode_yolov8_output(
            output,
            names=("model", "sn"),
            original_shape=(640, 640, 3),
            scale=1.0,
            pad=(0, 0),
            conf_threshold=0.25,
            nms_threshold=0.45,
        )

        self.assertEqual([p["class"] for p in preds], ["model", "sn"])
        self.assertEqual([p["class_name"] for p in preds], ["model", "sn"])
        self.assertAlmostEqual(preds[0]["x"], 100.0)
        self.assertAlmostEqual(preds[0]["y"], 80.0)
        self.assertAlmostEqual(preds[0]["width"], 40.0)
        self.assertAlmostEqual(preds[0]["height"], 20.0)
        self.assertAlmostEqual(preds[0]["confidence"], 0.9, places=5)

    def test_local_yolo_client_returns_predictions_dict(self):
        class FakeDetector:
            def __init__(self, spec, **_kwargs):
                self.spec = spec

            def predict(self, image_path):
                return [{"class": self.spec.names[0], "image_path": image_path}]

        client = local_yolo.LocalYoloClient(
            model_specs={"model/1": local_yolo.ModelSpec("unused.onnx", ("label",))},
            detector_cls=FakeDetector,
        )

        self.assertEqual(
            client.infer("image.jpg", model_id="model/1"),
            {"predictions": [{"class": "label", "image_path": "image.jpg"}]},
        )

    def test_crop_defaults_to_local_backend_without_api_key(self):
        fake_inference_sdk = types.ModuleType("inference_sdk")
        fake_inference_sdk.InferenceHTTPClient = mock.Mock()

        class FakeLocalYoloClient:
            pass

        fake_local_yolo = types.ModuleType("local_yolo")
        fake_local_yolo.LocalYoloClient = FakeLocalYoloClient

        old_crop = sys.modules.pop("crop", None)
        try:
            with mock.patch.dict(
                sys.modules,
                {"inference_sdk": fake_inference_sdk, "local_yolo": fake_local_yolo},
            ):
                with mock.patch.dict(
                    os.environ,
                    {"CROP_INFERENCE_BACKEND": "", "API_KEY": ""},
                ):
                    os.environ.pop("CROP_INFERENCE_BACKEND", None)
                    os.environ.pop("API_KEY", None)
                    crop = importlib.import_module("crop")
                    self.assertEqual(crop.inference_backend(), "local")
                    self.assertIsInstance(crop.get_inference_client(), FakeLocalYoloClient)
                    fake_inference_sdk.InferenceHTTPClient.assert_not_called()
        finally:
            sys.modules.pop("crop", None)
            if old_crop is not None:
                sys.modules["crop"] = old_crop


if __name__ == "__main__":
    unittest.main()
