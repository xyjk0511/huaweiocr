import concurrent.futures
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import numpy as np


fake_inference_sdk = types.ModuleType("inference_sdk")
fake_inference_sdk.InferenceHTTPClient = object
sys.modules.setdefault("inference_sdk", fake_inference_sdk)

import crop  # noqa: E402  (fake inference_sdk must be registered before importing crop)


@contextmanager
def writable_tempdir():
    root = os.environ.get("HUAWEIOCR_TEST_TMP")
    if root:
        os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root or None) as tmpdir:
        yield tmpdir


class Stage1RejectAccountingTests(unittest.TestCase):
    def setUp(self):
        crop.reset_stage1_reject_counts()

    def tearDown(self):
        crop.reset_stage1_reject_counts()

    def counts(self):
        with crop.STAGE1_REJECT_COUNTS_LOCK:
            return dict(crop.STAGE1_REJECT_COUNTS)

    def test_counts_accumulate_and_reset(self):
        crop.record_stage1_reject("prepare_failed")
        crop.record_stage1_reject("prepare_failed")
        crop.record_stage1_reject("box_conflict")

        counts = self.counts()
        self.assertEqual(counts["prepare_failed"], 2)
        self.assertEqual(counts["box_conflict"], 1)
        self.assertEqual(sum(counts.values()), 3)

        crop.reset_stage1_reject_counts()

        self.assertEqual(self.counts(), {})

    def test_multithreaded_counts_are_not_lost(self):
        def add_many():
            for _ in range(50):
                crop.record_stage1_reject("threaded")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: add_many(), range(4)))

        counts = self.counts()
        self.assertEqual(counts["threaded"], 200)
        self.assertEqual(sum(counts.values()), 200)
        self.assertEqual(set(counts), {"threaded"})

    def test_debug_mode_saves_rejected_crop(self):
        img = np.zeros((8, 12, 3), dtype=np.uint8)
        with writable_tempdir() as tmpdir:
            with mock.patch.object(crop, "LOG_LEVEL", "debug"), mock.patch.object(crop, "FAILED_DIR", tmpdir):
                crop.record_stage1_reject("debug_reason", img, img_path=os.path.join("src", "sample.jpg"), index=7)

            reject_dir = os.path.join(tmpdir, "stage1_rejects", "debug_reason")
            self.assertTrue(os.path.isdir(reject_dir))
            files = os.listdir(reject_dir)

        self.assertEqual(self.counts()["debug_reason"], 1)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".png"))
        self.assertIn("sample", files[0])
        self.assertIn("0007", files[0])

    def test_non_debug_mode_counts_without_saving(self):
        img = np.zeros((8, 12, 3), dtype=np.uint8)
        with writable_tempdir() as tmpdir:
            with mock.patch.object(crop, "LOG_LEVEL", "info"), mock.patch.object(crop, "FAILED_DIR", tmpdir):
                with mock.patch.object(crop, "save_png") as save_png:
                    crop.record_stage1_reject("no_save", img, img_path="sample.jpg", index=1)

            reject_root = os.path.join(tmpdir, "stage1_rejects")
            self.assertFalse(os.path.exists(reject_root))

        self.assertEqual(self.counts()["no_save"], 1)
        save_png.assert_not_called()

    def test_save_failures_do_not_propagate(self):
        img = np.zeros((8, 12, 3), dtype=np.uint8)
        with writable_tempdir() as tmpdir:
            with mock.patch.object(crop, "LOG_LEVEL", "debug"), mock.patch.object(crop, "FAILED_DIR", tmpdir):
                with mock.patch.object(crop, "save_png", side_effect=RuntimeError("boom")) as save_png:
                    crop.record_stage1_reject("save_raises", img, img_path="sample.jpg", index=2)

        self.assertEqual(self.counts()["save_raises"], 1)
        self.assertEqual(save_png.call_count, 1)
        self.assertIn("save_raises", save_png.call_args.args[0])

    def test_raw_mode_collector_records_box_conflicts(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        pred = {"class": "label", "confidence": 0.9, "x": 200, "y": 200, "width": 120, "height": 60}
        duplicate = dict(pred, confidence=0.8)

        accepted, dropped = crop._stage1_collect_raw_label_entries(
            img, [pred, duplicate], img_path="raw_sample.jpg"
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(dropped, 0)
        self.assertEqual(self.counts().get("raw_box_conflict"), 1)

    def test_reason_and_missing_source_are_sanitized_for_debug_path(self):
        img = np.zeros((8, 12, 3), dtype=np.uint8)
        with writable_tempdir() as tmpdir:
            with mock.patch.object(crop, "LOG_LEVEL", "debug"), mock.patch.object(crop, "FAILED_DIR", tmpdir):
                crop.record_stage1_reject("../bad reason", img, index=3)

            reject_root = os.path.join(tmpdir, "stage1_rejects")
            reason_dirs = os.listdir(reject_root)
            files = os.listdir(os.path.join(reject_root, reason_dirs[0]))

        self.assertEqual(reason_dirs, ["bad_reason"])
        self.assertEqual(len(files), 1)
        self.assertIn("unknown", files[0])
        self.assertIn("0003", files[0])


if __name__ == "__main__":
    unittest.main()
