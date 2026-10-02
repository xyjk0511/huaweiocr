import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from tools import run_stage1_review_export as current
from tools import run_stage1_review_export_head_snapshot as snapshot
from tools import merge_stage1_hardcase_zip, rebuild_yolo26_2class_dataset
from tools.stage1_review_images import save_image


class ReviewExportTest(unittest.TestCase):
    def test_exporters_reject_existing_or_overlapping_outputs_without_erasing_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input"
            source.mkdir()
            marker = source / "sample.png"
            marker.write_bytes(b"preserve input")
            existing = root / "previous"
            existing.mkdir()
            receipt = existing / "summary.csv"
            receipt.write_bytes(b"preserve evidence")
            for tool in (current, snapshot):
                for output in (source, root, source / "nested", existing):
                    with self.subTest(tool=tool.__name__, output=output):
                        argv = ["export", "--input", str(source), "--out", str(output)]
                        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr", new_callable=io.StringIO):
                            with self.assertRaises(SystemExit) as exc:
                                tool.main()
                        self.assertEqual(exc.exception.code, 2)
                        self.assertEqual(marker.read_bytes(), b"preserve input")
                        self.assertEqual(receipt.read_bytes(), b"preserve evidence")
            self.assertFalse((source / "nested").exists())

    def test_exporters_preserve_distinct_inputs_and_their_crop_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input"
            names = ("a.jpg", "a.png", "sub/a.png", "sub__a.png", "sub%5F%5Fa.png",
                     "sub_/a.png", "sub/_a.png")
            for name in names:
                save_image(source / name, np.full((32, 40, 3), 180, dtype=np.uint8))
            entry = {"box": (2, 3, 12, 13), "confidence": 0.9, "class_id": 1,
                     "crop": np.full((8, 9, 3), 90, dtype=np.uint8)}
            for tool, shape in ((current, (10, 10)), (snapshot, (8, 9))):
                with self.subTest(tool=tool.__name__):
                    output = root / tool.__name__
                    argv = ["export", "--input", str(source), "--out", str(output)]
                    with mock.patch.object(sys, "argv", argv), \
                         mock.patch.object(tool, "_detect_entries", return_value=([entry], 0, 0)), \
                         mock.patch.object(snapshot, "_load_head_snapshot", return_value=(current.crop, None)), \
                         mock.patch("sys.stdout", new_callable=io.StringIO):
                        tool.main()
                    with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual({row["relative_image"] for row in rows}, set(names))
                    self.assertEqual(len({row["crop_path"] for row in rows}), len(names))
                    for row in rows:
                        image = cv2.imdecode(np.fromfile(row["crop_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
                        self.assertEqual(image.shape[:2], shape)
                        self.assertEqual(row["class_id"], "1")
                    self.assertTrue((output / "overlay_contact_sheet.jpg").is_file())

    def test_dataset_tools_refuse_to_replace_existing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "dataset"
            source.mkdir()
            output.mkdir()
            marker = output / "annotations.txt"
            marker.write_bytes(b"keep annotations")
            with mock.patch.object(merge_stage1_hardcase_zip, "BASE_DATASET", source), \
                 mock.patch.object(merge_stage1_hardcase_zip, "OUT_DATASET", output):
                with self.assertRaises(FileExistsError):
                    merge_stage1_hardcase_zip.copy_base_dataset()
            argv = ["rebuild", "--zip", str(root / "input.zip"), "--out", str(output)]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                rebuild_yolo26_2class_dataset.main()
            self.assertEqual(marker.read_bytes(), b"keep annotations")


if __name__ == "__main__":
    unittest.main()
