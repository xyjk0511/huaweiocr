import tempfile
import unittest
import zipfile
from pathlib import Path

from huaweiocr.io.feedback import build_feedback_package


class FeedbackPackageTests(unittest.TestCase):
    def test_build_feedback_package_collects_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            stage2 = run_dir / "stage2_fields"
            miss_model = stage2 / "miss_model"
            miss_both = stage2 / "miss_both"
            miss_model.mkdir(parents=True)
            miss_both.mkdir(parents=True)
            (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
            (run_dir / "source_manifest.jsonl").write_text("{}\n", encoding="utf-8")
            (stage2 / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
            (stage2 / "model_sn_ocr.jsonl").write_text("{}\n", encoding="utf-8")
            (miss_model / "a.png").write_bytes(b"png-a")
            (miss_model / "b.png").write_bytes(b"png-b")
            (miss_both / "c.png").write_bytes(b"png-c")

            out_zip = root / "new" / "feedback.zip"
            stats = build_feedback_package(str(run_dir), str(out_zip))

            self.assertTrue(out_zip.is_file())
            self.assertEqual(stats["zip_path"], str(out_zip))
            self.assertEqual(stats["files"], 7)
            self.assertEqual(stats["skipped"], 0)
            self.assertGreater(stats["zip_bytes"], 0)
            self.assertEqual(stats["misses"]["miss_model"], 2)
            self.assertEqual(stats["misses"]["miss_sn"], 0)
            self.assertEqual(stats["misses"]["miss_both"], 1)
            self.assertEqual(stats["misses"]["failed"], 0)
            with zipfile.ZipFile(out_zip) as zf:
                names = set(zf.namelist())
            self.assertIn("run_summary.json", names)
            self.assertIn("source_manifest.jsonl", names)
            self.assertIn("stage2_fields/manifest.jsonl", names)
            self.assertIn("stage2_fields/model_sn_ocr.jsonl", names)
            self.assertIn("stage2_fields/miss_model/a.png", names)
            self.assertIn("stage2_fields/miss_model/b.png", names)
            self.assertIn("stage2_fields/miss_both/c.png", names)
            self.assertNotIn("stage2_fields/miss_sn", names)

    def test_build_feedback_package_skips_missing_inputs_and_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")

            out_zip = root / "missing" / "nested" / "feedback.zip"
            stats = build_feedback_package(str(run_dir), str(out_zip))

            self.assertTrue(out_zip.is_file())
            self.assertEqual(stats["files"], 1)
            self.assertEqual(stats["skipped"], 0)
            self.assertEqual(sum(stats["misses"].values()), 0)
            with zipfile.ZipFile(out_zip) as zf:
                self.assertEqual(zf.namelist(), ["run_summary.json"])


if __name__ == "__main__":
    unittest.main()
