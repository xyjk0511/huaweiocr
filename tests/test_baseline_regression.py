import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_baseline_regression as baseline_tool  # noqa: E402


def _row(label_id, model="AP162E", sn="4E25A0170000", model_src="part_no_barcode", sn_src="barcode"):
    return {
        "label_id": label_id,
        "model": model,
        "sn": sn,
        "model_src": model_src,
        "sn_src": sn_src,
    }


class CompareResultsTest(unittest.TestCase):
    """Fast, no I/O: exercises the comparison logic that decides whether a
    115-image baseline run counts as a regression (validation/baseline/README.md).
    Runs in the default suite / CI; the heavy end-to-end check below does not."""

    def test_identical_results_pass(self):
        baseline = {"a": _row("a")}
        fresh = {"a": _row("a")}
        report = baseline_tool.compare_results(fresh, baseline)
        self.assertTrue(report.passed)
        self.assertEqual(report.missing_labels, [])
        self.assertEqual(report.extra_labels, [])
        self.assertEqual(report.value_mismatches, [])

    def test_value_mismatch_fails_regardless_of_source(self):
        baseline = {"a": _row("a", sn="4E25A0170000")}
        fresh = {"a": _row("a", sn="4E25A0170001")}
        report = baseline_tool.compare_results(fresh, baseline)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.value_mismatches), 1)
        self.assertEqual(report.value_mismatches[0][0], "a")
        self.assertEqual(report.value_mismatches[0][1], "sn")

    def test_missing_label_fails(self):
        baseline = {"a": _row("a"), "b": _row("b")}
        fresh = {"a": _row("a")}
        report = baseline_tool.compare_results(fresh, baseline)
        self.assertFalse(report.passed)
        self.assertEqual(report.missing_labels, ["b"])

    def test_extra_label_fails(self):
        baseline = {"a": _row("a")}
        fresh = {"a": _row("a"), "b": _row("b")}
        report = baseline_tool.compare_results(fresh, baseline)
        self.assertFalse(report.passed)
        self.assertEqual(report.extra_labels, ["b"])

    def test_source_family_drift_within_limit_passes_when_values_equal(self):
        # A barcode CLI timeout that falls back to OCR and recovers the same
        # value is documented as tolerable drift, not a regression.
        baseline = {"a": _row("a", sn_src="barcode")}
        fresh = {"a": _row("a", sn_src="ocr")}
        report = baseline_tool.compare_results(fresh, baseline, max_source_drift=5)
        self.assertTrue(report.passed)
        self.assertEqual(report.source_drift_labels, ["a"])

    def test_source_family_drift_beyond_limit_fails(self):
        baseline = {str(i): _row(str(i), sn_src="barcode") for i in range(6)}
        fresh = {str(i): _row(str(i), sn_src="ocr") for i in range(6)}
        report = baseline_tool.compare_results(fresh, baseline, max_source_drift=5)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.source_drift_labels), 6)

    def test_barcode_visual_and_consensus_are_not_pure_barcode_family(self):
        # Mirrors the model_barcode_hits accounting fix in scan2.py: barcode_visual
        # and barcode_ocr_consensus both required OCR, so a flip between "barcode"
        # and either of those must count as source drift, not be silently ignored
        # as "still barcode".
        baseline = {"a": _row("a", model_src="part_no_barcode")}
        fresh = {"a": _row("a", model_src="barcode_visual")}
        report = baseline_tool.compare_results(fresh, baseline)
        self.assertEqual(report.source_drift_labels, ["a"])

    def test_render_reports_fail_and_lists_mismatch(self):
        baseline = {"a": _row("a", sn="4E25A0170000")}
        fresh = {"a": _row("a", sn="4E25A0170001")}
        report = baseline_tool.compare_results(fresh, baseline)
        text = report.render()
        self.assertIn("RESULT: FAIL", text)
        self.assertIn("4E25A0170000", text)
        self.assertIn("4E25A0170001", text)


class LoadResultsByLabelTest(unittest.TestCase):
    def test_loads_jsonl_keyed_by_label_id_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "result.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(_row("a")) + "\n")
                f.write("\n")
                f.write(json.dumps(_row("b")) + "\n")
            loaded = baseline_tool.load_results_by_label(path)
        self.assertEqual(set(loaded), {"a", "b"})


class LocateResultJsonlTest(unittest.TestCase):
    def test_prefers_run_summary_output_paths(self):
        with tempfile.TemporaryDirectory() as root:
            summary = {"output_paths": {"result_jsonl": os.path.join(root, "custom", "out.jsonl")}}
            with open(os.path.join(root, "run_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f)
            resolved = baseline_tool.locate_result_jsonl(root)
        self.assertEqual(str(resolved), os.path.join(root, "custom", "out.jsonl"))

    def test_falls_back_to_default_layout_without_run_summary(self):
        with tempfile.TemporaryDirectory() as root:
            resolved = baseline_tool.locate_result_jsonl(root)
        self.assertEqual(
            str(resolved),
            str(Path(root) / "stage2_fields" / "model_sn_ocr.jsonl"),
        )


@unittest.skipUnless(
    os.environ.get("HUAWEIOCR_RUN_BASELINE_REGRESSION") == "1",
    "Heavy, opt-in: runs the full 115-image pipeline. Set "
    "HUAWEIOCR_RUN_BASELINE_REGRESSION=1 to run after touching crop.py/scan2.py "
    "(see validation/baseline/README.md). Throttles CROP_WORKERS/SCAN2_WORKERS=2 "
    "by default but still uses real CPU/OCR for several minutes -- never runs in CI "
    "(the 115 source photos are gitignored).",
)
class FullBaselineRegressionTest(unittest.TestCase):
    def test_115_image_baseline_has_no_regression(self):
        input_dir = REPO_ROOT / baseline_tool.DEFAULT_INPUT
        if not input_dir.is_dir():
            self.skipTest(
                f"{input_dir} not present (gitignored 115-image set; see "
                "validation/baseline/README.md to materialize it)"
            )
        out_dir = REPO_ROOT / "batch_runs" / "baseline_check_unittest"
        proc = baseline_tool.run_pipeline(input_dir, out_dir)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

        fresh = baseline_tool.load_results_by_label(baseline_tool.locate_result_jsonl(out_dir))
        baseline = baseline_tool.load_results_by_label(REPO_ROOT / baseline_tool.DEFAULT_BASELINE)
        report = baseline_tool.compare_results(fresh, baseline)
        self.assertTrue(report.passed, report.render())


if __name__ == "__main__":
    unittest.main()
