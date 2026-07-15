"""Automate the manual walkthrough in validation/baseline/README.md.

Runs run_all.py against the gitignored 115-image baseline input, then diffs
the fresh stage2_fields/model_sn_ocr.jsonl against the frozen reference at
validation/baseline/old_dist_run/model_sn_ocr.jsonl.

Judgment matches the README:
  - hard gate: label_id set and every model/sn value must match exactly.
  - soft gate: a row may flip between the barcode family (barcode /
    part_no_barcode) and everything else (ocr*, part_no_hint, part_no_ocr,
    barcode_visual, barcode_ocr_consensus -- the latter two require OCR, see
    the model_barcode_hits accounting in scan2.py) while keeping the same
    value, but only up to --max-source-drift rows before it counts as a
    regression.

CLI usage:
    python tools/check_baseline_regression.py
    python tools/check_baseline_regression.py --skip-run --out batch_runs/baseline_check_final
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT = "batch_runs/baseline_input"
DEFAULT_OUT = "batch_runs/baseline_check_auto"
DEFAULT_BASELINE = "validation/baseline/old_dist_run/model_sn_ocr.jsonl"
DEFAULT_MAX_SOURCE_DRIFT = 5
COMPARED_FIELDS = ("model", "sn")

# Sources that come from an actual barcode decode with no OCR involvement.
# barcode_visual and barcode_ocr_consensus both require OCR (see the
# model_barcode_hits fix in scan2.py) so they belong to the "other" family.
BARCODE_FAMILY_SOURCES = {"barcode", "part_no_barcode"}


def _source_family(src: str) -> str:
    return "barcode" if src in BARCODE_FAMILY_SOURCES else "ocr_or_other"


def load_results_by_label(path) -> dict:
    """Load a model_sn_ocr.jsonl file keyed by label_id."""
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            results[row["label_id"]] = row
    return results


@dataclass
class RegressionReport:
    baseline_count: int
    fresh_count: int
    missing_labels: list = field(default_factory=list)
    extra_labels: list = field(default_factory=list)
    value_mismatches: list = field(default_factory=list)
    source_drift_labels: list = field(default_factory=list)
    max_source_drift: int = DEFAULT_MAX_SOURCE_DRIFT

    @property
    def passed(self) -> bool:
        return (
            not self.missing_labels
            and not self.extra_labels
            and not self.value_mismatches
            and len(self.source_drift_labels) <= self.max_source_drift
        )

    def render(self) -> str:
        lines = [
            "=== Baseline regression report ===",
            f"baseline rows: {self.baseline_count}   fresh rows: {self.fresh_count}",
        ]

        def _list(title, items, limit=20):
            lines.append(f"{title}: {len(items)}")
            for item in items[:limit]:
                lines.append(f"  - {item}")
            if len(items) > limit:
                lines.append(f"  ... and {len(items) - limit} more")

        if self.missing_labels:
            _list("MISSING labels (in baseline, not in fresh run)", self.missing_labels)
        if self.extra_labels:
            _list("EXTRA labels (in fresh run, not in baseline)", self.extra_labels)
        if self.value_mismatches:
            lines.append(f"VALUE mismatches: {len(self.value_mismatches)}")
            for label_id, field_name, baseline_value, fresh_value in self.value_mismatches[:20]:
                lines.append(
                    f"  - {label_id} [{field_name}] baseline={baseline_value!r} fresh={fresh_value!r}"
                )
            if len(self.value_mismatches) > 20:
                lines.append(f"  ... and {len(self.value_mismatches) - 20} more")
        lines.append(
            f"source drift (barcode<->OCR family, value-equal rows): "
            f"{len(self.source_drift_labels)} (limit {self.max_source_drift})"
        )
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def compare_results(fresh: dict, baseline: dict, max_source_drift: int = DEFAULT_MAX_SOURCE_DRIFT) -> RegressionReport:
    missing = sorted(set(baseline) - set(fresh))
    extra = sorted(set(fresh) - set(baseline))
    value_mismatches = []
    drift_labels = []

    for label_id in sorted(set(baseline) & set(fresh)):
        base_row, fresh_row = baseline[label_id], fresh[label_id]
        row_has_value_mismatch = False
        for field_name in COMPARED_FIELDS:
            base_value = base_row.get(field_name, "")
            fresh_value = fresh_row.get(field_name, "")
            if base_value != fresh_value:
                value_mismatches.append((label_id, field_name, base_value, fresh_value))
                row_has_value_mismatch = True
        if not row_has_value_mismatch:
            model_drift = _source_family(base_row.get("model_src", "")) != _source_family(
                fresh_row.get("model_src", "")
            )
            sn_drift = _source_family(base_row.get("sn_src", "")) != _source_family(
                fresh_row.get("sn_src", "")
            )
            if model_drift or sn_drift:
                drift_labels.append(label_id)

    return RegressionReport(
        baseline_count=len(baseline),
        fresh_count=len(fresh),
        missing_labels=missing,
        extra_labels=extra,
        value_mismatches=value_mismatches,
        source_drift_labels=drift_labels,
        max_source_drift=max_source_drift,
    )


def run_pipeline(input_dir, out_dir, python_exe=None, extra_env=None, log_level="info"):
    """Invoke run_all.py as a subprocess (matches the documented developer workflow).

    Throttles CROP_WORKERS/SCAN2_WORKERS to 2 by default so a full 115-image run
    does not saturate the machine (see the batch-runs-low-priority guidance in
    validation/baseline/README.md); callers can override via extra_env.
    """
    env = os.environ.copy()
    env.setdefault("CROP_WORKERS", "2")
    env.setdefault("SCAN2_WORKERS", "2")
    if extra_env:
        env.update(extra_env)
    exe = python_exe or sys.executable
    cmd = [exe, "run_all.py", "--input", str(input_dir), "--out", str(out_dir), "--log-level", log_level]
    # crop.py reconfigures the child's stdout to utf-8 on import; decode
    # explicitly rather than relying on the parent locale (cp936 on zh-CN
    # Windows), which would raise on the child's Chinese log lines.
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def locate_result_jsonl(out_dir) -> Path:
    out_dir = Path(out_dir)
    summary_path = out_dir / "run_summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            result = data.get("output_paths", {}).get("result_jsonl")
            if result:
                return Path(result)
        except (OSError, json.JSONDecodeError):
            pass
    return out_dir / "stage2_fields" / "model_sn_ocr.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a fresh run_all.py pass against the frozen 115-image "
            "baseline (validation/baseline/README.md)."
        ),
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input image dir (default: {DEFAULT_INPUT})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output dir for the fresh run (default: {DEFAULT_OUT})")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="Frozen baseline model_sn_ocr.jsonl")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip re-running the pipeline; just compare an existing --out directory",
    )
    parser.add_argument(
        "--max-source-drift",
        type=int,
        default=DEFAULT_MAX_SOURCE_DRIFT,
        help=f"Max rows allowed to flip barcode<->OCR source family with an unchanged value (default: {DEFAULT_MAX_SOURCE_DRIFT})",
    )
    parser.add_argument("--log-level", default="info", choices=["info", "debug"])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input)
    out_dir = Path(args.out)
    baseline_path = Path(args.baseline)

    if not baseline_path.exists():
        print(f"Error: baseline file not found: {baseline_path}", file=sys.stderr)
        return 2

    if not args.skip_run:
        if not input_dir.is_dir():
            print(
                f"Error: input directory not found: {input_dir}\n"
                "       This is the gitignored 115-image baseline set -- see "
                "validation/baseline/README.md for how to materialize it.",
                file=sys.stderr,
            )
            return 2
        print(
            f"Running pipeline: {input_dir} -> {out_dir} "
            "(CROP_WORKERS/SCAN2_WORKERS throttled to 2 by default)"
        )
        t0 = time.perf_counter()
        proc = run_pipeline(input_dir, out_dir, log_level=args.log_level)
        elapsed = time.perf_counter() - t0
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"Error: run_all.py exited {proc.returncode} after {elapsed:.1f}s", file=sys.stderr)
            return 1
        print(f"Pipeline finished in {elapsed:.1f}s")

    fresh_jsonl = locate_result_jsonl(out_dir)
    if not fresh_jsonl.exists():
        print(f"Error: result JSONL not found: {fresh_jsonl}", file=sys.stderr)
        return 2

    fresh = load_results_by_label(fresh_jsonl)
    baseline = load_results_by_label(baseline_path)
    report = compare_results(fresh, baseline, max_source_drift=args.max_source_drift)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
