from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from sn_barcode import extract_sn_from_payload, scan_sn_barcodes


DEFAULT_MANIFEST = os.path.join("validation", "sn_barcode_manifest.jsonl")
DEFAULT_THRESHOLD = 0.90
DEFAULT_MIN_ACCEPTED = 50
DEFAULT_TEMPLATE_NOTE = (
    "TODO manually verify expected_sn, barcode_present, and accepted_quality; "
    "pipeline/OCR candidates are hints only and are not ground truth."
)


@dataclass
class ValidationSummary:
    manifest_path: str
    threshold: float
    min_accepted: int
    total_rows: int = 0
    accepted_quality_rows: int = 0
    accepted_barcode_rows: int = 0
    barcode_present_rows: int = 0
    denominator: int = 0
    exact_hits: int = 0
    hit_rate: float = 0.0
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    failure_counts: dict[str, int] = field(default_factory=lambda: {
        "decoder_miss": 0,
        "parse_failure": 0,
        "ambiguous": 0,
        "quality_reject": 0,
        "wrong_sn": 0,
        "missing_expected_sn": 0,
        "missing_label_local_source": 0,
        "not_barcode_present": 0,
    })
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "threshold": self.threshold,
            "min_accepted": self.min_accepted,
            "total_rows": self.total_rows,
            "accepted_quality_rows": self.accepted_quality_rows,
            "accepted_barcode_rows": self.accepted_barcode_rows,
            "barcode_present_rows": self.barcode_present_rows,
            "denominator": self.denominator,
            "exact_hits": self.exact_hits,
            "hit_rate": self.hit_rate,
            "passed": self.passed,
            "errors": self.errors,
            "failure_counts": self.failure_counts,
            "rows": self.rows,
        }


def _resolve_path(base_dir: str, value: str) -> str:
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _resolve_existing_path(primary_base: str, value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    if value.lower() in {"none", "null"}:
        return ""
    if os.path.isabs(value):
        return os.path.normpath(value)

    candidates = [
        os.path.abspath(value),
        os.path.abspath(os.path.join(primary_base, value)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    return os.path.normpath(candidates[0])


def _manifest_relative_path(output_dir: str, value: str) -> str:
    if not value:
        return ""
    try:
        return os.path.normpath(os.path.relpath(value, output_dir))
    except ValueError:
        return os.path.normpath(value)


def _load_manifest(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            item["_line_no"] = line_no
            rows.append(item)
    return rows


def _load_rows_by_label(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = _load_manifest(path)
    by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        label_id = str(row.get("label_id", "")).strip()
        if label_id and label_id not in by_label:
            by_label[label_id] = row
    return by_label


def _validate_schema(row: dict[str, Any], manifest_dir: str) -> list[str]:
    errors: list[str] = []
    line = row.get("_line_no", "?")
    required = ["label_id", "expected_sn", "barcode_present", "accepted_quality"]
    for key in required:
        if key not in row:
            errors.append(f"line {line}: missing required field {key}")
    for key in ("barcode_present", "accepted_quality"):
        if key in row and not isinstance(row[key], bool):
            errors.append(f"line {line}: {key} must be boolean")
    if row.get("expected_sn"):
        expected = extract_sn_from_payload(str(row.get("expected_sn")))
        if not expected:
            errors.append(f"line {line}: expected_sn does not parse as a valid SN")
    for key in ("label_crop", "sn_path"):
        value = row.get(key)
        if value:
            resolved = _resolve_path(manifest_dir, str(value))
            if not os.path.isfile(resolved):
                errors.append(f"line {line}: {key} does not exist: {value}")
    return errors


def _sources_for_row(row: dict[str, Any], manifest_dir: str) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    if row.get("sn_path"):
        sources.append(("sn", _resolve_path(manifest_dir, str(row["sn_path"]))))
    if row.get("label_crop"):
        sources.append(("label", _resolve_path(manifest_dir, str(row["label_crop"]))))
    return sources


def build_manifest_template_from_stage2(
    stage2_manifest_path: str,
    output_path: str,
    *,
    recognized_jsonl: str = "",
) -> dict[str, Any]:
    stage2_manifest_path = os.path.abspath(stage2_manifest_path)
    output_path = os.path.abspath(output_path)
    stage2_dir = os.path.dirname(stage2_manifest_path)
    output_dir = os.path.dirname(output_path) or "."

    rows = _load_manifest(stage2_manifest_path)
    recognized_by_label = _load_rows_by_label(recognized_jsonl) if recognized_jsonl else {}

    os.makedirs(output_dir, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            label_id = str(row.get("label_id", "")).strip()
            if not label_id:
                raise ValueError(f"{stage2_manifest_path}:{row.get('_line_no', '?')}: missing label_id")

            label_crop = _resolve_existing_path(stage2_dir, str(row.get("label_crop", "")))
            sn_path = _resolve_existing_path(stage2_dir, str(row.get("sn_path", "")))
            original_image = _resolve_existing_path(stage2_dir, str(row.get("original_image_path", "")))
            image_path = _resolve_existing_path(
                stage2_dir,
                str(row.get("image_path") or row.get("original_image_path") or row.get("label_crop") or row.get("sn_path") or ""),
            )

            template_row: dict[str, Any] = {
                "image_path": _manifest_relative_path(output_dir, image_path),
                "label_id": label_id,
                "expected_sn": "",
                "barcode_present": False,
                "accepted_quality": False,
                "notes": DEFAULT_TEMPLATE_NOTE,
            }
            if label_crop:
                template_row["label_crop"] = _manifest_relative_path(output_dir, label_crop)
            if sn_path:
                template_row["sn_path"] = _manifest_relative_path(output_dir, sn_path)
            if original_image:
                template_row["original_image_path"] = _manifest_relative_path(output_dir, original_image)
            if row.get("model_conf") is not None:
                template_row["model_conf"] = row.get("model_conf")
            if row.get("sn_conf") is not None:
                template_row["sn_conf"] = row.get("sn_conf")

            recognized = recognized_by_label.get(label_id)
            if recognized:
                candidate_sn = extract_sn_from_payload(str(recognized.get("sn", "")))
                if candidate_sn:
                    template_row["pipeline_candidate_sn"] = candidate_sn
                    template_row["pipeline_candidate_source"] = str(recognized.get("sn_src", ""))

            out.write(json.dumps(template_row, ensure_ascii=False) + "\n")
            written += 1

    return {
        "stage2_manifest_path": stage2_manifest_path,
        "output_path": output_path,
        "rows_written": written,
        "recognized_rows_loaded": len(recognized_by_label),
        "accepted_quality_rows": 0,
        "release_gate_ready": False,
    }


def evaluate_manifest(
    manifest_path: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_accepted: int = DEFAULT_MIN_ACCEPTED,
    debug_dir: str = "",
) -> ValidationSummary:
    manifest_path = os.path.abspath(manifest_path)
    manifest_dir = os.path.dirname(manifest_path)
    summary = ValidationSummary(
        manifest_path=manifest_path,
        threshold=threshold,
        min_accepted=min_accepted,
    )

    if not os.path.isfile(manifest_path):
        summary.errors.append(f"manifest not found: {manifest_path}")
        return summary

    try:
        rows = _load_manifest(manifest_path)
    except Exception as exc:
        summary.errors.append(str(exc))
        return summary

    summary.total_rows = len(rows)
    for row in rows:
        schema_errors = _validate_schema(row, manifest_dir)
        if schema_errors:
            summary.errors.extend(schema_errors)
            continue

        barcode_present = bool(row["barcode_present"])
        accepted_quality = bool(row["accepted_quality"])
        expected_sn = extract_sn_from_payload(str(row.get("expected_sn", "")))
        sources = _sources_for_row(row, manifest_dir)
        if barcode_present:
            summary.barcode_present_rows += 1
        else:
            summary.failure_counts["not_barcode_present"] += 1
        if accepted_quality:
            summary.accepted_quality_rows += 1
        if accepted_quality and barcode_present and expected_sn and sources:
            summary.accepted_barcode_rows += 1

        row_result = {
            "label_id": row.get("label_id", ""),
            "accepted_quality": accepted_quality,
            "barcode_present": barcode_present,
            "status": "skipped",
            "expected_sn": expected_sn,
            "actual_sn": "",
        }

        if not barcode_present:
            summary.rows.append(row_result)
            continue
        if not accepted_quality:
            row_result["status"] = "quality_reject"
            summary.failure_counts["quality_reject"] += 1
            summary.rows.append(row_result)
            continue
        if not expected_sn:
            row_result["status"] = "missing_expected_sn"
            summary.failure_counts["missing_expected_sn"] += 1
            summary.rows.append(row_result)
            continue
        if not sources:
            row_result["status"] = "missing_label_local_source"
            summary.failure_counts["missing_label_local_source"] += 1
            summary.errors.append(
                f"line {row.get('_line_no', '?')}: accepted barcode row requires label-local source "
                "(sn_path or label_crop); image_path/original_image_path are provenance only"
            )
            summary.rows.append(row_result)
            continue

        summary.denominator += 1
        report = scan_sn_barcodes(
            sources,
            label_id=str(row.get("label_id", "")),
            debug_dir=debug_dir,
            early_exit=True,
        )
        row_result["status"] = report.status
        row_result["actual_sn"] = report.sn
        row_result["barcode_attempts"] = report.attempts
        row_result["decoded_count"] = report.decoded_count

        if report.status == "hit" and report.sn == expected_sn:
            summary.exact_hits += 1
            row_result["status"] = "hit"
        elif report.status == "hit":
            summary.failure_counts["wrong_sn"] += 1
            row_result["status"] = "wrong_sn"
        elif report.status in summary.failure_counts:
            summary.failure_counts[report.status] += 1
        else:
            summary.failure_counts["decoder_miss"] += 1

        summary.rows.append(row_result)

    if summary.denominator:
        summary.hit_rate = summary.exact_hits / float(summary.denominator)
    if summary.accepted_barcode_rows < min_accepted:
        summary.errors.append(
            f"accepted-quality barcode sample count {summary.accepted_barcode_rows} is below required minimum {min_accepted}"
        )
    if summary.hit_rate < threshold:
        summary.errors.append(
            f"exact barcode-derived SN hit rate {summary.hit_rate:.3f} is below threshold {threshold:.3f}"
        )
    summary.passed = not summary.errors and summary.hit_rate >= threshold
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate exact SN barcode hit rate from a ground-truth JSONL manifest.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-accepted", type=int, default=DEFAULT_MIN_ACCEPTED)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--debug-candidates-dir", default="")
    parser.add_argument("--init-template-from-stage2", default="")
    parser.add_argument("--template-out", default="")
    parser.add_argument("--recognized-jsonl", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.init_template_from_stage2:
        if not args.template_out:
            parser.error("--template-out is required with --init-template-from-stage2")
        data = build_manifest_template_from_stage2(
            args.init_template_from_stage2,
            args.template_out,
            recognized_jsonl=args.recognized_jsonl,
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    summary = evaluate_manifest(
        args.manifest,
        threshold=args.threshold,
        min_accepted=args.min_accepted,
        debug_dir=args.debug_candidates_dir,
    )
    data = summary.to_dict()
    output = json.dumps(data, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(output + "\n")
    return 0 if summary.passed else 1


if __name__ == "__main__":
    sys.exit(main())
