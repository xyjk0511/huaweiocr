# run_all.py
import argparse
import json
import os
import sys
import time
import traceback

SUPPORTED_INPUT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: crop -> barcode -> OCR -> JSONL output",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="new_images",
        help="Input image directory (default: new_images)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=".",
        help="Output root directory (default: current directory)",
    )
    parser.add_argument(
        "--format",
        default="jsonl",
        help="Output format (jsonl only)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Label for timing output (cpu or gpu)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["info", "debug"],
        help="Log level (info or debug)",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause before exit (useful for double-click runs)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing generated output folders before running",
    )
    parser.add_argument(
        "--summary-json-out",
        default=None,
        help="Machine-readable run summary path (default: <out>/run_summary.json)",
    )
    parser.add_argument(
        "--excel-out",
        default=None,
        help="Optional Excel export path equivalent to the GUI result table",
    )
    return parser


def _list_images(folder: str) -> list[str]:
    images: list[str] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in SUPPORTED_INPUT_EXTS:
                images.append(os.path.join(dirpath, name))
    return sorted(images, key=lambda p: os.path.normcase(os.path.relpath(p, folder)))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _round_seconds(value):
    if value is None:
        return None
    return round(float(value), 3)


def _summary_path(out_dir: str, summary_json_out: str | None) -> str:
    if summary_json_out:
        return summary_json_out
    return os.path.join(out_dir or ".", "run_summary.json")


def _write_run_summary(path: str, summary: dict) -> str | None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    except Exception as exc:
        error = f"failed to write run summary: {path} ({exc})"
        print(f"Warning: {error}", file=sys.stderr)
        return error
    return None


def _export_jsonl_to_excel(jsonl_path: str, excel_path: str) -> dict:
    import openpyxl

    headers = ("label_id", "model", "sn", "model_src", "sn_src")
    os.makedirs(os.path.dirname(os.path.abspath(excel_path)) or ".", exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(list(headers))

    rows = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            ws.append([row.get(name, "") for name in headers])
            rows += 1

    wb.save(excel_path)
    return {"path": excel_path, "rows": rows}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started_at = time.perf_counter()
    out_dir = args.out
    summary_path = _summary_path(out_dir, args.summary_json_out)
    summary = {
        "schema_version": 1,
        "input_dir": args.input,
        "output_paths": {
            "out_dir": out_dir,
            "summary_json": summary_path,
        },
        "timing_sec": {
            "total": None,
            "crop": None,
            "scan": None,
        },
        "image_count": 0,
        "crop_stats": {},
        "scan2_stats": {},
        "exit_status": None,
        "status": "running",
    }

    def finish(exit_status: int, message: str = "", error: str = "") -> int:
        summary["exit_status"] = exit_status
        summary["status"] = "success" if exit_status == 0 else "failed"
        summary["timing_sec"]["total"] = _round_seconds(time.perf_counter() - started_at)
        if message:
            summary["message"] = message
        if error:
            summary["error"] = error
        summary_write_error = _write_run_summary(summary_path, summary)
        if summary_write_error and exit_status == 0:
            print(f"Error: {summary_write_error}; returning failure status.", file=sys.stderr)
            return 1
        return exit_status

    if args.format.lower() != "jsonl":
        message = "Only jsonl is supported for --format."
        print(message, file=sys.stderr)
        return finish(2, message=message)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    os.environ["LOG_LEVEL"] = args.log_level

    if not os.path.isdir(args.input):
        message = f"Input directory does not exist: {args.input}"
        print(message, file=sys.stderr)
        return finish(2, message=message)

    try:
        input_images = _list_images(args.input)
    except OSError as exc:
        message = f"Input directory is not readable: {args.input} ({exc})"
        print(message, file=sys.stderr)
        return finish(2, message=message)

    if not input_images:
        message = f"No supported images found in input directory: {args.input}"
        print(message, file=sys.stderr)
        return finish(2, message=message)

    total_images = len(input_images)
    summary["image_count"] = total_images

    try:
        import crop
        import scan2

        crop.set_log_level(args.log_level)
        scan2.set_log_level(args.log_level)

        print("===== [1/2] Crop labels & model/sn fields =====")
        t0 = time.perf_counter()
        crop_stats = crop.main(input_dir=args.input, out_dir=out_dir, log_level=args.log_level, clean=args.clean)
        t1 = time.perf_counter()
        summary["timing_sec"]["crop"] = _round_seconds(t1 - t0)
        crop_paths = crop_stats if isinstance(crop_stats, dict) else {}
        summary["crop_stats"] = crop_stats if isinstance(crop_stats, dict) else {"value": crop_stats}
        summary["output_paths"].update(
            {
                "stage1_dir": crop_paths.get("stage1_dir", ""),
                "stage2_dir": crop_paths.get("stage2_dir", ""),
                "model_dir": crop_paths.get("model_dir", ""),
                "sn_dir": crop_paths.get("sn_dir", ""),
            }
        )

        if not isinstance(crop_stats, dict) or crop_stats.get("label_count", 0) <= 0:
            message = "No label crops were generated; stopping before OCR."
            print(message, file=sys.stderr)
            return finish(1, message=message)
        if crop_stats.get("manifest_rows", 0) <= 0:
            message = "No manifest rows were generated; stopping before OCR."
            print(message, file=sys.stderr)
            return finish(1, message=message)

        print("\n===== [2/2] Barcode + OCR for MODEL/SN =====")
        t2 = time.perf_counter()
        result_jsonl = os.path.join(crop_paths["stage2_dir"], "model_sn_ocr.jsonl")
        debug_log = os.path.join(crop_paths["stage2_dir"], "debug_ocr_barcode.log")
        summary["output_paths"].update(
            {
                "result_jsonl": result_jsonl,
                "debug_log": debug_log,
            }
        )
        stats = scan2.main(
            model_dir=crop_paths["model_dir"],
            sn_dir=crop_paths["sn_dir"],
            out_jsonl=result_jsonl,
            debug_log=debug_log,
            log_level=args.log_level,
        )
        t3 = time.perf_counter()
        summary["timing_sec"]["scan"] = _round_seconds(t3 - t2)
        summary["scan2_stats"] = stats if isinstance(stats, dict) else {"value": stats}

        print("\nDone. Outputs:")
        print(f"  - {crop_paths['stage1_dir']}")
        print(f"  - {crop_paths['model_dir']}")
        print(f"  - {crop_paths['sn_dir']}")
        print(f"  - {result_jsonl}")
        if args.excel_out:
            export_stats = _export_jsonl_to_excel(result_jsonl, args.excel_out)
            summary["output_paths"]["excel"] = args.excel_out
            summary["excel_export"] = export_stats
            print(f"  - {args.excel_out}")
        total_time = (t3 - t0)
        if total_images > 0:
            avg_time = total_time / total_images
            print(f"\nStats (mode={args.device}):")
            print(f"  - images: {total_images}")
            print(f"  - time_total_sec: {total_time:.2f}")
            print(f"  - time_avg_sec: {avg_time:.2f}")
            print(f"  - time_crop_sec: {t1 - t0:.2f}")
            print(f"  - time_scan_sec: {t3 - t2:.2f}")
        if isinstance(stats, dict) and stats.get("sn_total"):
            model_total = stats.get("model_total", 0)
            if model_total:
                model_success = stats.get("model_success", 0)
                model_success_rate = model_success / model_total
                print("\nModel Metrics:")
                print(f"  - model_total: {model_total}")
                print(f"  - model_success: {model_success}")
                print(f"  - model_success_rate: {model_success_rate:.3f}")
                print(f"  - model_barcode_hits: {stats.get('model_barcode_hits', 0)}")
                print(f"  - model_barcode_hit_rate: {stats.get('model_barcode_hit_rate', 0.0):.3f}")
                print(f"  - model_ocr_recoveries: {stats.get('model_ocr_recoveries', 0)}")
            sn_total = stats.get("sn_total", 0)
            sn_success = stats.get("sn_success", 0)
            sn_attempted = stats.get("sn_attempted", 0)
            success_rate = (sn_success / sn_total) if sn_total else 0.0
            regex_rate = (sn_success / sn_attempted) if sn_attempted else 0.0
            print("\nSN Metrics:")
            print(f"  - sn_total: {sn_total}")
            print(f"  - sn_success: {sn_success}")
            print(f"  - sn_success_rate: {success_rate:.3f}")
            print(f"  - sn_regex_pass_rate: {regex_rate:.3f}")
            print(f"  - sn_barcode_attempts: {stats.get('sn_barcode_attempts', 0)}")
            print(f"  - sn_barcode_hits: {stats.get('sn_barcode_hits', 0)}")
            print(f"  - sn_barcode_hit_rate: {stats.get('sn_barcode_hit_rate', 0.0):.3f}")
            print(f"  - sn_ocr_recoveries: {stats.get('sn_ocr_recoveries', 0)}")
            print("  - error_distribution:")
            print(f"    - barcode_fail: {stats.get('barcode_fail', 0)}")
            print(f"    - ocr_fail: {stats.get('ocr_fail', 0)}")
            print(f"    - regex_fail: {stats.get('regex_fail', 0)}")
            print(f"    - barcode_parse_failures: {stats.get('sn_barcode_parse_failures', 0)}")
            print(f"    - barcode_decoder_misses: {stats.get('sn_barcode_decoder_misses', 0)}")
            print(f"    - barcode_ambiguous: {stats.get('sn_barcode_ambiguous', 0)}")
            print(f"    - barcode_quality_rejects: {stats.get('sn_barcode_quality_rejects', 0)}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.log_level == "debug":
            traceback.print_exc()
        return finish(1, error=f"{exc.__class__.__name__}: {exc}")
    finally:
        if args.pause:
            input("\nPress Enter to exit...")

    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
