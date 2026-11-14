"""Command-line entry point for cropping, recognition and result export."""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from huaweiocr.io.images import SUPPORTED_IMAGE_EXTS, list_images
from huaweiocr.io.results import RESULT_COLUMNS, export_excel, iter_jsonl

SUPPORTED_INPUT_EXTS = set(SUPPORTED_IMAGE_EXTS)


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
    return list_images(folder, extensions=SUPPORTED_INPUT_EXTS)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    rows = ([row.get(name, "") for name in RESULT_COLUMNS] for row in iter_jsonl(jsonl_path))
    return export_excel(excel_path, rows)


def _print_recognition_metrics(stats):
    if not isinstance(stats, dict):
        return
    for field, title in (("model", "Model"), ("sn", "SN")):
        total = stats.get(f"{field}_total", 0)
        if not total:
            continue
        success = stats.get(f"{field}_success", 0)
        print(f"\n{title} Metrics:")
        print(f"  - {field}_total: {total}")
        print(f"  - {field}_success: {success}")
        print(f"  - {field}_success_rate: {success / total:.3f}")
        if field == "sn":
            attempted = stats.get("sn_attempted", 0)
            regex_rate = success / attempted if attempted else 0.0
            print(f"  - sn_regex_pass_rate: {regex_rate:.3f}")
            print(f"  - sn_barcode_attempts: {stats.get('sn_barcode_attempts', 0)}")
        print(f"  - {field}_barcode_hits: {stats.get(f'{field}_barcode_hits', 0)}")
        print(f"  - {field}_barcode_hit_rate: {stats.get(f'{field}_barcode_hit_rate', 0.0):.3f}")
        print(f"  - {field}_ocr_recoveries: {stats.get(f'{field}_ocr_recoveries', 0)}")
        if field == "sn":
            print("  - error_distribution:")
            for name in ("barcode_fail", "ocr_fail", "regex_fail", "sn_barcode_parse_failures",
                         "sn_barcode_decoder_misses", "sn_barcode_ambiguous", "sn_barcode_quality_rejects",
                         "sn_barcode_part_no_conflicts"):
                print(f"    - {name.removeprefix('sn_')}: {stats.get(name, 0)}")


def _start_ocr_prewarm(scan2):
    """Load the OCR engine while Stage1/Stage2 cropping runs.

    Importing PaddleOCR takes tens of seconds; scan2 waits on the same engine
    lock, so overlapping it with cropping only removes idle time.
    """
    fallback_enabled = getattr(scan2, "scan_ocr_fallback_enabled", None)
    prewarm = getattr(scan2, "prewarm_ocr_engine", None)
    if not callable(fallback_enabled) or not callable(prewarm) or not fallback_enabled():
        return None
    # A thread-local engine built here would be invisible to scan workers.
    if getattr(scan2, "OCR_THREAD_LOCAL_MODE", False):
        return None
    from huaweiocr.io.envutil import env_flag_default

    if not env_flag_default("HUAWEIOCR_PREWARM_OCR", True):
        return None
    thread = threading.Thread(
        target=prewarm,
        name="HuaweiOCR-OCR-Prewarm",
        daemon=True,
    )
    thread.start()
    return thread


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started_at = time.perf_counter()
    out_dir = args.out
    summary_path = args.summary_json_out or os.path.join(out_dir or ".", "run_summary.json")
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
        summary["timing_sec"]["total"] = round(time.perf_counter() - started_at, 3)
        if message:
            print(message, file=sys.stderr)
            summary["message"] = message
        if error:
            summary["error"] = error
        summary_write_error = _write_run_summary(summary_path, summary)
        if summary_write_error and exit_status == 0:
            print(f"Error: {summary_write_error}; returning failure status.", file=sys.stderr)
            return 1
        return exit_status

    if args.format.lower() != "jsonl":
        return finish(2, message="Only jsonl is supported for --format.")

    os.environ["LOG_LEVEL"] = args.log_level

    if not os.path.isdir(args.input):
        return finish(2, message=f"Input directory does not exist: {args.input}")

    try:
        total_images = len(_list_images(args.input))
    except OSError as exc:
        return finish(2, message=f"Input directory is not readable: {args.input} ({exc})")

    if not total_images:
        return finish(2, message=f"No supported images found in input directory: {args.input}")

    summary["image_count"] = total_images

    prewarm_thread = None
    try:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        import crop
        import scan2

        crop.set_log_level(args.log_level)
        scan2.set_log_level(args.log_level)
        prewarm_thread = _start_ocr_prewarm(scan2)

        print("===== [1/2] Crop labels & model/sn fields =====")
        t0 = time.perf_counter()
        crop_stats = crop.main(input_dir=args.input, out_dir=out_dir, log_level=args.log_level, clean=args.clean)
        t1 = time.perf_counter()
        summary["timing_sec"]["crop"] = round(t1 - t0, 3)
        if not isinstance(crop_stats, dict):
            crop_stats = {"value": crop_stats}
        summary["crop_stats"] = crop_stats
        summary["output_paths"].update(
            (name, crop_stats.get(name, ""))
            for name in ("stage1_dir", "stage2_dir", "model_dir", "sn_dir")
        )

        if crop_stats.get("label_count", 0) <= 0:
            return finish(1, message="No label crops were generated; stopping before OCR.")
        if crop_stats.get("manifest_rows", 0) <= 0:
            return finish(1, message="No manifest rows were generated; stopping before OCR.")

        print("\n===== [2/2] Barcode + OCR for MODEL/SN =====")
        t2 = time.perf_counter()
        result_jsonl = os.path.join(crop_stats["stage2_dir"], "model_sn_ocr.jsonl")
        debug_log = os.path.join(crop_stats["stage2_dir"], "debug_ocr_barcode.log")
        summary["output_paths"].update(
            {
                "result_jsonl": result_jsonl,
                "debug_log": debug_log,
            }
        )
        stats = scan2.main(
            model_dir=crop_stats["model_dir"],
            sn_dir=crop_stats["sn_dir"],
            out_jsonl=result_jsonl,
            debug_log=debug_log,
            log_level=args.log_level,
        )
        t3 = time.perf_counter()
        summary["timing_sec"]["scan"] = round(t3 - t2, 3)
        summary["scan2_stats"] = stats if isinstance(stats, dict) else {"value": stats}

        print("\nDone. Outputs:")
        for name in ("stage1_dir", "model_dir", "sn_dir"):
            print(f"  - {crop_stats[name]}")
        print(f"  - {result_jsonl}")
        if args.excel_out:
            export_stats = _export_jsonl_to_excel(result_jsonl, args.excel_out)
            summary["output_paths"]["excel"] = args.excel_out
            summary["excel_export"] = export_stats
            print(f"  - {args.excel_out}")
        total_time = t3 - t0
        print(f"\nStats (mode={args.device}):")
        print(f"  - images: {total_images}")
        print(f"  - time_total_sec: {total_time:.2f}")
        print(f"  - time_avg_sec: {total_time / total_images:.2f}")
        print(f"  - time_crop_sec: {t1 - t0:.2f}")
        print(f"  - time_scan_sec: {t3 - t2:.2f}")
        _print_recognition_metrics(stats)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.log_level == "debug":
            traceback.print_exc()
        return finish(1, error=f"{exc.__class__.__name__}: {exc}")
    finally:
        # Never let the interpreter shut down while the daemon thread is still
        # inside Paddle's native initialization (all-barcode batches can finish
        # before the prewarm does).
        if prewarm_thread is not None:
            prewarm_thread.join()
        if args.pause:
            input("\nPress Enter to exit...")

    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
