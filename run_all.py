# run_all.py
import argparse
import os
import sys
import time
import traceback


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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.format.lower() != "jsonl":
        print("Only jsonl is supported for --format.", file=sys.stderr)
        return 2

    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    os.environ["LOG_LEVEL"] = args.log_level

    def _list_images(folder: str) -> list[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        try:
            return [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in exts
            ]
        except Exception:
            return []

    input_images = _list_images(args.input)
    total_images = len(input_images)

    try:
        import crop
        import scan2

        crop.set_log_level(args.log_level)
        scan2.set_log_level(args.log_level)

        print("===== [1/2] Crop labels & model/sn fields =====")
        t0 = time.perf_counter()
        crop.main(input_dir=args.input, out_dir=out_dir, log_level=args.log_level)
        t1 = time.perf_counter()

        print("\n===== [2/2] Barcode + OCR for MODEL/SN =====")
        t2 = time.perf_counter()
        stats = scan2.main(out_dir=out_dir, log_level=args.log_level)
        t3 = time.perf_counter()

        print("\nDone. Outputs:")
        print(f"  - {os.path.join(out_dir, 'stage1_labels')}")
        print(f"  - {os.path.join(out_dir, 'stage2_fields', 'model')}")
        print(f"  - {os.path.join(out_dir, 'stage2_fields', 'sn')}")
        print(f"  - {os.path.join(out_dir, 'model_sn_ocr.jsonl')}")
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
            print("  - error_distribution:")
            print(f"    - barcode_fail: {stats.get('barcode_fail', 0)}")
            print(f"    - ocr_fail: {stats.get('ocr_fail', 0)}")
            print(f"    - regex_fail: {stats.get('regex_fail', 0)}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.log_level == "debug":
            traceback.print_exc()
        return 1
    finally:
        if args.pause:
            input("\nPress Enter to exit...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
