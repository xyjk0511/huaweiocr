# run_all.py
import argparse
import os
import sys
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

    try:
        import crop
        import scan2

        crop.set_log_level(args.log_level)
        scan2.set_log_level(args.log_level)

        print("===== [1/2] Crop labels & model/sn fields =====")
        crop.main(input_dir=args.input, out_dir=out_dir, log_level=args.log_level)

        print("\n===== [2/2] Barcode + OCR for MODEL/SN =====")
        scan2.main(out_dir=out_dir, log_level=args.log_level)

        print("\nDone. Outputs:")
        print(f"  - {os.path.join(out_dir, 'stage1_labels')}")
        print(f"  - {os.path.join(out_dir, 'stage2_fields', 'model')}")
        print(f"  - {os.path.join(out_dir, 'stage2_fields', 'sn')}")
        print(f"  - {os.path.join(out_dir, 'model_sn_ocr.jsonl')}")
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
