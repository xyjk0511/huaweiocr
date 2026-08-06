from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TRAINING_ROOT = Path(r"F:\HuaweiOCR\local_models\training")
QUEUE_LOG = TRAINING_ROOT / "label_detector_v4_training_queue.log"
RUN_NAMES = [
    "label_detector_v4s_yolov8s_1280",
    "label_detector_v4n_yolov8n_960",
    "label_detector_v4n_yolov8n_1280",
]


@dataclass
class RunSummary:
    name: str
    status: str
    last_epoch: int | None = None
    last_precision: float | None = None
    last_recall: float | None = None
    last_map50: float | None = None
    last_map5095: float | None = None
    best_epoch: int | None = None
    best_map5095: float | None = None
    results_mtime: str | None = None
    best_weight_mtime: str | None = None


def fmt_ts(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def read_queue_events() -> list[str]:
    if not QUEUE_LOG.exists():
        return []
    raw = QUEUE_LOG.read_bytes()
    for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            text = None
    if text is None:
        text = raw.decode("utf-8", errors="ignore")
    text = text.replace("\x00", "")
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def infer_status(name: str, events: list[str], run_dir: Path, results_path: Path) -> str:
    started = any(f"START {name}" in event for event in events)
    done = any(f"DONE {name}" in event for event in events)
    failed = any(f"FAIL {name}" in event for event in events)
    if failed:
        return "failed"
    if done:
        return "completed"
    if started:
        return "running"
    if run_dir.exists() or results_path.exists():
        return "prepared"
    return "pending"


def summarize_run(name: str, events: list[str]) -> RunSummary:
    run_dir = TRAINING_ROOT / name
    results_path = run_dir / "results.csv"
    summary = RunSummary(
        name=name,
        status=infer_status(name, events, run_dir, results_path),
        results_mtime=fmt_ts(results_path),
        best_weight_mtime=fmt_ts(run_dir / "weights" / "best.pt"),
    )
    if not results_path.exists():
        return summary

    with results_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return summary

    last = rows[-1]
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    summary.last_epoch = int(float(last["epoch"]))
    summary.last_precision = float(last["metrics/precision(B)"])
    summary.last_recall = float(last["metrics/recall(B)"])
    summary.last_map50 = float(last["metrics/mAP50(B)"])
    summary.last_map5095 = float(last["metrics/mAP50-95(B)"])
    summary.best_epoch = int(float(best["epoch"]))
    summary.best_map5095 = float(best["metrics/mAP50-95(B)"])
    return summary


def main() -> None:
    events = read_queue_events()
    print("Queue log:", QUEUE_LOG)
    if events:
        for event in events[-10:]:
            print("  ", event)
    else:
        print("   (no queue events found)")
    print()

    for name in RUN_NAMES:
        summary = summarize_run(name, events)
        print(f"[{summary.name}]")
        print("  status:", summary.status)
        if summary.last_epoch is not None:
            print("  last_epoch:", summary.last_epoch)
            print("  last_precision:", f"{summary.last_precision:.5f}")
            print("  last_recall:", f"{summary.last_recall:.5f}")
            print("  last_map50:", f"{summary.last_map50:.5f}")
            print("  last_map50_95:", f"{summary.last_map5095:.5f}")
            print("  best_epoch:", summary.best_epoch)
            print("  best_map50_95:", f"{summary.best_map5095:.5f}")
        print("  results_mtime:", summary.results_mtime)
        print("  best_weight_mtime:", summary.best_weight_mtime)
        print()


if __name__ == "__main__":
    main()
