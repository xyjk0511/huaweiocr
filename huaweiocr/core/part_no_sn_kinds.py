"""Which SN family each PartNo uses, learned from genuine barcode decodes.

Huawei 20-char SNs embed the label PartNo ("21" + PartNo + ...), while some
switch series print 12-char "4E" SNs instead.  Code128 pixel repair can turn
an unreadable SN barcode into a structurally valid value of the wrong family,
so recognition cross-checks repaired SNs against the label PartNo and against
the families seen for that PartNo in real (non-repaired) decodes.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any, Iterable

try:
    from app_paths import get_user_data_dir
except ImportError:
    from huaweiocr.io.paths_runtime import get_user_data_dir

from huaweiocr.core.extract import is_generic_part_no


SN_KIND_PART_NO = "21"
SN_KIND_4E = "4E"
_KINDS = (SN_KIND_PART_NO, SN_KIND_4E)
_MAX_PART_NOS = 4096
_LOCK = threading.RLock()


def sn_embedded_part_no(sn: str) -> str:
    value = str(sn or "").strip().upper()
    if len(value) == 20 and value.startswith("21") and value[2:10].isdigit():
        return value[2:10]
    return ""


def sn_kind(sn: str) -> str:
    if sn_embedded_part_no(sn):
        return SN_KIND_PART_NO
    if str(sn or "").strip().upper().startswith("4E"):
        return SN_KIND_4E
    return ""


def genuine_sn_kind_for_part_no(sn: str, part_no: str) -> str:
    """Kind evidence from a genuine decode, or "" when it is not usable."""
    if not is_generic_part_no(part_no):
        return ""
    kind = sn_kind(sn)
    if kind == SN_KIND_PART_NO and sn_embedded_part_no(sn) != part_no:
        return ""
    return kind


def repaired_sn_conflict(sn: str, part_no: str, kinds: Iterable[str]) -> str:
    """Why a pixel-repaired SN cannot belong to this label, or ""."""
    if not is_generic_part_no(part_no):
        return ""
    embedded = sn_embedded_part_no(sn)
    if embedded:
        return "part_no_mismatch" if embedded != part_no else ""
    if sn_kind(sn) == SN_KIND_4E and set(kinds or ()) == {SN_KIND_PART_NO}:
        return "sn_family_conflict"
    return ""


def part_no_sn_kinds_path() -> str:
    override = os.environ.get("SCAN2_PART_NO_SN_KINDS_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(get_user_data_dir(), "part_no_sn_kinds.json")


def _read_state(path: str, *, strict: bool = False) -> dict[str, dict[str, int]]:
    """Load the store; ``strict`` raises OSError when an existing file is unusable.

    Writers use ``strict`` so a corrupt or locked file is left alone instead of
    being replaced by the current batch's counts only.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        if strict:
            raise OSError(f"unreadable learning store: {path}") from exc
        return {}
    raw = data.get("part_no_sn_kinds") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        if strict:
            raise OSError(f"unexpected learning store layout: {path}")
        return {}
    out: dict[str, dict[str, int]] = {}
    for part_no, counts in raw.items():
        if not is_generic_part_no(str(part_no)) or not isinstance(counts, dict):
            continue
        clean = {}
        for kind in _KINDS:
            try:
                count = int(counts.get(kind, 0) or 0)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                clean[kind] = count
        if clean:
            out[str(part_no)] = clean
    return out


def load_part_no_sn_kinds() -> dict[str, set[str]]:
    with _LOCK:
        state = _read_state(part_no_sn_kinds_path())
    return {part_no: set(counts) for part_no, counts in state.items()}


def _atomic_write(path: str, data: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".part_no_sn_kinds.", suffix=".json", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def record_part_no_sn_kinds(observations: Iterable[tuple[str, str]]) -> int:
    """Persist (part_no, kind) evidence from genuine decodes; returns count."""
    pairs = [
        (str(part_no), kind)
        for part_no, kind in observations or ()
        if kind in _KINDS and is_generic_part_no(str(part_no))
    ]
    if not pairs:
        return 0
    path = part_no_sn_kinds_path()
    try:
        with _LOCK:
            state = _read_state(path, strict=True)
            for part_no, kind in pairs:
                counts = state.setdefault(part_no, {})
                counts[kind] = counts.get(kind, 0) + 1
            if len(state) > _MAX_PART_NOS:
                keep = sorted(state, key=lambda key: -sum(state[key].values()))[:_MAX_PART_NOS]
                state = {key: state[key] for key in keep}
            _atomic_write(path, {"schema_version": 1, "part_no_sn_kinds": state})
    except OSError:
        return 0
    return len(pairs)
