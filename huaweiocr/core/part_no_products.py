"""EAN-13/UPC-A product codes seen per PartNo, learned from labels.

Each Huawei PartNo prints one EAN-13 (and on APs one UPC-A).  Once a code has
been seen repeatedly with one PartNo, a label pairing it with a different
PartNo points at a wrong PartNo read or a crop that mixed two labels.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Iterable

try:
    from app_paths import get_user_data_dir
except ImportError:
    from huaweiocr.io.paths_runtime import get_user_data_dir

from huaweiocr.core.extract import is_generic_part_no
from huaweiocr.core.part_no_sn_kinds import _atomic_write

_MIN_EVIDENCE = 2
_DOMINANCE = 3
_MAX_PART_NOS = 4096
_LOCK = threading.RLock()


def part_no_products_path() -> str:
    override = os.environ.get("SCAN2_PART_NO_PRODUCTS_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(get_user_data_dir(), "part_no_products.json")


def _valid_code(code: str) -> bool:
    return len(code) == 13 and code.isdigit()


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
    raw = data.get("part_no_products") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        if strict:
            raise OSError(f"unexpected learning store layout: {path}")
        return {}
    out: dict[str, dict[str, int]] = {}
    for part_no, counts in raw.items():
        if not is_generic_part_no(str(part_no)) or not isinstance(counts, dict):
            continue
        clean = {}
        for code, count in counts.items():
            try:
                count = int(count or 0)
            except (TypeError, ValueError):
                continue
            if _valid_code(str(code)) and count > 0:
                clean[str(code)] = count
        if clean:
            out[str(part_no)] = clean
    return out


def load_product_owners() -> dict[str, dict[str, int]]:
    """product code -> {part_no: times seen}."""
    with _LOCK:
        state = _read_state(part_no_products_path())
    owners: dict[str, dict[str, int]] = {}
    for part_no, counts in state.items():
        for code, count in counts.items():
            owners.setdefault(code, {})[part_no] = count
    return owners


def product_part_no_conflict(
    part_no: str, codes: Iterable[str], owners: dict, own_owners: dict | None = None,
) -> str:
    """The PartNo the label's product codes belong to, if not ``part_no``.

    Each code is judged on its own: it conflicts when exactly one other PartNo
    has seen it at least ``_MIN_EVIDENCE`` times and at least ``_DOMINANCE``
    times as often as ``part_no`` has, so one earlier wrong pairing does not
    exempt ``part_no`` forever.  Rival counts come from ``owners`` (which
    should exclude the label itself); ``part_no``'s own support comes from
    ``own_owners`` (default ``owners``), so callers can keep not-yet-verified
    pairings from vouching for each other.
    """
    if not is_generic_part_no(part_no):
        return ""
    if own_owners is None:
        own_owners = owners
    for code in codes or ():
        if not _valid_code(code):
            continue
        seen = owners.get(code) or {}
        own = (own_owners.get(code) or {}).get(part_no, 0)
        needed = max(_MIN_EVIDENCE, _DOMINANCE * own)
        rivals = [
            owner for owner, count in seen.items()
            if owner != part_no and count >= needed
        ]
        if len(rivals) == 1:
            return rivals[0]
    return ""


def record_part_no_products(observations: Iterable[tuple[str, str]]) -> int:
    """Persist (part_no, product code) pairs; returns how many were kept."""
    pairs = [
        (str(part_no), str(code))
        for part_no, code in observations or ()
        if is_generic_part_no(str(part_no)) and _valid_code(str(code))
    ]
    if not pairs:
        return 0
    path = part_no_products_path()
    try:
        with _LOCK:
            state = _read_state(path, strict=True)
            for part_no, code in pairs:
                counts = state.setdefault(part_no, {})
                counts[code] = counts.get(code, 0) + 1
            if len(state) > _MAX_PART_NOS:
                keep = sorted(state, key=lambda key: -sum(state[key].values()))[:_MAX_PART_NOS]
                state = {key: state[key] for key in keep}
            _atomic_write(path, {"schema_version": 1, "part_no_products": state})
    except OSError:
        return 0
    return len(pairs)
