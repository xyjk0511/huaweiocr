from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from typing import Any

try:
    from app_paths import get_user_data_dir
except ImportError:
    from huaweiocr.io.paths_runtime import get_user_data_dir


_LOCK = threading.RLock()
_TTL_SECONDS = 30 * 86400
_MAX_CANDIDATES = 128
_MAX_EVIDENCE = 16
_PART_NO_RE = re.compile(r"(?:500|980)\d{5}")
_MODEL_RE = re.compile(r"[A-Z][A-Z0-9-]{1,23}")


def model_learning_candidates_path() -> str:
    override = os.environ.get("SCAN2_MODEL_LEARNING_CANDIDATES_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(get_user_data_dir(), "model_learning_candidates.json")


def _read_state(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {"schema_version": 1, "candidates": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("candidates"), dict):
            return data
    except Exception:
        pass
    return {"schema_version": 1, "candidates": {}}


def _atomic_write(path: str, data: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".model_learning.", suffix=".json", dir=directory, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _source_key(original_image_path: Any, label_id: str) -> str:
    if isinstance(original_image_path, str) and original_image_path.strip():
        return os.path.normcase(os.path.abspath(original_image_path.strip())).replace("\\", "/")
    return re.sub(r"__label_\d+$", "", label_id.strip()) or "unknown_source"


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values and len(values) < _MAX_EVIDENCE:
        values.append(value)


def commit_model_observations(
    observations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Persist strong PartNo+MODEL observations and return newly promoted pairs."""
    result: dict[str, list[dict[str, Any]]] = {
        "promoted": [],
        "pending": [],
        "rejected": [],
        "conflicts": [],
    }
    if not observations:
        return result

    now = time.time()
    path = model_learning_candidates_path()
    with _LOCK:
        state = _read_state(path)
        raw_candidates = state.get("candidates", {})
        candidates = {
            key: value
            for key, value in raw_candidates.items()
            if isinstance(value, dict)
            and now - float(value.get("updated_at", now)) <= _TTL_SECONDS
        }

        for observation in observations:
            if not isinstance(observation, dict):
                result["rejected"].append({"reason": "invalid_observation"})
                continue
            part_no = re.sub(r"\D", "", str(observation.get("part_no", "")))
            model = re.sub(r"[^0-9A-Z-]", "", str(observation.get("model", "")).upper())
            label_id = str(observation.get("label_id", "") or "").strip()
            if not _PART_NO_RE.fullmatch(part_no) or not _MODEL_RE.fullmatch(model) or not re.search(r"\d", model):
                result["rejected"].append(
                    {"part_no": part_no, "model": model, "reason": "invalid_pair"}
                )
                continue

            key = f"{part_no}|{model}"
            candidate = candidates.setdefault(
                key,
                {
                    "part_no": part_no,
                    "model": model,
                    "created_at": now,
                    "updated_at": now,
                    "label_ids": [],
                    "source_keys": [],
                },
            )
            candidate["updated_at"] = now
            _append_unique(candidate.setdefault("label_ids", []), label_id)
            _append_unique(
                candidate.setdefault("source_keys", []),
                _source_key(observation.get("original_image_path"), label_id),
            )

        models_by_part_no: dict[str, set[str]] = {}
        for candidate in candidates.values():
            models_by_part_no.setdefault(candidate.get("part_no", ""), set()).add(
                candidate.get("model", "")
            )

        promoted_keys = []
        for key, candidate in sorted(candidates.items()):
            part_no = candidate.get("part_no", "")
            model = candidate.get("model", "")
            models = {value for value in models_by_part_no.get(part_no, set()) if value}
            if len(models) > 1:
                result["conflicts"].append(
                    {"part_no": part_no, "models": sorted(models)}
                )
                result["pending"].append(dict(candidate))
                continue
            if (
                len(set(candidate.get("label_ids", []))) >= 2
                and len(set(candidate.get("source_keys", []))) >= 2
            ):
                result["promoted"].append({"part_no": part_no, "model": model})
                promoted_keys.append(key)
            else:
                result["pending"].append(dict(candidate))

        for key in promoted_keys:
            candidates.pop(key, None)
        if len(candidates) > _MAX_CANDIDATES:
            keep = sorted(
                candidates,
                key=lambda key: float(candidates[key].get("updated_at", 0)),
                reverse=True,
            )[:_MAX_CANDIDATES]
            candidates = {key: candidates[key] for key in keep}

        _atomic_write(path, {"schema_version": 1, "candidates": candidates})
    return result
