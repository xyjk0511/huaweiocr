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

CANDIDATE_TTL_SECONDS = 30 * 86400
MAX_CANDIDATE_FAMILIES = 64
MAX_EVIDENCE_PER_CANDIDATE = 16

_MASK_CHAR_ALLOWED = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ#@*")
_LOCK = threading.RLock()
_ACTIVE_RULES_CACHE: list[dict[str, Any]] | None = None
_COMPILED_PATTERNS_CACHE: list[re.Pattern] | None = None


def learned_sn_families_path() -> str:
    """Return the filesystem path for active learned SN families."""
    override = os.environ.get("HUAWEIOCR_LEARNED_SN_FAMILIES_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(get_user_data_dir(), "learned_sn_families.json")


def sn_family_candidates_path() -> str:
    """Return the filesystem path for pending SN family candidates."""
    override = os.environ.get("HUAWEIOCR_SN_FAMILY_CANDIDATES_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(get_user_data_dir(), "sn_family_candidates.json")


def _quarantine_corrupt_file(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        if os.path.getsize(path) == 0:
            return
        timestamp = int(time.time())
        corrupt_path = f"{path}.corrupt.{timestamp}"
        count = 1
        while os.path.exists(corrupt_path):
            corrupt_path = f"{path}.corrupt.{timestamp}_{count}"
            count += 1
        os.replace(path, corrupt_path)
    except OSError:
        pass


def _atomic_write_json(path: str, data: Any) -> None:
    target_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_sn_", dir=target_dir, text=True)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        for attempt in range(3):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _safe_read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        size = os.path.getsize(path)
        if size == 0:
            return default
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return default
        return json.loads(content)
    except Exception:
        _quarantine_corrupt_file(path)
        return default


def _clean_sn_value(value: Any) -> str:
    if not value or not isinstance(value, str):
        return ""
    return re.sub(r"[^0-9A-Za-z]", "", value).upper()


def _compile_mask(mask: str) -> re.Pattern | None:
    if not mask or not isinstance(mask, str):
        return None
    cleaned_mask = mask.strip().upper()
    if not cleaned_mask or any(ch not in _MASK_CHAR_ALLOWED for ch in cleaned_mask):
        return None
    parts = []
    for ch in cleaned_mask:
        if ch == "#":
            parts.append(r"[0-9]")
        elif ch == "@":
            parts.append(r"[A-Z]")
        elif ch == "*":
            parts.append(r"[0-9A-Z]")
        else:
            parts.append(re.escape(ch))
    try:
        return re.compile(r"^" + "".join(parts) + r"$")
    except re.error:
        return None


def _validate_active_rule(rule: Any) -> dict[str, Any] | None:
    if not isinstance(rule, dict):
        return None
    rule_id = str(rule.get("id", "")).strip()
    mask = str(rule.get("mask", "")).strip().upper()
    if not rule_id or not mask:
        return None
    pattern = _compile_mask(mask)
    if pattern is None:
        return None

    total_len = rule.get("total_length", len(mask))
    if not isinstance(total_len, int) or total_len != len(mask):
        total_len = len(mask)

    validated: dict[str, Any] = {
        "id": rule_id,
        "mask": mask,
        "prefix_digits": int(rule["prefix_digits"]) if rule.get("prefix_digits") is not None else None,
        "marker": str(rule.get("marker", "")).strip().upper() or None,
        "suffix_kind": str(rule.get("suffix_kind", "")).strip().lower() or None,
        "suffix_length": int(rule["suffix_length"]) if rule.get("suffix_length") is not None else None,
        "total_length": total_len,
        "status": "active",
    }
    if "evidence" in rule and isinstance(rule["evidence"], dict):
        validated["evidence"] = dict(rule["evidence"])
    return validated


def reload_sn_rules() -> None:
    """Clear active cache and force reload on next access."""
    global _ACTIVE_RULES_CACHE, _COMPILED_PATTERNS_CACHE
    with _LOCK:
        _ACTIVE_RULES_CACHE = None
        _COMPILED_PATTERNS_CACHE = None


def get_active_sn_rules() -> list[dict[str, Any]]:
    """Return validated active learned SN rules, failing closed on corrupt/missing files."""
    global _ACTIVE_RULES_CACHE, _COMPILED_PATTERNS_CACHE
    with _LOCK:
        if _ACTIVE_RULES_CACHE is not None:
            return [dict(r) for r in _ACTIVE_RULES_CACHE]

        path = learned_sn_families_path()
        data = _safe_read_json(path, default={"schema_version": 1, "families": []})

        raw_list = []
        if isinstance(data, dict):
            raw_list = data.get("families", [])
        elif isinstance(data, list):
            raw_list = data

        valid_rules: list[dict[str, Any]] = []
        patterns: list[re.Pattern] = []
        seen_ids: set[str] = set()

        if isinstance(raw_list, list):
            for entry in raw_list:
                v = _validate_active_rule(entry)
                if v is not None and v["id"] not in seen_ids:
                    seen_ids.add(v["id"])
                    valid_rules.append(v)
                    pat = _compile_mask(v["mask"])
                    if pat is not None:
                        patterns.append(pat)

        _ACTIVE_RULES_CACHE = valid_rules
        _COMPILED_PATTERNS_CACHE = patterns
        return [dict(r) for r in _ACTIVE_RULES_CACHE]


def match_learned_sn(value: str) -> str:
    """Clean to uppercase alphanumeric and full-match any active declarative mask.

    Returns the cleaned value if matched, else empty string.
    """
    cleaned = _clean_sn_value(value)
    if not cleaned:
        return ""

    with _LOCK:
        if _COMPILED_PATTERNS_CACHE is None:
            get_active_sn_rules()
        patterns = list(_COMPILED_PATTERNS_CACHE or [])

    for pat in patterns:
        if pat.fullmatch(cleaned):
            return cleaned
    return ""


def infer_standard_sn_family(value: str) -> dict[str, Any] | None:
    """Infer standard Huawei 20-char SN family envelope.

    Deterministic envelope:
    - 20-char value shaped '215' + total 10/11 digit prefix + 2-4 uppercase marker + 6/7 uppercase-alphanumeric suffix.
    - Suffix kind inferred as 'digits' or 'alnum'.
    - Returns schema dict or None on nonstandard/ambiguous shape.
    """
    cleaned = _clean_sn_value(value)
    if len(cleaned) != 20 or not cleaned.startswith("215"):
        return None

    if not cleaned[:10].isdigit():
        return None

    if cleaned[10].isdigit():
        prefix_digits = 11
    elif cleaned[10].isalpha():
        prefix_digits = 10
    else:
        return None

    if not cleaned[:prefix_digits].isdigit():
        return None

    remainder = cleaned[prefix_digits:]
    if prefix_digits == 11:
        if not remainder[:2].isalpha():
            return None
        if remainder[2].isalpha():
            marker_len = 3
            suffix_len = 6
        else:
            marker_len = 2
            suffix_len = 7
    else:
        if not remainder[:3].isalpha():
            return None
        if remainder[3].isalpha():
            marker_len = 4
            suffix_len = 6
        else:
            marker_len = 3
            suffix_len = 7

    marker = remainder[:marker_len]
    suffix = remainder[marker_len:]

    # SS is deliberately prefix-bound in the built-in parser. A real baseline
    # image produced a structurally valid false SS decode for another prefix;
    # promoting a generic SS mask would re-accept that known false value.
    if marker == "SS":
        return None

    if len(suffix) != suffix_len or not suffix.isalnum():
        return None

    if suffix.isdigit():
        suffix_kind = "digits"
        suffix_mask = "#" * suffix_len
    else:
        suffix_kind = "alnum"
        suffix_mask = "*" * suffix_len

    prefix_mask = "215" + ("#" * (prefix_digits - 3))
    mask = prefix_mask + marker + suffix_mask
    family_id = f"p{prefix_digits}-{marker.lower()}-{'d' if suffix_kind == 'digits' else 'a'}{suffix_len}"

    return {
        "id": family_id,
        "mask": mask,
        "prefix_digits": prefix_digits,
        "marker": marker,
        "suffix_kind": suffix_kind,
        "suffix_length": suffix_len,
        "total_length": 20,
        "status": "active",
    }


def _derive_source_key(original_image_path: Any, label_id: str) -> str:
    if original_image_path and isinstance(original_image_path, str) and original_image_path.strip():
        norm = os.path.normcase(os.path.abspath(original_image_path.strip())).replace("\\", "/")
        return norm
    if label_id and isinstance(label_id, str) and label_id.strip():
        return re.sub(r"__label_\d+$", "", label_id.strip())
    return "unknown_source"


def _prune_and_cap_candidates(candidates: dict[str, dict[str, Any]], now: float) -> dict[str, dict[str, Any]]:
    active_cands = {
        cid: c
        for cid, c in candidates.items()
        if now - float(c.get("updated_at", c.get("created_at", now))) <= CANDIDATE_TTL_SECONDS
    }
    if len(active_cands) > MAX_CANDIDATE_FAMILIES:
        sorted_keys = sorted(
            active_cands.keys(),
            key=lambda k: float(active_cands[k].get("updated_at", active_cands[k].get("created_at", 0))),
            reverse=True,
        )
        active_cands = {k: active_cands[k] for k in sorted_keys[:MAX_CANDIDATE_FAMILIES]}
    return active_cands


def commit_sn_observations(observations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Commit verified barcode+OCR consensus observations into pending candidates and promote if thresholds met.

    Thresholds for promotion:
    - >=2 distinct values
    - >=2 distinct label_ids
    - >=2 distinct physical source keys
    """
    if not observations:
        return {"promoted": [], "pending": [], "rejected": []}

    now = time.time()
    promoted: list[dict[str, Any]] = []
    pending_res: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    with _LOCK:
        active_rules = get_active_sn_rules()
        active_by_id = {r["id"]: r for r in active_rules}

        cand_data = _safe_read_json(sn_family_candidates_path(), default={"schema_version": 1, "candidates": {}})
        raw_cands = cand_data.get("candidates", {}) if isinstance(cand_data, dict) else {}
        if isinstance(raw_cands, list):
            candidates: dict[str, dict[str, Any]] = {
                c["id"]: c for c in raw_cands if isinstance(c, dict) and "id" in c
            }
        elif isinstance(raw_cands, dict):
            candidates = {k: v for k, v in raw_cands.items() if isinstance(v, dict)}
        else:
            candidates = {}

        candidates = _prune_and_cap_candidates(candidates, now)
        active_rules_modified = False
        candidates_modified = False

        for obs in observations:
            if not isinstance(obs, dict):
                rejected.append({"observation": obs, "reason": "invalid_observation_format"})
                continue

            raw_val = obs.get("value", "")
            label_id = str(obs.get("label_id", "") or "")
            orig_img = obs.get("original_image_path")
            cleaned_sn = _clean_sn_value(raw_val)

            family = infer_standard_sn_family(cleaned_sn)
            if family is None:
                rejected.append({"value": raw_val, "reason": "nonstandard_or_ambiguous_envelope"})
                continue

            fam_id = family["id"]
            fam_mask = family["mask"]

            if fam_id in active_by_id:
                existing_mask = active_by_id[fam_id]["mask"]
                if existing_mask != fam_mask:
                    rejected.append({"family_id": fam_id, "mask": fam_mask, "reason": "active_conflict"})
                else:
                    rejected.append({"family_id": fam_id, "mask": fam_mask, "reason": "already_active"})
                continue

            src_key = _derive_source_key(orig_img, label_id)

            cand = candidates.get(fam_id)
            if cand is None:
                cand = {
                    "id": fam_id,
                    "mask": fam_mask,
                    "prefix_digits": family["prefix_digits"],
                    "marker": family["marker"],
                    "suffix_kind": family["suffix_kind"],
                    "suffix_length": family["suffix_length"],
                    "total_length": family["total_length"],
                    "status": "pending",
                    "created_at": now,
                    "updated_at": now,
                    "evidence": {
                        "values": [cleaned_sn],
                        "label_ids": [label_id] if label_id else [],
                        "source_keys": [src_key],
                    },
                }
                candidates[fam_id] = cand
                candidates_modified = True
            else:
                if cand.get("mask") != fam_mask:
                    rejected.append({"family_id": fam_id, "mask": fam_mask, "reason": "candidate_mask_conflict"})
                    continue

                ev = cand.setdefault("evidence", {})
                values_list = ev.setdefault("values", [])
                labels_list = ev.setdefault("label_ids", [])
                sources_list = ev.setdefault("source_keys", [])

                if cleaned_sn not in values_list and len(values_list) < MAX_EVIDENCE_PER_CANDIDATE:
                    values_list.append(cleaned_sn)
                if label_id and label_id not in labels_list and len(labels_list) < MAX_EVIDENCE_PER_CANDIDATE:
                    labels_list.append(label_id)
                if src_key not in sources_list and len(sources_list) < MAX_EVIDENCE_PER_CANDIDATE:
                    sources_list.append(src_key)

                cand["updated_at"] = now
                candidates_modified = True

            ev = cand.get("evidence", {})
            d_values = len(set(ev.get("values", [])))
            d_labels = len(set(ev.get("label_ids", [])))
            d_sources = len(set(ev.get("source_keys", [])))

            if d_values >= 2 and d_labels >= 2 and d_sources >= 2:
                promoted_rule = {
                    "id": cand["id"],
                    "mask": cand["mask"],
                    "prefix_digits": cand["prefix_digits"],
                    "marker": cand["marker"],
                    "suffix_kind": cand["suffix_kind"],
                    "suffix_length": cand["suffix_length"],
                    "total_length": cand["total_length"],
                    "status": "active",
                    "promoted_at": now,
                    "evidence": {
                        "distinct_values": d_values,
                        "distinct_label_ids": d_labels,
                        "distinct_sources": d_sources,
                    },
                }
                active_by_id[fam_id] = promoted_rule
                active_rules.append(promoted_rule)
                active_rules_modified = True

                del candidates[fam_id]
                candidates_modified = True

                promoted.append(promoted_rule)
            else:
                pending_res.append(dict(cand))

        if active_rules_modified:
            _atomic_write_json(learned_sn_families_path(), {"schema_version": 1, "families": active_rules})
            reload_sn_rules()

        if candidates_modified:
            candidates = _prune_and_cap_candidates(candidates, now)
            _atomic_write_json(sn_family_candidates_path(), {"schema_version": 1, "candidates": candidates})

        return {
            "promoted": promoted,
            "pending": pending_res,
            "rejected": rejected,
        }


def reset_learned_sn_rules() -> None:
    """Safely reset and remove active and candidate learned SN rule files."""
    with _LOCK:
        reload_sn_rules()
        for path in (learned_sn_families_path(), sn_family_candidates_path()):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    try:
                        _atomic_write_json(path, {"schema_version": 1, "families": []})
                    except Exception:
                        pass
