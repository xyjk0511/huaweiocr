import json
import os
import sys

from app_paths import get_user_data_dir


PART_NO_MODEL_MAP = {
    "50087144": "AP265E",
    "50087147": "AP362E",
    "50087149": "AP162E",
    "50087288": "AP162E",
    "50087289": "AP162E",
    "50087290": "AP162E",
    "50010838": "AR180",
    "50010843": "AR180Pro",
    "98012123": "S380-L4P1T",
    "98012125": "S380-S8P2T",
    "98012403": "S110-5T",
    "98012404": "S110-8T",
    "98012406": "S110-8P1T",
}
KNOWN_MODEL_CODES = set(PART_NO_MODEL_MAP.values())
KNOWN_MODEL_CODES_UPPER = {code.upper() for code in KNOWN_MODEL_CODES}

BUILTIN_PROFILE = {
    "schema_version": 1,
    "known_model_codes": sorted(KNOWN_MODEL_CODES),
    "part_no_model_map": dict(PART_NO_MODEL_MAP),
}

_PROFILE_CACHE = None


def _warn(message: str) -> None:
    print(f"HuaweiOCR product profile: {message}", file=sys.stderr)


def _external_profile_path() -> str:
    return os.environ.get("HUAWEIOCR_PRODUCT_PROFILE") or os.path.join(
        get_user_data_dir(),
        "product_profile.json",
    )


def _read_external_profile(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        _warn(f"ignoring unreadable {path}: {exc}")
        return None
    if not isinstance(data, dict):
        _warn(f"ignoring invalid {path}: root must be an object")
        return None
    return data


def _merge_profile(external: dict | None) -> dict:
    known_model_codes = set(BUILTIN_PROFILE["known_model_codes"])
    part_no_model_map = dict(BUILTIN_PROFILE["part_no_model_map"])
    if external:
        external_codes = external.get("known_model_codes", [])
        if isinstance(external_codes, list):
            for code in external_codes:
                if isinstance(code, str):
                    known_model_codes.add(code)
        external_map = external.get("part_no_model_map", {})
        if isinstance(external_map, dict):
            for part_no, model in external_map.items():
                part_no = str(part_no)
                if part_no in BUILTIN_PROFILE["part_no_model_map"]:
                    if str(model) != BUILTIN_PROFILE["part_no_model_map"][part_no]:
                        _warn(f"ignoring conflicting part_no_model_map entry: {part_no}")
                    continue
                part_no_model_map[part_no] = str(model)
    known_model_codes.update(part_no_model_map.values())
    return {
        "schema_version": BUILTIN_PROFILE["schema_version"],
        "known_model_codes": sorted(known_model_codes),
        "part_no_model_map": part_no_model_map,
    }


def load_profile() -> dict:
    """Built-in profile plus optional additive external product profile."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = _merge_profile(_read_external_profile(_external_profile_path()))
    return _PROFILE_CACHE


def reload_profile() -> dict:
    global _PROFILE_CACHE
    _PROFILE_CACHE = None
    return load_profile()