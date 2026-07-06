import re

from sn_barcode import (
    SN12_BODY_PATTERN,
    SN12_RE,
    SN20_RE,
    extract_sn_from_payload,
)
MODEL_LINE_RE = re.compile(
    r"MODEL[:：]?\s*([A-Z0-9\-]{2,32})",
    re.I,
)
S380_S8P2T_RE = re.compile(r"S380\W*S8P2T", re.I)
S380_S8P2T_NOISY_RE = re.compile(r"\bM[O0]8S\W*[O0]8[O0]2\b", re.I)

BAD_MODEL_WORDS = {
    "MODEL", "DESC", "DESCRIPTION", "QTY", "REV",
    "WAN", "LAN", "BASE", "UPC", "SN", "MAC",
}

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
MODEL_CODE_ACCEPT_RE = re.compile(
    r"(?:AP[0-9]{3,4}E|AR[0-9]{3,4}(?:PRO)?|S[0-9]{3,4}-[A-Z0-9]+)",
    re.I,
)

PART_NO_RE = re.compile(
    r"(?:^|[^0-9A-Z])(?:1P|P/N|PN|PART\s*NO\.?\s*[:：]?)?\s*(500\d{5}|980\d{5})(?=[^0-9A-Z]|[A-Z]{2,}|$)",
    re.I,
)


def extract_part_numbers_from_text(text: str) -> list[str]:
    if not text:
        return []
    seen = set()
    out = []
    for match in PART_NO_RE.finditer(str(text).upper()):
        part_no = match.group(1)
        if part_no in seen:
            continue
        seen.add(part_no)
        out.append(part_no)
    return out


def _normalize_part_no(value: str) -> str:
    matches = extract_part_numbers_from_text(str(value or ""))
    if matches:
        return matches[0]
    cleaned = re.sub(r"[^0-9]", "", str(value or ""))
    if PART_NO_RE.fullmatch(cleaned):
        return cleaned
    return ""


def normalize_model(code: str) -> str:
    c = re.sub(r"[^0-9A-Z\-]", "", code.upper())

    if len(c) % 2 == 0 and c[:len(c)//2] == c[len(c)//2:]:
        c = c[:len(c)//2]

    c = re.sub(r"DESC$", "", c)

    if c.startswith("S1108T"):
        c = "S110-8T"
    elif c.startswith("S1108P1T"):
        c = "S110-8P1T"
    if c == "S110-6T":
        c = "S110-5T"

    if c.startswith("S380S8P2T"):
        c = "S380-" + c[len("S380"):]
    if c == "S8P27":
        c = "S380-S8P2T"
    if re.match(r"^M[O0]8S-[O0]8[O0]2", c):
        c = "S380-S8P2T"
    if c in {"S380-", "S380-S", "S380-S8P", "S380-S8P2"}:
        c = "S380-S8P2T"
    if c == "AR180PRO":
        return "AR180Pro"

    m = re.match(r"^([A-Z0-9\-]*[A-Z])[0-9]{5,}$", c)
    if m:
        c = m.group(1)

    return c


def _unknown_model_candidate_is_reasonable(code: str) -> bool:
    c = normalize_model(code)
    cu = c.upper()
    if not c:
        return False
    if len(cu) < 2 or len(cu) > 24:
        return False
    if not re.search(r"\d", cu):
        return False
    if any(word in cu for word in BAD_MODEL_WORDS):
        return False
    if extract_sn_from_payload(cu) or SN20_RE.search(cu) or SN12_RE.search(cu):
        return False
    if PART_NO_RE.fullmatch(cu) or re.fullmatch(r"[0-9]{6,}", cu):
        return False
    return True


# ========= SN RULES =========

def _clean_code(s: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", s.upper())


SN_OCR_BOUNDED_RES = (
    re.compile(r"(215[0-9]{7,8}(?:ER[A-Z]?|AGQ[A-Z])[0-9]{6})(?![0-9A-Z])"),
    re.compile(r"(215[0-9]{7,8}(?:LDR[A-Z]?|LDS|SRA)[0-9]{7})(?![0-9A-Z])"),
    re.compile(r"(215[0-9]{7,8}ES[0-9A-Z]{7})(?![0-9A-Z])"),
    re.compile(rf"({SN12_BODY_PATTERN})(?![0-9A-Z])"),
)
UNKNOWN_SN_NON_PREFIX_RE = re.compile(
    r"^(SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART|PN|MODEL|DESC|ROUTE|WAYBILL|SNMP|IMEI)"
)


def _extract_sn_from_ocr_text_bounds(text: str) -> str:
    raw = (text or "").upper()
    if not raw:
        return ""
    for pattern in SN_OCR_BOUNDED_RES:
        match = pattern.search(raw)
        if match:
            return _clean_code(match.group(1))
    return ""


def extract_sn_from_text(text: str) -> str:
    sn = _extract_sn_from_ocr_text_bounds(text)
    if sn:
        return sn

    s = _clean_code(text)
    sn = extract_sn_from_payload(s)
    if sn:
        return sn

    m = SN20_RE.search(s)
    if m:
        return m.group(1)

    m = SN12_RE.search(s)
    if m and len(m.group(0)) == 12:
        return m.group(0)

    return ""


def extract_sn_from_barcode_candidate(text: str) -> str:
    return extract_sn_from_payload(text)


def _extract_unknown_sn_candidates(text: str) -> list[str]:
    raw = str(text or "").upper()
    if not raw:
        return []
    out = []
    seen = set()
    for token in re.findall(r"[A-Z0-9][A-Z0-9\-]{7,31}", raw):
        cleaned = _clean_code(token)
        while cleaned.startswith("SN"):
            cleaned = cleaned[2:]
        for prefix in ("SERIALNO", "SERIAL", "SNO"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        if not cleaned or cleaned in seen:
            continue
        if len(cleaned) < 8 or len(cleaned) > 24:
            continue
        if not re.search(r"[A-Z]", cleaned) or not re.search(r"\d", cleaned):
            continue
        if UNKNOWN_SN_NON_PREFIX_RE.match(cleaned):
            continue
        if extract_sn_from_payload(cleaned) or SN20_RE.search(cleaned) or SN12_RE.search(cleaned):
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def filter_sn_lines(lines: list[str]) -> list[str]:
    filtered = []
    for ln in lines:
        s = _clean_code(ln)
        if SN20_RE.search(s) or SN12_RE.search(s):
            filtered.append(ln)
    return filtered
