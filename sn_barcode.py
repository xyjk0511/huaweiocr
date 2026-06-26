from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import cv2
except Exception:  # pragma: no cover - exercised only in stripped test envs
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


SN20_PREFIX_PATTERN = r"215[0-9]{7,8}"
SN20_BODY_PATTERN = rf"{SN20_PREFIX_PATTERN}(?:ER[A-Z]?|LDR[A-Z]?|LDS|SRA|AGQ[A-Z])[0-9]{{6,7}}"
SN12_BODY_PATTERN = r"4E[0-9]{2}(?:[0-9]{8}|[A-Z][0-9]{7})"
SN20_RE = re.compile(rf"({SN20_BODY_PATTERN})")
SN12_RE = re.compile(rf"({SN12_BODY_PATTERN})")
SERIAL_FIELD_SN_RE = re.compile(rf"S({SN20_BODY_PATTERN}|{SN12_BODY_PATTERN})(?![0-9A-Z])")
PURE_LONG_DIGITS_RE = re.compile(r"[0-9]{16,}")
DIRECT_SCANNED_SN_RE = re.compile(rf"{SN20_PREFIX_PATTERN}ES[0-9A-Z]{{7}}")

NON_SN_PREFIX_RE = re.compile(
    r"^(SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART|PN|MODEL|DESC|ROUTE|WAYBILL|SNMP|IMEI)"
)

DEFAULT_MAX_CANDIDATES = 96
DEFAULT_MAX_DECODER_ATTEMPTS = 96
DEFAULT_LABEL_MAX_DECODER_ATTEMPTS = 48
DEFAULT_MIN_BARCODE_WIDTH = 120
DEFAULT_MIN_BARCODE_HEIGHT = 22
DEFAULT_BLUR_VARIANCE = 18.0
DEFAULT_DESKEW_ANGLES = (0, -4, 4, -8, 8)
DEFAULT_DECODERS = ("pyzbar", "zxingcpp")
DEFAULT_PIXEL_REPAIR_TEMPLATE = "4E26########,###########ER@######,###########ES#######"
DEFAULT_PIXEL_REPAIR_LENGTHS = "12,20"
DEFAULT_PIXEL_REPAIR_ACCEPT_SCORE = 0.24
DEFAULT_PIXEL_REPAIR_MIN_MARGIN = 0.0
_PIXEL_REPAIR_LOCK = threading.Lock()


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _env_float(name: str, default: float, *, min_value: float = 0.0) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _path_has_non_ascii(path: str) -> bool:
    try:
        return any(ord(ch) > 127 for ch in os.fspath(path))
    except Exception:
        return False


def _read_image_unicode_safe(path: str) -> Any:
    if cv2 is None or np is None or not hasattr(np, "fromfile") or not hasattr(cv2, "imdecode"):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data is None:
            return None
        return cv2.imdecode(data, getattr(cv2, "IMREAD_COLOR", 1))
    except Exception:
        return None


def _env_flag_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_decoder_names() -> tuple[str, ...]:
    raw = os.environ.get("SN_BARCODE_DECODERS", "")
    if not raw.strip():
        return DEFAULT_DECODERS
    names = []
    for part in raw.split(","):
        name = part.strip().lower().replace("-", "_")
        if name:
            names.append(name)
    return tuple(names) or DEFAULT_DECODERS


def _selected_decoders():
    decoder_map = {
        "pyzbar": _decode_pyzbar,
        "zxingcpp": _decode_zxingcpp,
        "zxing_cpp": _decode_zxingcpp,
        "cli": _decode_cli,
        "legacy_cli": _decode_cli,
    }
    decoders = []
    for name in _env_decoder_names():
        decoder = decoder_map.get(name)
        if decoder is not None and decoder not in decoders:
            decoders.append(decoder)
    return tuple(decoders) or (_decode_pyzbar, _decode_zxingcpp)


def decoder_attempt_budget_for_source(
    source: str,
    max_decoder_attempts: int,
    *,
    has_primary_sn_source: bool = False,
) -> int:
    budget = max(1, int(max_decoder_attempts))
    if str(source).lower() == "label" and has_primary_sn_source:
        label_budget = _env_int(
            "SN_BARCODE_LABEL_MAX_DECODER_ATTEMPTS",
            DEFAULT_LABEL_MAX_DECODER_ATTEMPTS,
        )
        return min(budget, label_budget)
    return budget


@dataclass(frozen=True)
class CandidateImage:
    image: Any
    source: str
    source_region: str
    variant: str
    rotation: int = 0
    deskew_angle: int = 0
    rect: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class DecoderResult:
    decoder_name: str
    raw_text: str
    source: str
    source_region: str
    rotation: int = 0
    confidence: float | None = None
    rect: tuple[int, int, int, int] | None = None
    barcode_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder": self.decoder_name,
            "raw_text": self.raw_text,
            "source": self.source,
            "source_region": self.source_region,
            "rotation": self.rotation,
            "confidence": self.confidence,
            "rect": list(self.rect) if self.rect else None,
            "type": self.barcode_type,
        }


@dataclass(frozen=True)
class SnCandidate:
    sn: str
    raw_text: str
    source: str
    source_region: str
    decoder_name: str
    rotation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sn": self.sn,
            "raw_text": self.raw_text,
            "source": self.source,
            "source_region": self.source_region,
            "decoder": self.decoder_name,
            "rotation": self.rotation,
        }


@dataclass
class SnBarcodeReport:
    status: str
    sn: str = ""
    raw_text: str = ""
    source: str = ""
    source_region: str = ""
    decoder_name: str = ""
    attempts: int = 0
    decoded_count: int = 0
    results: list[DecoderResult] = field(default_factory=list)
    sn_candidates: list[SnCandidate] = field(default_factory=list)
    non_sn_payloads: list[str] = field(default_factory=list)
    ambiguous_sns: list[str] = field(default_factory=list)
    quality_issues: list[dict[str, Any]] = field(default_factory=list)
    decoder_errors: list[str] = field(default_factory=list)

    def to_meta(self) -> dict[str, Any]:
        return {
            "barcode_status": self.status,
            "barcode_found": self.decoded_count > 0,
            "barcode_attempts": self.attempts,
            "barcode_decoded_count": self.decoded_count,
            "barcode_sources": sorted({r.source for r in self.results}),
            "barcode_source_regions": sorted({r.source_region for r in self.results}),
            "barcode_decoder_names": sorted({r.decoder_name for r in self.results}),
            "barcode_non_sn_payloads": self.non_sn_payloads,
            "barcode_ambiguous_sns": self.ambiguous_sns,
            "barcode_quality_issues": self.quality_issues,
            "barcode_decoder_errors": self.decoder_errors,
            "barcode_results": [r.to_dict() for r in self.results],
            "barcode_sn_candidates": [c.to_dict() for c in self.sn_candidates],
        }


def _clean_code(value: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", (value or "").upper())


def extract_sn_from_payload(value: str) -> str:
    cleaned = _clean_code(value)
    if not cleaned:
        return ""
    if cleaned.startswith("F") or PURE_LONG_DIGITS_RE.fullmatch(cleaned):
        return ""
    if NON_SN_PREFIX_RE.match(cleaned):
        return ""
    while cleaned.startswith("SN"):
        cleaned = cleaned[2:]
    if "S215" in cleaned:
        cleaned = cleaned[cleaned.index("S215") + 1:]
    m = SERIAL_FIELD_SN_RE.search(cleaned)
    if m:
        return m.group(1)
    for prefix in ("SERIALNO", "SERIAL", "SNO"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if NON_SN_PREFIX_RE.match(cleaned):
        return ""
    if cleaned.startswith("F") or PURE_LONG_DIGITS_RE.fullmatch(cleaned):
        return ""

    if SN20_RE.fullmatch(cleaned):
        return cleaned
    m = SN20_RE.match(cleaned)
    if m:
        suffix = cleaned[m.end():]
        if suffix.isalpha() and len(suffix) <= 2:
            return m.group(1)

    if SN12_RE.fullmatch(cleaned):
        return cleaned

    # Some Huawei box SN barcodes encode the full scanner payload directly
    # instead of the older ERA/ERB/ERC/SRA family pattern.  Do not force those
    # through OCR; if the scanner gives one compact SN-like payload, keep it.
    if DIRECT_SCANNED_SN_RE.fullmatch(cleaned):
        return cleaned

    return ""


def _source_rank(source_region: str) -> int:
    value = (source_region or "").lower()
    if value.startswith("sn"):
        return 0
    if value.startswith("label"):
        return 1
    if value.startswith("original"):
        return 2
    if value.startswith("barcode-region") or ".region." in value:
        return 3
    return 9


def _sn_value_rank(sn: str) -> tuple[int, int]:
    if sn.startswith("215"):
        return (0, -len(sn))
    if sn.startswith("4E"):
        return (1, -len(sn))
    return (2, -len(sn))


def select_sn_from_decoder_results(results: Iterable[DecoderResult]) -> SnBarcodeReport:
    decoder_results = list(results)
    sn_candidates: list[SnCandidate] = []
    non_sn_payloads: list[str] = []
    seen_candidates: set[tuple[str, str, str, str]] = set()
    seen_payloads: set[str] = set()

    for result in decoder_results:
        raw = result.raw_text or ""
        sn = extract_sn_from_payload(raw)
        if not sn:
            if raw and raw not in seen_payloads:
                seen_payloads.add(raw)
                non_sn_payloads.append(raw)
            continue
        key = (sn, raw, result.source_region, result.decoder_name)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        sn_candidates.append(
            SnCandidate(
                sn=sn,
                raw_text=raw,
                source=result.source,
                source_region=result.source_region,
                decoder_name=result.decoder_name,
                rotation=result.rotation,
            )
        )

    best_source_rank = None
    source_rank_candidates: list[SnCandidate] = []
    for candidate in sn_candidates:
        rank = _source_rank(candidate.source_region)
        if best_source_rank is None or rank < best_source_rank:
            best_source_rank = rank
            source_rank_candidates = [candidate]
        elif rank == best_source_rank:
            source_rank_candidates.append(candidate)

    best_value_rank = None
    best_rank_candidates: list[SnCandidate] = []
    for candidate in source_rank_candidates:
        rank = _sn_value_rank(candidate.sn)
        if best_value_rank is None or rank < best_value_rank:
            best_value_rank = rank
            best_rank_candidates = [candidate]
        elif rank == best_value_rank:
            best_rank_candidates.append(candidate)

    unique_sns = sorted({candidate.sn for candidate in best_rank_candidates})
    if len(unique_sns) > 1:
        return SnBarcodeReport(
            status="ambiguous",
            attempts=0,
            decoded_count=len(decoder_results),
            results=decoder_results,
            sn_candidates=sn_candidates,
            non_sn_payloads=non_sn_payloads,
            ambiguous_sns=unique_sns,
        )

    if len(unique_sns) == 1:
        chosen = sorted(
            best_rank_candidates,
            key=lambda c: (
                _source_rank(c.source_region),
                _sn_value_rank(c.sn),
                c.rotation,
                c.decoder_name,
            ),
        )[0]
        return SnBarcodeReport(
            status="hit",
            sn=chosen.sn,
            raw_text=chosen.raw_text,
            source=chosen.source,
            source_region=chosen.source_region,
            decoder_name=chosen.decoder_name,
            attempts=0,
            decoded_count=len(decoder_results),
            results=decoder_results,
            sn_candidates=sn_candidates,
            non_sn_payloads=non_sn_payloads,
        )

    return SnBarcodeReport(
        status="parse_failure" if decoder_results else "decoder_miss",
        attempts=0,
        decoded_count=len(decoder_results),
        results=decoder_results,
        sn_candidates=[],
        non_sn_payloads=non_sn_payloads,
    )


def _read_image(path: str) -> Any:
    if cv2 is None:
        return None
    if _path_has_non_ascii(path):
        img = _read_image_unicode_safe(path)
        if img is not None:
            return img
    try:
        img = cv2.imread(path, getattr(cv2, "IMREAD_COLOR", 1))
    except Exception:
        img = None
    if img is not None:
        return img
    return _read_image_unicode_safe(path)


def _shape(image: Any) -> tuple[int, int]:
    try:
        h, w = image.shape[:2]
        return int(h), int(w)
    except Exception:
        return 0, 0


def _to_gray(image: Any) -> Any:
    if cv2 is None or image is None:
        return image
    try:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        return image


def _rotate(image: Any, rotation: int) -> Any:
    if image is None or rotation % 360 == 0:
        return image
    if cv2 is not None:
        try:
            if rotation == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            if rotation == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            if rotation == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception:
            pass
    if np is not None:
        try:
            return np.rot90(image, rotation // 90).copy()
        except Exception:
            return image
    return image


def _resize(image: Any, scale: float) -> Any:
    if cv2 is None or image is None or scale <= 1.01:
        return image
    h, w = _shape(image)
    if not h or not w:
        return image
    try:
        return cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    except Exception:
        return image


def _deskew(image: Any, angle: int) -> Any:
    if cv2 is None or image is None or angle == 0:
        return image
    h, w = _shape(image)
    if not h or not w:
        return image
    try:
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        return cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except Exception:
        return image


def pad_quiet_zone(image: Any, pad_x: int = 40, pad_y: int = 20) -> Any:
    if cv2 is None or image is None:
        return image
    value = 255
    try:
        if len(image.shape) != 2:
            value = (255, 255, 255)
    except Exception:
        pass
    try:
        return cv2.copyMakeBorder(
            image,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            borderType=cv2.BORDER_CONSTANT,
            value=value,
        )
    except Exception:
        return image


def _threshold(image: Any) -> Any:
    if cv2 is None or image is None:
        return image
    gray = _to_gray(image)
    try:
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    except Exception:
        return gray


def _enhance(image: Any) -> Any:
    if cv2 is None or image is None:
        return image
    gray = _to_gray(image)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    except Exception:
        return gray


def detect_barcode_regions(image: Any, max_regions: int = 8) -> list[tuple[int, int, int, int]]:
    if cv2 is None or image is None:
        return []
    gray = _to_gray(image)
    try:
        grad = cv2.Sobel(gray, getattr(cv2, "CV_32F", 5), 1, 0, ksize=-1)
        grad = cv2.convertScaleAbs(grad)
        grad = cv2.GaussianBlur(grad, (5, 5), 0)
        binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours_info = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    except Exception:
        return []

    h_img, w_img = _shape(image)
    regions: list[tuple[int, int, int, int]] = []
    for contour in contours:
        try:
            x, y, w, h = cv2.boundingRect(contour)
        except Exception:
            continue
        if w < DEFAULT_MIN_BARCODE_WIDTH or h < DEFAULT_MIN_BARCODE_HEIGHT:
            continue
        if w * h < 1800:
            continue
        if w / float(max(h, 1)) < 1.8:
            continue
        pad_x = max(int(w * 0.12), 12)
        pad_y = max(int(h * 0.35), 10)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        regions.append((x1, y1, x2 - x1, y2 - y1))

    regions.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
    return regions[:max_regions]


def _grid_barcode_regions(image: Any, max_regions: int = 8) -> list[tuple[int, int, int, int]]:
    h, w = _shape(image)
    if not h or not w:
        return []

    y_bands = [
        (0.00, 0.35),
        (0.25, 0.60),
        (0.50, 0.85),
        (0.65, 1.00),
    ]
    x_bands = [
        (0.00, 1.00),
        (0.00, 0.55),
        (0.45, 1.00),
        (0.20, 0.80),
    ]

    regions: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for y1f, y2f in y_bands:
        for x1f, x2f in x_bands:
            x1 = int(round(w * x1f))
            x2 = int(round(w * x2f))
            y1 = int(round(h * y1f))
            y2 = int(round(h * y2f))
            rect = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
            if rect[2] < 50 or rect[3] < 14:
                continue
            if rect in seen:
                continue
            seen.add(rect)
            regions.append(rect)
            if len(regions) >= max_regions:
                return regions
    return regions


def _sn_focus_regions(image: Any) -> list[tuple[int, int, int, int]]:
    h, w = _shape(image)
    if not h or not w:
        return []

    specs = [
        (0.00, 1.00, 0.10, 0.58),
        (0.00, 1.00, 0.16, 0.64),
        (0.02, 0.98, 0.20, 0.68),
    ]
    regions: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for x1f, x2f, y1f, y2f in specs:
        x1 = int(round(w * x1f))
        x2 = int(round(w * x2f))
        y1 = int(round(h * y1f))
        y2 = int(round(h * y2f))
        rect = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
        if rect[2] < DEFAULT_MIN_BARCODE_WIDTH or rect[3] < max(DEFAULT_MIN_BARCODE_HEIGHT * 3, 72):
            continue
        if rect in seen:
            continue
        seen.add(rect)
        regions.append(rect)
    return regions


def _candidate_scales(image: Any) -> list[float]:
    h, w = _shape(image)
    longest = max(h, w)
    if longest <= 180:
        return [1.0, 3.0, 6.0, 10.0, 14.0]
    if longest <= 360:
        return [1.0, 2.5, 5.0, 8.0, 12.0]
    if longest <= 720:
        return [1.0, 2.0, 4.0, 6.0]
    return [1.0, 1.5, 2.5]


def _base_candidate_images(
    image: Any,
    source: str,
    max_bases: int = 14,
) -> list[tuple[str, Any, int, tuple[int, int, int, int] | None]]:
    bases: list[tuple[str, Any, int, tuple[int, int, int, int] | None]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    h, w = _shape(image)
    rotations = (90, 270, 0, 180) if h > w else (0, 180, 90, 270)
    source_lc = str(source).lower()

    for base_rotation in rotations:
        oriented = _rotate(image, base_rotation)
        if source_lc == "sn":
            focus_index = 1
            for rect in _sn_focus_regions(oriented):
                x, y, w, h = rect
                key = (f"focus:{base_rotation}", x, y, w, h)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    crop = oriented[y:y + h, x:x + w]
                except Exception:
                    continue
                if crop is None or not _shape(crop)[0]:
                    continue
                region = f"{source}.rot{base_rotation}.focus.{focus_index}"
                bases.append((region, crop, base_rotation, rect))
                if len(bases) >= max_bases:
                    return bases
                focus_index += 1
        bases.append((f"{source}.rot{base_rotation}.full", oriented, base_rotation, None))
        if len(bases) >= max_bases:
            return bases
        region_index = 1
        regions = detect_barcode_regions(oriented, max_regions=4) + _grid_barcode_regions(oriented, max_regions=8)
        for rect in regions:
            x, y, w, h = rect
            key = (str(base_rotation), x, y, w, h)
            if key in seen:
                continue
            seen.add(key)
            try:
                crop = oriented[y:y + h, x:x + w]
            except Exception:
                continue
            if crop is None or not _shape(crop)[0]:
                continue
            region = f"{source}.rot{base_rotation}.region.{region_index}"
            bases.append((region, crop, base_rotation, rect))
            if len(bases) >= max_bases:
                return bases
            region_index += 1

    return bases


def diagnose_quality(image: Any) -> list[str]:
    issues: list[str] = []
    h, w = _shape(image)
    if not h or not w:
        return ["unreadable"]
    if max(w, h) < DEFAULT_MIN_BARCODE_WIDTH or min(w, h) < DEFAULT_MIN_BARCODE_HEIGHT:
        issues.append("too_small")

    if cv2 is not None:
        try:
            gray = _to_gray(image)
            lap = cv2.Laplacian(gray, getattr(cv2, "CV_64F", 6))
            variance = float(lap.var())
            if variance < DEFAULT_BLUR_VARIANCE:
                issues.append("blurred")
        except Exception:
            pass

        try:
            gray = _to_gray(image)
            band = max(2, int(w * 0.04))
            left = gray[:, :band]
            right = gray[:, max(0, w - band):]
            if float(left.mean()) < 210 or float(right.mean()) < 210:
                issues.append("quiet_zone_missing")
        except Exception:
            pass

    return issues


def iter_candidate_images(
    image: Any,
    source: str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Iterable[CandidateImage]:
    if image is None:
        return

    yielded = 0
    bases = _base_candidate_images(image, source)
    variants = [
        ("raw", lambda img: img),
        ("quiet", pad_quiet_zone),
        ("enhanced", _enhance),
        ("threshold", _threshold),
    ]

    combo_specs = [
        (0, 0, "raw"),
        (1, 0, "raw"),
        (2, 0, "raw"),
        (2, -4, "raw"),
        (2, 4, "raw"),
        (3, 0, "raw"),
        (3, -4, "raw"),
        (3, 4, "raw"),
        (2, 0, "enhanced"),
        (3, 0, "enhanced"),
        (2, 0, "threshold"),
        (3, 0, "threshold"),
        (4, 0, "raw"),
        (4, -8, "raw"),
        (4, 8, "raw"),
        (3, 0, "quiet"),
    ]
    variant_map = {name: fn for name, fn in variants}
    for scale_index, angle, variant_name in combo_specs:
        variant_fn = variant_map[variant_name]
        for source_region, base_img, base_rotation, rect in bases:
            scales = _candidate_scales(base_img)
            if scale_index >= len(scales):
                continue
            scale = scales[scale_index]
            deskewed = _deskew(base_img, angle)
            scaled = _resize(deskewed, scale)
            variant_img = variant_fn(scaled)
            yield CandidateImage(
                image=variant_img,
                source=source,
                source_region=source_region,
                variant=variant_name,
                rotation=base_rotation,
                deskew_angle=angle,
                rect=rect,
            )
            yielded += 1
            if yielded >= max_candidates:
                return


def generate_candidate_images(
    image: Any,
    source: str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[CandidateImage]:
    return list(iter_candidate_images(image, source, max_candidates=max_candidates))


_DEFAULT_GENERATE_CANDIDATE_IMAGES = generate_candidate_images


def _dump_candidate(debug_dir: str, label_id: str, index: int, candidate: CandidateImage) -> None:
    if cv2 is None or not debug_dir or candidate.image is None:
        return
    os.makedirs(debug_dir, exist_ok=True)
    digest = hashlib.sha256(f"{label_id}:{index}:{candidate.source_region}".encode("utf-8")).hexdigest()[:16]
    filename = f"candidate_{index:03d}_{digest}.png"
    try:
        cv2.imwrite(os.path.join(debug_dir, filename), candidate.image)
    except Exception:
        pass


def _decode_pyzbar(candidate: CandidateImage) -> tuple[list[DecoderResult], list[str]]:
    errors: list[str] = []
    try:
        from pyzbar import pyzbar
    except Exception as exc:
        return [], [f"pyzbar_unavailable:{exc.__class__.__name__}"]

    try:
        symbols = []
        zbar_symbol = getattr(pyzbar, "ZBarSymbol", object())
        for name in ("CODE128", "QRCODE", "CODE39", "CODE93", "I25"):
            symbol = getattr(zbar_symbol, name, None)
            if symbol is not None:
                symbols.append(symbol)
        decoded = pyzbar.decode(candidate.image, symbols=symbols or None)
    except Exception as exc:
        return [], [f"pyzbar_error:{exc.__class__.__name__}"]

    results: list[DecoderResult] = []
    for item in decoded:
        try:
            raw = item.data.decode("utf-8", errors="ignore")
        except Exception:
            raw = repr(getattr(item, "data", b""))
        rect = None
        try:
            r = item.rect
            rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        except Exception:
            pass
        if raw:
            results.append(
                DecoderResult(
                    decoder_name="pyzbar",
                    raw_text=raw,
                    source=candidate.source,
                    source_region=candidate.source_region,
                    rotation=candidate.rotation,
                    rect=rect,
                    barcode_type=getattr(item, "type", ""),
                )
            )
    return results, errors


def _decode_cli(candidate: CandidateImage) -> tuple[list[DecoderResult], list[str]]:
    try:
        import barcode as barcode_module
    except Exception as exc:
        return [], [f"barcode_module_unavailable:{exc.__class__.__name__}"]

    raw_results: list[dict[str, Any]] = []
    decoder_errors: list[str] = []
    try:
        if hasattr(barcode_module, "decode_with_cli"):
            raw_results = barcode_module.decode_with_cli(
                candidate.image,
                f"{candidate.source_region}:{candidate.variant}",
                decoder_errors=decoder_errors,
            )
        if not raw_results and hasattr(barcode_module, "decode_cli_multi"):
            raw_results = barcode_module.decode_cli_multi(
                candidate.image,
                f"{candidate.source_region}:{candidate.variant}",
                {"limit": 4, "calls": 0},
                decoder_errors=decoder_errors,
            )
        elif hasattr(barcode_module, "decode_small_patch"):
            info = barcode_module.decode_small_patch(candidate.image)
            raw_results = list(info.get("results", [])) if isinstance(info, dict) else []
            if isinstance(info, dict):
                for error in info.get("decoder_errors", []):
                    if error and error not in decoder_errors:
                        decoder_errors.append(error)
    except Exception as exc:
        decoder_errors.append(f"BarcodeReaderCLI_error:{exc.__class__.__name__}")
        return [], decoder_errors

    results: list[DecoderResult] = []
    for item in raw_results:
        raw = item.get("data", "") if isinstance(item, dict) else ""
        if not raw:
            continue
        rotation = int(item.get("rotation_k90", 0) or 0) * 90 if isinstance(item, dict) else 0
        results.append(
            DecoderResult(
                decoder_name="BarcodeReaderCLI",
                raw_text=raw,
                source=candidate.source,
                source_region=candidate.source_region,
                rotation=(candidate.rotation + rotation) % 360,
                rect=tuple(item["rect"]) if isinstance(item, dict) and item.get("rect") else None,
                barcode_type=item.get("type", "CLI") if isinstance(item, dict) else "CLI",
            )
        )
    return results, decoder_errors


def _decode_zxingcpp(candidate: CandidateImage) -> tuple[list[DecoderResult], list[str]]:
    try:
        import zxingcpp
    except Exception as exc:
        return [], [f"zxingcpp_unavailable:{exc.__class__.__name__}"]

    try:
        formats = (
            zxingcpp.BarcodeFormat.Code128
            | zxingcpp.BarcodeFormat.Code39
            | zxingcpp.BarcodeFormat.Code93
            | zxingcpp.BarcodeFormat.ITF
            | zxingcpp.BarcodeFormat.EAN13
            | zxingcpp.BarcodeFormat.QRCode
            | zxingcpp.BarcodeFormat.DataMatrix
        )
        decoded = zxingcpp.read_barcodes(
            candidate.image,
            formats=formats,
            try_rotate=True,
            try_downscale=False,
            try_invert=True,
            return_errors=False,
        )
    except Exception as exc:
        return [], [f"zxingcpp_error:{exc.__class__.__name__}"]

    results: list[DecoderResult] = []
    for item in decoded:
        raw = getattr(item, "text", "") or ""
        if not raw:
            continue
        results.append(
            DecoderResult(
                decoder_name="zxingcpp",
                raw_text=raw,
                source=candidate.source,
                source_region=candidate.source_region,
                rotation=candidate.rotation,
                barcode_type=str(getattr(item, "format", "")),
            )
        )
    return results, []


def _fallback_decode_path(
    source: str,
    path: str,
    fallback_path_decoder: Callable[[str], list[str]] | None,
) -> tuple[list[DecoderResult], int, list[str]]:
    if fallback_path_decoder is None:
        return [], 0, []
    try:
        lines = fallback_path_decoder(path)
    except Exception as exc:
        return [], 1, [f"legacy_path_decoder_error:{exc.__class__.__name__}"]
    results = [
        DecoderResult(
            decoder_name="legacy_path_decoder",
            raw_text=line,
            source=source,
            source_region=source,
            rotation=0,
            barcode_type="legacy",
        )
        for line in lines
        if line
    ]
    return results, 1, []


def _pixel_repair_decode_path(source: str, path: str) -> tuple[list[DecoderResult], int, list[str]]:
    if not _env_flag_default("SN_BARCODE_PIXEL_REPAIR", True):
        return [], 0, []
    allowed_sources = {
        part.strip().lower()
        for part in os.environ.get("SN_BARCODE_PIXEL_REPAIR_SOURCES", "sn").split(",")
        if part.strip()
    }
    if str(source).lower() not in allowed_sources:
        return [], 0, []
    if not path or not os.path.exists(path):
        return [], 0, []

    try:
        import linear_barcode_repair
    except Exception as exc:  # pragma: no cover - depends on optional local module/imports
        return [], 0, [f"code128_pixel_repair_unavailable:{exc.__class__.__name__}"]

    mode = os.environ.get("SN_BARCODE_REPAIR_CHARSET", "alnum").strip().lower() or "alnum"
    lengths = os.environ.get("SN_BARCODE_REPAIR_LENGTHS", DEFAULT_PIXEL_REPAIR_LENGTHS).strip()
    template_spec = os.environ.get("SN_BARCODE_REPAIR_TEMPLATE", DEFAULT_PIXEL_REPAIR_TEMPLATE).strip()
    regex = os.environ.get("SN_BARCODE_REPAIR_REGEX", "").strip()
    max_profiles = _env_int("SN_BARCODE_REPAIR_MAX_PROFILES", 6)
    accept_score = _env_float(
        "SN_BARCODE_REPAIR_ACCEPT_SCORE",
        DEFAULT_PIXEL_REPAIR_ACCEPT_SCORE,
    )
    min_margin = _env_float(
        "SN_BARCODE_REPAIR_MIN_MARGIN",
        DEFAULT_PIXEL_REPAIR_MIN_MARGIN,
    )

    try:
        with _PIXEL_REPAIR_LOCK:
            templates = linear_barcode_repair.parse_templates(template_spec)
            compiled_regex = linear_barcode_repair.compile_regex_or_empty(regex)
            candidates, _profiles, _strategy = linear_barcode_repair.decode_image(
                Path(path),
                mode=mode,
                lengths_arg=lengths,
                regex=compiled_regex,
                templates=templates,
                accept_score=accept_score,
                max_profiles=max_profiles,
            )
    except Exception as exc:
        return [], 1, [f"code128_pixel_repair_error:{exc.__class__.__name__}"]

    if not candidates:
        return [], 1, []

    best = candidates[0]
    score = float(getattr(best, "score", 1.0))
    if score > accept_score:
        return [], 1, [f"code128_pixel_repair_score_reject:{score:.6f}"]
    margin = float(candidates[1].score - best.score) if len(candidates) > 1 else 999.0
    if margin < min_margin:
        return [], 1, [f"code128_pixel_repair_margin_reject:{margin:.6f}"]

    raw_text = str(getattr(best, "text", "") or "")
    if not extract_sn_from_payload(raw_text):
        return [], 1, ["code128_pixel_repair_parse_failure"]

    return [
        DecoderResult(
            decoder_name="code128_pixel_repair",
            raw_text=raw_text,
            source=source,
            source_region=f"{source}.code128_pixel_repair",
            rotation=0,
            confidence=max(0.0, 1.0 - score),
            barcode_type="CODE128",
        )
    ], 1, []


def scan_sn_barcodes(
    sources: Iterable[tuple[str, str]],
    *,
    fallback_path_decoder: Callable[[str], list[str]] | None = None,
    label_id: str = "",
    max_candidates: int | None = None,
    max_decoder_attempts: int | None = None,
    debug_dir: str = "",
    early_exit: bool = True,
) -> SnBarcodeReport:
    if max_candidates is None:
        max_candidates = _env_int("SN_BARCODE_MAX_CANDIDATES", DEFAULT_MAX_CANDIDATES)
    if max_decoder_attempts is None:
        max_decoder_attempts = _env_int("SN_BARCODE_MAX_DECODER_ATTEMPTS", DEFAULT_MAX_DECODER_ATTEMPTS)

    all_results: list[DecoderResult] = []
    quality_issues: list[dict[str, Any]] = []
    decoder_errors: list[str] = []
    attempts = 0
    seen_results: set[tuple[str, str, str]] = set()
    source_items = [(source, path) for source, path in sources]
    has_primary_sn_source = any(str(source).lower() == "sn" and path for source, path in source_items)
    decoders = _selected_decoders()

    def append_results(results: Iterable[DecoderResult]) -> None:
        for result in results:
            key = (result.raw_text, result.source_region, result.decoder_name)
            if key in seen_results:
                continue
            seen_results.add(key)
            all_results.append(result)

    states: list[dict[str, Any]] = []

    for order, (source, path) in enumerate(source_items):
        if not path:
            continue
        source_budget = decoder_attempt_budget_for_source(
            source,
            max_decoder_attempts,
            has_primary_sn_source=has_primary_sn_source,
        )
        states.append(
            {
                "order": order,
                "source": source,
                "path": path,
                "budget": source_budget,
                "attempts": 0,
                "index": 0,
                "candidates": None,
                "initialized": False,
                "exhausted": False,
                "fallback_done": False,
            }
        )

    def run_fallback(state: dict[str, Any]) -> None:
        nonlocal attempts
        if state["fallback_done"]:
            return
        fallback, fallback_attempts, errors = _fallback_decode_path(
            state["source"],
            state["path"],
            fallback_path_decoder,
        )
        attempts += fallback_attempts
        decoder_errors.extend(errors)
        append_results(fallback)
        if fallback_path_decoder is not None:
            partial = select_sn_from_decoder_results(all_results)
            if partial.status not in {"hit", "ambiguous"}:
                repair, repair_attempts, repair_errors = _pixel_repair_decode_path(
                    state["source"],
                    state["path"],
                )
                attempts += repair_attempts
                decoder_errors.extend(repair_errors)
                append_results(repair)
        state["fallback_done"] = True

    def initialize_state(state: dict[str, Any]) -> None:
        if state["initialized"]:
            return
        state["initialized"] = True
        image = _read_image(state["path"])
        if image is None:
            state["exhausted"] = True
            run_fallback(state)
            return
        if generate_candidate_images is _DEFAULT_GENERATE_CANDIDATE_IMAGES:
            candidates = iter_candidate_images(image, state["source"], max_candidates=max_candidates)
        else:
            candidates = generate_candidate_images(image, state["source"], max_candidates=max_candidates)
        state["candidates"] = iter(candidates)

    def finalize_prior_fallbacks(order: int) -> None:
        for prior_state in states:
            if prior_state["order"] >= order:
                continue
            run_fallback(prior_state)

    while True:
        progressed = False
        for state in states:
            if state["exhausted"] or state["attempts"] >= state["budget"]:
                state["exhausted"] = True
                run_fallback(state)
                continue
            initialize_state(state)
            if state["exhausted"]:
                continue
            try:
                candidate = next(state["candidates"])
            except StopIteration:
                state["exhausted"] = True
                run_fallback(state)
                continue
            progressed = True
            state["index"] += 1
            if debug_dir:
                _dump_candidate(debug_dir, label_id or state["source"], state["index"], candidate)
            issues = diagnose_quality(candidate.image)
            if issues:
                quality_issues.append(
                    {
                        "source": candidate.source,
                        "source_region": candidate.source_region,
                        "variant": candidate.variant,
                        "rotation": candidate.rotation,
                        "issues": issues,
                    }
                )

            for decoder in decoders:
                if state["attempts"] >= state["budget"]:
                    break
                attempts += 1
                state["attempts"] += 1
                decoded, errors = decoder(candidate)
                decoder_errors.extend(errors)
                append_results(decoded)

                if early_exit:
                    partial = select_sn_from_decoder_results(all_results)
                    if partial.status in {"hit", "ambiguous"}:
                        finalize_prior_fallbacks(state["order"])
                        partial = select_sn_from_decoder_results(all_results)
                        partial.attempts = attempts
                        partial.quality_issues = quality_issues
                        partial.decoder_errors = decoder_errors
                        return partial

            if str(state["source"]).lower() == "sn" and not state["fallback_done"]:
                run_fallback(state)
                if early_exit:
                    partial = select_sn_from_decoder_results(all_results)
                    if partial.status in {"hit", "ambiguous"}:
                        finalize_prior_fallbacks(state["order"])
                        partial = select_sn_from_decoder_results(all_results)
                        partial.attempts = attempts
                        partial.quality_issues = quality_issues
                        partial.decoder_errors = decoder_errors
                        return partial

        if not progressed:
            break

    report = select_sn_from_decoder_results(all_results)
    report.attempts = attempts
    report.quality_issues = quality_issues
    report.decoder_errors = decoder_errors
    if report.status == "decoder_miss" and quality_issues:
        severe = {"too_small", "blurred", "quiet_zone_missing", "unreadable"}
        if any(severe.intersection(set(item.get("issues", []))) for item in quality_issues):
            report.status = "quality_reject"
    return report
