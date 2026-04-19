from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

try:
    import cv2
except Exception:  # pragma: no cover - exercised only in stripped test envs
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


SN20_RE = re.compile(r"(2[0-9]{9,10}(?:ERA|ER|LDR|LDRA|SRA)[0-9]{4,7})")
SN12_RE = re.compile(r"(4E[0-9A-Z]{10})")

NON_SN_PREFIX_RE = re.compile(
    r"^(SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART|PN|MODEL|DESC|ROUTE|WAYBILL|SNMP|IMEI)"
)

DEFAULT_MAX_CANDIDATES = 32
DEFAULT_MAX_DECODER_ATTEMPTS = 64
DEFAULT_MIN_BARCODE_WIDTH = 120
DEFAULT_MIN_BARCODE_HEIGHT = 22
DEFAULT_BLUR_VARIANCE = 18.0


@dataclass(frozen=True)
class CandidateImage:
    image: Any
    source: str
    source_region: str
    variant: str
    rotation: int = 0
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
    if NON_SN_PREFIX_RE.match(cleaned):
        return ""
    while cleaned.startswith("SN"):
        cleaned = cleaned[2:]
    for prefix in ("SERIALNO", "SERIAL", "SNO"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if NON_SN_PREFIX_RE.match(cleaned):
        return ""

    m = SN20_RE.search(cleaned)
    if m and len(m.group(1)) == 20:
        return m.group(1)

    m = SN12_RE.search(cleaned)
    if m and len(m.group(1)) == 12:
        return m.group(1)

    return ""


def _source_rank(source_region: str) -> int:
    value = (source_region or "").lower()
    if value.startswith("sn"):
        return 0
    if ".region." in value or value.startswith("barcode-region"):
        return 1
    if value.startswith("label"):
        return 2
    if value.startswith("original"):
        return 3
    return 9


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

    unique_sns = sorted({candidate.sn for candidate in sn_candidates})
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
            sn_candidates,
            key=lambda c: (_source_rank(c.source_region), c.rotation, c.decoder_name),
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
    img = None
    try:
        img = cv2.imread(path, getattr(cv2, "IMREAD_COLOR", 1))
    except Exception:
        img = None
    if img is not None or np is None or not hasattr(cv2, "imdecode"):
        return img
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, getattr(cv2, "IMREAD_COLOR", 1))
    except Exception:
        return None


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


def generate_candidate_images(
    image: Any,
    source: str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[CandidateImage]:
    if image is None:
        return []

    bases: list[tuple[str, Any, tuple[int, int, int, int] | None]] = [(source, image, None)]
    for index, rect in enumerate(detect_barcode_regions(image), 1):
        x, y, w, h = rect
        try:
            crop = image[y:y + h, x:x + w]
        except Exception:
            continue
        bases.append((f"{source}.region.{index}", crop, rect))

    candidates: list[CandidateImage] = []
    rotations = [0, 90, 180, 270]
    scales = [1.0, 1.8, 2.6]
    variants = [
        ("raw", lambda img: img),
        ("quiet", pad_quiet_zone),
        ("enhanced", _enhance),
        ("threshold", _threshold),
    ]

    for source_region, base_img, rect in bases:
        for rotation in rotations:
            rotated = _rotate(base_img, rotation)
            for scale in scales:
                scaled = _resize(rotated, scale)
                for variant_name, variant_fn in variants:
                    variant_img = variant_fn(scaled)
                    candidates.append(
                        CandidateImage(
                            image=variant_img,
                            source=source,
                            source_region=source_region,
                            variant=variant_name,
                            rotation=rotation,
                            rect=rect,
                        )
                    )
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates


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
        code128 = getattr(getattr(pyzbar, "ZBarSymbol", object()), "CODE128", None)
        if code128 is not None:
            symbols = [code128]
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
    try:
        if hasattr(barcode_module, "decode_cli_multi"):
            raw_results = barcode_module.decode_cli_multi(
                candidate.image,
                f"{candidate.source_region}:{candidate.variant}",
                {"limit": 4, "calls": 0},
            )
        elif hasattr(barcode_module, "decode_small_patch"):
            info = barcode_module.decode_small_patch(candidate.image)
            raw_results = list(info.get("results", [])) if isinstance(info, dict) else []
    except Exception as exc:
        return [], [f"BarcodeReaderCLI_error:{exc.__class__.__name__}"]

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


def scan_sn_barcodes(
    sources: Iterable[tuple[str, str]],
    *,
    fallback_path_decoder: Callable[[str], list[str]] | None = None,
    label_id: str = "",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_decoder_attempts: int = DEFAULT_MAX_DECODER_ATTEMPTS,
    debug_dir: str = "",
    early_exit: bool = True,
) -> SnBarcodeReport:
    all_results: list[DecoderResult] = []
    quality_issues: list[dict[str, Any]] = []
    decoder_errors: list[str] = []
    attempts = 0
    seen_results: set[tuple[str, str, str]] = set()

    for source, path in sources:
        if not path:
            continue
        image = _read_image(path)
        if image is None:
            fallback, fallback_attempts, errors = _fallback_decode_path(source, path, fallback_path_decoder)
            attempts += fallback_attempts
            decoder_errors.extend(errors)
            all_results.extend(fallback)
            continue

        candidates = generate_candidate_images(image, source, max_candidates=max_candidates)
        for index, candidate in enumerate(candidates, 1):
            if attempts >= max_decoder_attempts:
                break
            if debug_dir:
                _dump_candidate(debug_dir, label_id or source, index, candidate)
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

            for decoder in (_decode_pyzbar, _decode_cli):
                if attempts >= max_decoder_attempts:
                    break
                attempts += 1
                decoded, errors = decoder(candidate)
                decoder_errors.extend(errors)
                for result in decoded:
                    key = (result.raw_text, result.source_region, result.decoder_name)
                    if key in seen_results:
                        continue
                    seen_results.add(key)
                    all_results.append(result)

                if early_exit:
                    partial = select_sn_from_decoder_results(all_results)
                    if partial.status == "hit":
                        partial.attempts = attempts
                        partial.quality_issues = quality_issues
                        partial.decoder_errors = decoder_errors
                        return partial

        fallback, fallback_attempts, errors = _fallback_decode_path(source, path, fallback_path_decoder)
        attempts += fallback_attempts
        decoder_errors.extend(errors)
        for result in fallback:
            key = (result.raw_text, result.source_region, result.decoder_name)
            if key not in seen_results:
                seen_results.add(key)
                all_results.append(result)

    report = select_sn_from_decoder_results(all_results)
    report.attempts = attempts
    report.quality_issues = quality_issues
    report.decoder_errors = decoder_errors
    if report.status == "decoder_miss" and quality_issues:
        severe = {"too_small", "blurred", "quiet_zone_missing", "unreadable"}
        if any(severe.intersection(set(item.get("issues", []))) for item in quality_issues):
            report.status = "quality_reject"
    return report
