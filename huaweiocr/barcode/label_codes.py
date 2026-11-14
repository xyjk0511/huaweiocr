"""Label-wide codes used to cross-check the SN/PartNo fields.

Huawei device labels carry, besides the per-field Code128 barcodes:

* a QR code in ISO/IEC 15434 format 06 (``[)>`` RS ``06`` GS ``1P<PartNo>``
  GS ``18V<vendor>`` GS ``S<SN>`` RS EOT).  It is Reed-Solomon protected, so a
  decoded payload is far more trustworthy than a pixel-repaired Code128 SN.
* an EAN-13 (and on APs a UPC-A) product code, one per PartNo.

Only the Stage1 label crop is read here (label-local evidence); the original
photo can contain several labels and other parcels' QR codes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from huaweiocr.barcode.sn import _read_image, extract_sn_from_payload

try:
    import cv2
except Exception:  # OpenCV is optional at import time
    cv2 = None

_RS, _GS, _EOT = "\x1e", "\x1d", "\x04"
# zxing-cpp renders control characters as Unicode control pictures.
_CONTROL_PICTURES = str.maketrans({"␞": _RS, "␝": _GS, "␄": _EOT})
_PART_NO_RE = re.compile(r"[0-9]{8}")
_EAN13_RE = re.compile(r"[0-9]{13}")
# Whole payload: envelope, 8-digit PartNo, then vendor + serial elements.
_COMPACT_RE = re.compile(r"\[\)>06(?:1P([0-9]{8}))([0-9A-Z]+)")
_UPCA_RE = re.compile(r"[0-9]{12}")
# Upscale/sharpen variants for small or soft QR codes; tried in order and only
# when the plain read finds no QR code.
_QR_RETRY_SCALES = (1.5, 2.0, 3.0)


@dataclass(frozen=True)
class LabelQr:
    part_no: str
    sn: str
    variant: str = "plain"


@dataclass
class LabelCodes:
    qr: list[LabelQr] = field(default_factory=list)
    products: list[str] = field(default_factory=list)

    def qr_sns_for_part_no(self, part_no: str) -> list[str]:
        """Distinct QR SNs whose PartNo matches the label (all if unknown)."""
        # With a known PartNo only QR codes carrying that same PartNo count; a
        # QR without one could belong to anything the crop caught.
        sns = [q.sn for q in self.qr if (q.part_no == part_no if part_no else True)]
        return list(dict.fromkeys(sns))

    def sole_qr(self) -> LabelQr | None:
        """The label's QR when the crop holds exactly one, with a PartNo.

        Two different payloads mean the crop also caught a neighbour label,
        so neither can be attributed to this label on its own.
        """
        payloads = {(q.part_no, q.sn) for q in self.qr}
        if len(payloads) != 1:
            return None
        qr = self.qr[0]
        return qr if qr.part_no else None

    def label_product_codes(self) -> list[str]:
        """Distinct EAN-13/UPC-A codes when they can belong to one label.

        A label has at most one EAN-13 and one UPC-A (read as a 13-digit code
        with a leading 0).  More than that means the crop caught a neighbour
        label, so nothing is returned.
        """
        codes = list(dict.fromkeys(self.products))
        upcs = [code for code in codes if code.startswith("0")]
        eans = [code for code in codes if not code.startswith("0")]
        if len(upcs) > 1 or len(eans) > 1:
            return []
        return codes


def _accept_sn(candidate: str) -> str:
    """The SN in ``candidate`` (optionally followed by a short letter suffix)."""
    candidate = candidate.upper()
    extracted = extract_sn_from_payload(candidate) if candidate else ""
    if extracted and candidate.startswith(extracted) and len(candidate) - len(extracted) <= 2:
        return extracted
    return ""


def _parse_compact(payload: str) -> LabelQr | None:
    """Separator-less variant printed on some switch labels (S110-5T/8T...):
    ``[)>06 1P<PartNo> 18V<vendor> S<SN>`` with nothing between elements."""
    m = _COMPACT_RE.match(payload.strip())
    if not m:
        return None
    rest = m.group(2)
    index = rest.find("S")
    while index >= 0:
        sn = _accept_sn(rest[index + 1:])
        if sn:
            return LabelQr(part_no=m.group(1), sn=sn)
        index = rest.find("S", index + 1)
    return None


def parse_iso15434(text: str) -> LabelQr | None:
    """PartNo and SN from a Huawei ISO/IEC 15434 format-06 QR payload."""
    payload = str(text or "").translate(_CONTROL_PICTURES)
    start = payload.find("[)>" + _RS + "06" + _GS)
    if start < 0:
        return _parse_compact(payload)
    body = payload[start + 6:].split(_RS, 1)[0]
    part_no = ""
    sn = ""
    for element in body.split(_GS):
        element = element.strip()
        if element.startswith("1P") and not part_no:
            candidate = element[2:]
            if _PART_NO_RE.fullmatch(candidate):
                part_no = candidate
        elif element.startswith("S") and not sn:
            sn = _accept_sn(element[1:])
    if not sn:
        return None
    return LabelQr(part_no=part_no, sn=sn)


def _product_code(fmt_name: str, text: str) -> str:
    value = str(text or "").strip()
    if fmt_name == "EAN13" and _EAN13_RE.fullmatch(value):
        return value
    if fmt_name == "UPCA" and (_UPCA_RE.fullmatch(value) or _EAN13_RE.fullmatch(value)):
        return value.zfill(13)
    return ""


def _decode(image: Any, formats) -> list[Any]:
    import zxingcpp

    return list(zxingcpp.read_barcodes(image, formats=formats, try_rotate=True, try_invert=True))


def _qr_retry_variants(image: Any):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    for scale in _QR_RETRY_SCALES:
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        yield f"x{scale:g}", up
        blur = cv2.GaussianBlur(up, (0, 0), 2)
        yield f"x{scale:g}sharp", cv2.addWeighted(up, 1.8, blur, -0.8, 0)


def read_label_codes(path: str, *, thorough: bool = False) -> LabelCodes:
    """Decode the QR (ISO 15434) and EAN/UPC codes on a label crop.

    ``thorough`` adds upscaled retries when the plain read finds no QR; use it
    only for labels whose SN evidence is weak, since it costs ~0.5 s per label.
    """
    if cv2 is None or not path:
        return LabelCodes()
    return read_label_codes_image(_read_image(path), thorough=thorough)


def read_label_codes_image(image: Any, *, thorough: bool = False) -> LabelCodes:
    """``read_label_codes`` for a label crop already in memory (BGR)."""
    codes = LabelCodes()
    if cv2 is None or image is None:
        return codes
    try:
        import zxingcpp
    except Exception:
        return codes
    fmt = zxingcpp.BarcodeFormat
    try:
        decoded = _decode(image, [fmt.QRCode, fmt.EAN13, fmt.UPCA])
    except Exception:
        return codes
    for item in decoded:
        name = getattr(item.format, "name", str(item.format))
        if name == "QRCode":
            qr = parse_iso15434(item.text)
            if qr:
                codes.qr.append(qr)
        else:
            code = _product_code(name, item.text)
            if code:
                codes.products.append(code)
    if codes.qr or not thorough:
        return codes
    for variant, candidate in _qr_retry_variants(image):
        try:
            decoded = _decode(candidate, [fmt.QRCode])
        except Exception:
            continue
        found = [qr for qr in (parse_iso15434(item.text) for item in decoded) if qr]
        if found:
            codes.qr.extend(LabelQr(qr.part_no, qr.sn, variant) for qr in found)
            break
    return codes
