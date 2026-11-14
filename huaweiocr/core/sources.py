"""Shared labels for recognition evidence sources, without GUI dependencies."""

SOURCE_LABELS = {
    "zh": {
        "source_barcode": "扫描条形码",
        "source_barcode_ocr_consensus": "条码+文字识别一致",
        "source_barcode_visual": "条形码视觉校验",
        "source_ocr": "文字识别",
        "source_ocr_no_match": "文字识别未匹配",
        "source_barcode_ambiguous": "条形码结果冲突",
        "source_barcode_parse_fail": "条形码解析失败",
        "source_barcode_quality_reject": "条形码质量不足",
        "source_barcode_decoder_miss": "未扫到条形码",
        "source_barcode_no_match": "条形码未匹配",
        "source_part_no": "Part No模板",
        "source_part_no_no_match": "Part No未匹配",
        "source_barcode_label_qr": "标签二维码",
        "source_barcode_qr_conflict": "条码与二维码不一致（需复核）",
        "source_part_no_product_conflict": "Part No与EAN不符（需复核）",
        "source_ocr_repair_conflict": "条码修复与文字不一致，采用文字（需复核）",
        "source_missing": "缺失",
        "source_none": "未识别",
        "source_sn_hint_suffix": "+SN辅助判断",
    },
    "en": {
        "source_barcode": "barcode",
        "source_barcode_ocr_consensus": "barcode + OCR agreement",
        "source_barcode_visual": "barcode visual verification",
        "source_ocr": "OCR",
        "source_ocr_no_match": "OCR no match",
        "source_barcode_ambiguous": "barcode ambiguous",
        "source_barcode_parse_fail": "barcode parse fail",
        "source_barcode_quality_reject": "barcode quality reject",
        "source_barcode_decoder_miss": "barcode miss",
        "source_barcode_no_match": "barcode no match",
        "source_part_no": "Part No mapping",
        "source_part_no_no_match": "Part No no match",
        "source_barcode_label_qr": "label QR code",
        "source_barcode_qr_conflict": "barcode/QR mismatch (review)",
        "source_part_no_product_conflict": "Part No/EAN mismatch (review)",
        "source_ocr_repair_conflict": "OCR over repaired barcode (review)",
        "source_missing": "missing",
        "source_none": "not recognized",
        "source_sn_hint_suffix": "+SN hint",
    },
}

_SOURCE_KEYS = {
    "barcode": "source_barcode",
    "barcode_ocr_consensus": "source_barcode_ocr_consensus",
    "barcode_visual": "source_barcode_visual",
    "ocr_no_match": "source_ocr_no_match",
    "barcode_ambiguous": "source_barcode_ambiguous",
    "barcode_parse_fail": "source_barcode_parse_fail",
    "barcode_quality_reject": "source_barcode_quality_reject",
    "barcode_decoder_miss": "source_barcode_decoder_miss",
    "barcode_no_match": "source_barcode_no_match",
    "part_no_hint": "source_part_no",
    "part_no_barcode": "source_part_no",
    "part_no_learned": "source_part_no",
    "barcode_label_qr": "source_barcode_label_qr",
    "barcode_qr_conflict": "source_barcode_qr_conflict",
    "part_no_product_conflict": "source_part_no_product_conflict",
    "ocr_repair_conflict": "source_ocr_repair_conflict",
    "missing": "source_missing",
    "none": "source_none",
}


def display_source(value: str, strings=None) -> str:
    src = str(value or "")
    labels = strings or SOURCE_LABELS["zh"]
    if "+sn_hint" in src:
        return display_source(src.replace("+sn_hint", ""), labels) + labels["source_sn_hint_suffix"]
    key = _SOURCE_KEYS.get(src)
    if key:
        return labels.get(key, src)
    if src.startswith("ocr"):
        return labels["source_ocr"]
    if src.startswith("part_no"):
        return labels.get("source_part_no_no_match", src)
    return src
