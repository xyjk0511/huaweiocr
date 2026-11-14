"""Recognition counters; pure barcode hits stay separate from OCR recovery."""


def new_stats() -> dict:
    return {
        "records": 0,
        "model_total": 0,
        "model_success": 0,
        "model_barcode_hits": 0,
        "model_part_no_hits": 0,
        "model_part_no_learned": 0,
        "model_consensus_learned": 0,
        "model_deferred_crops": 0,
        "model_barcode_hit_rate": 0.0,
        "model_ocr_recoveries": 0,
        "part_no_total": 0,
        "part_no_decoded": 0,
        "sn_total": 0,
        "sn_attempted": 0,
        "sn_success": 0,
        "sn_barcode_attempts": 0,
        "sn_barcode_hits": 0,
        "sn_barcode_hit_rate": 0.0,
        "sn_ocr_recoveries": 0,
        "sn_families_learned": 0,
        "sn_barcode_parse_failures": 0,
        "sn_barcode_decoder_misses": 0,
        "sn_barcode_ambiguous": 0,
        "sn_barcode_quality_rejects": 0,
        "sn_barcode_part_no_conflicts": 0,
        "sn_label_qr_rescues": 0,
        "sn_label_qr_first": 0,
        "sn_label_qr_conflicts": 0,
        "part_no_product_conflicts": 0,
        "sn_repair_ocr_conflicts": 0,
        "sn_problem": 0,
        "regex_fail": 0,
        "barcode_fail": 0,
        "ocr_fail": 0,
    }


def record_sn_metrics(stats: dict, input_available: bool, code: str, source: str, meta: dict):
    if not input_available:
        return False, ""
    stats["sn_total"] += 1
    status = meta.get("barcode_status", "not_attempted")
    stats["sn_barcode_attempts"] += int(meta.get("barcode_attempts", 0) or 0)
    if meta.get("sn_repair_check") == "ocr_conflict":
        stats["sn_repair_ocr_conflicts"] += 1
    if source in {"barcode", "barcode_label_qr", "barcode_qr_conflict"}:
        stats["sn_barcode_hits"] += 1
        if source == "barcode_label_qr" and meta.get("label_qr_check") == "rescued":
            stats["sn_label_qr_rescues"] += 1
        elif source == "barcode_label_qr":
            stats["sn_label_qr_first"] += 1
        elif source == "barcode_qr_conflict":
            stats["sn_label_qr_conflicts"] += 1
    elif code and source in {"barcode_ocr_consensus", "ocr_repair_conflict"}:
        stats["sn_ocr_recoveries"] += 1
    elif code and source.startswith("ocr") and status in {
        "decoder_miss", "parse_failure", "ambiguous", "quality_reject", "part_no_conflict",
    }:
        stats["sn_ocr_recoveries"] += 1
    status_counters = {
        "parse_failure": "sn_barcode_parse_failures",
        "decoder_miss": "sn_barcode_decoder_misses",
        "ambiguous": "sn_barcode_ambiguous",
        "quality_reject": "sn_barcode_quality_rejects",
        "part_no_conflict": "sn_barcode_part_no_conflicts",
    }
    if status in status_counters:
        stats[status_counters[status]] += 1
    evidence_found = bool(meta.get("barcode_found") or meta.get("ocr_text_found"))
    if evidence_found:
        stats["sn_attempted"] += 1
    if code:
        stats["sn_success"] += 1
        return False, ""
    stats["sn_problem"] += 1
    if evidence_found:
        stats["regex_fail"] += 1
    if not meta.get("barcode_found"):
        stats["barcode_fail"] += 1
    if not meta.get("ocr_text_found"):
        stats["ocr_fail"] += 1
    return True, source or status or "missing"


def record_model_metrics(stats: dict, input_available: bool, code: str, source: str):
    if not input_available:
        return
    stats["model_total"] += 1
    if code:
        stats["model_success"] += 1
    base_source = source.removesuffix("+sn_hint")
    # A visual match or consensus still needs OCR, including after an SN hint.
    if base_source.startswith("barcode") and base_source not in (
        "barcode_visual", "barcode_ocr_consensus",
    ):
        stats["model_barcode_hits"] += 1
    elif base_source.startswith("part_no"):
        stats["model_part_no_hits"] += 1
    elif base_source.startswith("ocr") or base_source in (
        "barcode_visual", "barcode_ocr_consensus",
    ):
        stats["model_ocr_recoveries"] += 1


def finalize_rates(stats: dict):
    if stats["sn_total"]:
        stats["sn_barcode_hit_rate"] = stats["sn_barcode_hits"] / float(stats["sn_total"])
    if stats["model_total"]:
        stats["model_barcode_hit_rate"] = stats["model_barcode_hits"] / float(stats["model_total"])
