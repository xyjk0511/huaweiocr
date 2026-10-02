"""Result interchange and Excel export shared by CLI and desktop callers."""
import json
import os


RESULT_COLUMNS = ("label_id", "model", "sn", "model_src", "sn_src")


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def export_excel(path, rows, *, headers=RESULT_COLUMNS, sheet_title="results") -> dict:
    import openpyxl

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    workbook = openpyxl.Workbook()
    try:
        sheet = workbook.active
        sheet.title = sheet_title
        sheet.append(list(headers))
        count = 0
        for row in rows:
            sheet.append(list(row))
            count += 1
        workbook.save(path)
        return {"path": path, "rows": count}
    finally:
        workbook.close()
