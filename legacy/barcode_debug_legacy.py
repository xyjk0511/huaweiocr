"""Disabled legacy barcode debug prototype.

The original prototype is preserved as ``barcode_debug_legacy.py.disabled.txt``
for manual reference only. Supported barcode decoding lives in ``barcode.py``.
"""


def main() -> int:
    print("This legacy prototype is disabled. Use barcode.py through run_all.py.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
