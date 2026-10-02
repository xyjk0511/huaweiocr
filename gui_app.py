"""Compatibility entry point for huaweiocr.ui.app (also used by PyInstaller)."""
import sys

import huaweiocr.ui.app as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
