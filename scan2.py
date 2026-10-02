"""Compatibility entry point for huaweiocr.recognition.pipeline."""
import sys

import huaweiocr.recognition.pipeline as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
