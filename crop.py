"""Compatibility entry point for huaweiocr.detect.crop_pipeline."""
import sys

import huaweiocr.detect.crop_pipeline as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
