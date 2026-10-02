"""Compatibility entry point for huaweiocr.cli."""
import sys

import huaweiocr.cli as _impl

if __name__ == "__main__":
    sys.exit(_impl.main())
else:
    sys.modules[__name__] = _impl
