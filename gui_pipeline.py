"""Compatibility alias for huaweiocr.ui.pipeline."""
import sys

import huaweiocr.ui.pipeline as _impl

sys.modules[__name__] = _impl
