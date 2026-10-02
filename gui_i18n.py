"""Compatibility alias for huaweiocr.ui.i18n."""
import sys

import huaweiocr.ui.i18n as _impl

sys.modules[__name__] = _impl
