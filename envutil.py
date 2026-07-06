"""Compatibility shim: canonical implementation lives in huaweiocr.io.envutil.

The module object is aliased (not re-exported) so attribute access,
monkeypatching, and module identity all hit the real implementation.
The import below is a static statement (not importlib) so PyInstaller's
static analysis collects the implementation package into frozen builds.
"""
import sys

import huaweiocr.io.envutil as _impl

sys.modules[__name__] = _impl
