"""
pycream2
========

The former name of :mod:`discotau` (renamed in October 2026). ``import pycream2``
still works but warns; use ``import discotau``. This alias will be removed in a
future release.

Every ``pycream2.<module>`` is the same module object as ``discotau.<module>``,
so setting an attribute through either name (for example swapping
``pycream2.model.response_function``) changes it for both.
"""

import importlib
import pkgutil
import sys
import warnings

import discotau

warnings.warn(
    "pycream2 has been renamed discotau: use `import discotau` instead. "
    "The pycream2 alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

for _info in pkgutil.iter_modules(discotau.__path__):
    sys.modules[f"{__name__}.{_info.name}"] = importlib.import_module(f"discotau.{_info.name}")

sys.modules[__name__] = discotau
