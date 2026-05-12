"""morie.fn -- lazy-loaded function modules (PEP 562).

The package hosts ~36k auto-generated callable modules.  Previously
``__init__.py`` was a 36k-line eager-import block; importing
``morie.fn`` parsed every submodule up front, costing minutes on
cold caches.  Replaced with a small lazy loader that reads the
symbol->submodule mapping from sibling ``_lazy_map.json`` and resolves
callables on first access via :pep:`562` ``__getattr__``.

Behaviour for callers is unchanged: both ``from morie.fn import X``
and ``import morie.fn; morie.fn.X`` resolve the same way they did in
the eager version.  Cold ``import morie.fn`` drops from ~minutes
to ~1 s.
"""

import importlib
import json
import os

_MAP_PATH = os.path.join(os.path.dirname(__file__), "_lazy_map.json")
with open(_MAP_PATH) as _f:
    _LAZY_MAP = json.load(_f)


def __getattr__(name):
    if name in _LAZY_MAP:
        mod = importlib.import_module("." + _LAZY_MAP[name], package=__name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_MAP) + ["__getattr__", "__dir__"])


__all__ = sorted(_LAZY_MAP)
