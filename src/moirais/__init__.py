"""Backwards-compatibility shim for the renamed package.

The toolkit was renamed from ``moirais`` to ``morie`` in v0.1.3. This
module re-exports everything from :mod:`morie` so existing code that
does ``import moirais`` or ``from moirais import ...`` keeps working.

Migration: replace ``import moirais`` with ``import morie``.
"""

from __future__ import annotations

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "The 'moirais' package was renamed to 'morie' in v0.1.3. "
    "Update imports from 'moirais' to 'morie'. "
    "This alias is provided for backwards compatibility and may be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

import morie as _morie
from morie import __version__  # noqa: F401

# Re-export everything from morie at package top level.
from morie import *  # noqa: F401,F403

# Alias every loaded morie submodule under the moirais. namespace so
# imports like ``from moirais.fn.crba import crba`` resolve to the
# corresponding morie module without duplicating module objects.
for _name, _mod in list(_sys.modules.items()):
    if _name == "morie" or _name.startswith("morie."):
        _alias = "moirais" + _name[len("morie"):]
        _sys.modules.setdefault(_alias, _mod)

# Install an import hook so future ``moirais.X`` imports also resolve
# to ``morie.X`` even if X hasn't been imported yet.
from importlib import import_module as _import_module
from importlib.abc import MetaPathFinder as _MetaPathFinder, Loader as _Loader
from importlib.machinery import ModuleSpec as _ModuleSpec


class _MoiraisAliasLoader(_Loader):
    def create_module(self, spec):
        target = "morie" + spec.name[len("moirais"):]
        return _import_module(target)

    def exec_module(self, module):
        return None


class _MoiraisAliasFinder(_MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "moirais" and not fullname.startswith("moirais."):
            return None
        if fullname in _sys.modules:
            return None
        target = "morie" + fullname[len("moirais"):]
        try:
            _import_module(target)
        except ImportError:
            return None
        return _ModuleSpec(fullname, _MoiraisAliasLoader(), is_package=False)


_sys.meta_path.append(_MoiraisAliasFinder())


def __getattr__(name):
    """Forward attribute access on the moirais module to morie.

    Required so ``import moirais.X; moirais.X`` returns the same
    submodule as ``import morie.X; morie.X``.
    """
    try:
        return getattr(_morie, name)
    except AttributeError:
        pass
    try:
        return _import_module("morie." + name)
    except ImportError as exc:  # pragma: no cover - mirrors stdlib behaviour
        raise AttributeError(f"module 'moirais' has no attribute {name!r}") from exc
