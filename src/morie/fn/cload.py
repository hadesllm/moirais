# morie.fn -- function file (hadesllm/morie)
"""Load a table from the SQLite cache."""

from morie.data import cache_load as _fn

cload = _fn
cache_load = _fn


def cheatsheet() -> str:
    return "cload() -> Load a table from the SQLite cache."
