# morie.fn -- function file (hadesllm/morie)
"""Store a DataFrame in the SQLite cache."""

from morie.data import cache_store as _fn

cstor = _fn
cache_store = _fn


def cheatsheet() -> str:
    return "cstor() -> Store a DataFrame in the SQLite cache."
