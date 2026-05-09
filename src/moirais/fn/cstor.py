# moirais.fn — function file (hadesllm/moirais)
"""Store a DataFrame in the SQLite cache."""

from moirais.data import cache_store as _fn

cstor = _fn
cache_store = _fn


def cheatsheet() -> str:
    return "cstor() -> Store a DataFrame in the SQLite cache."
