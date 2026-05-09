# moirais.fn — function file (hadesllm/moirais)
"""Open or create the MOIRAIS SQLite cache database."""

from moirais.data import cache_connect as _fn

cconn = _fn
cache_connect = _fn


def cheatsheet() -> str:
    return "cconn() -> Open or create the MOIRAIS SQLite cache database."
