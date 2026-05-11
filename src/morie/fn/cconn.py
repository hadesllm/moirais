# morie.fn — function file (hadesllm/morie)
"""Open or create the MORIE SQLite cache database."""

from morie.data import cache_connect as _fn

cconn = _fn
cache_connect = _fn


def cheatsheet() -> str:
    return "cconn() -> Open or create the MORIE SQLite cache database."
