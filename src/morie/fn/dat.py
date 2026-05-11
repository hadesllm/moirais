# morie.fn — function file (hadesllm/morie)
"""Return path to the built-in morie.db database."""

from morie.data import morie_db as _fn

dat = _fn
morie_db = _fn


def cheatsheet() -> str:
    return "dat() -> Return path to the built-in morie.db database."
