# moirais.fn — function file (hadesllm/moirais)
"""Return path to the built-in moirais.db database."""

from moirais.data import moirais_db as _fn

dat = _fn
moirais_db = _fn


def cheatsheet() -> str:
    return "dat() -> Return path to the built-in moirais.db database."
