# moirais.fn — function file (hadesllm/moirais)
"""List all cached tables with row counts."""

from moirais.data import cache_list as _fn

clist = _fn
cache_list = _fn


def cheatsheet() -> str:
    return "clist() -> List all cached tables with row counts."
