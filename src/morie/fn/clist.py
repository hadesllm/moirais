# morie.fn -- function file (hadesllm/morie)
"""List all cached tables with row counts."""

from morie.data import cache_list as _fn

clist = _fn
cache_list = _fn


def cheatsheet() -> str:
    return "clist() -> List all cached tables with row counts."
