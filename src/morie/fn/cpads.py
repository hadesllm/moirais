# morie.fn — function file (hadesllm/morie)
"""Load CPADS data: local files, cache, or CKAN API."""

from morie.data import load_cpads as _fn

cpads = _fn
load_cpads = _fn


def cheatsheet() -> str:
    return "cpads() -> Load CPADS data: local files, cache, or CKAN API."
