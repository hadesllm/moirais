# morie.fn — function file (hadesllm/morie)
"""Fetch a dataset from CKAN and store in the SQLite cache."""

from morie.data import fetch_ckan_to_cache as _fn

ckan = _fn
fetch_ckan_to_cache = _fn


def cheatsheet() -> str:
    return "ckan() -> Fetch a dataset from CKAN and store in the SQLite cache."
