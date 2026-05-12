# morie.fn -- function file (hadesllm/morie)
"""Return full metadata for a dataset by catalog key."""

from morie.data import dataset_info as _fn

dsinfo = _fn
dataset_info = _fn


def cheatsheet() -> str:
    return "dsinfo() -> Return full metadata for a dataset by catalog key."
