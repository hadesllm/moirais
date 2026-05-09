# moirais.fn — function file (hadesllm/moirais)
"""Return full metadata for a dataset by catalog key."""

from moirais.data import dataset_info as _fn

dsinfo = _fn
dataset_info = _fn


def cheatsheet() -> str:
    return "dsinfo() -> Return full metadata for a dataset by catalog key."
