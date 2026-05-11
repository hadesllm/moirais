# morie.fn — function file (hadesllm/morie)
"""Load a dataset by catalog key (data module)."""

from morie.data import load_dataset as _fn

lds = _fn
load_dataset = _fn


def cheatsheet() -> str:
    return "lds() -> Load a dataset by catalog key (data module)."
