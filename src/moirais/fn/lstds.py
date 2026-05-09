# moirais.fn — function file (hadesllm/moirais)
"""List all datasets with cache status."""

from moirais.data import list_datasets as _fn

lstds = _fn
list_datasets = _fn


def cheatsheet() -> str:
    return "lstds() -> List all datasets with cache status."
