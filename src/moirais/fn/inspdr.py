# moirais.fn — function file (hadesllm/moirais)
"""Inspect all CSV files in a directory."""

from moirais.inspector import inspect_directory as _fn

inspdr = _fn
inspect_directory = _fn


def cheatsheet() -> str:
    return "inspdr() -> Inspect all CSV files in a directory."
