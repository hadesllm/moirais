# morie.fn -- function file (hadesllm/morie)
"""Inspect all CSV files in a directory."""

from morie.inspector import inspect_directory as _fn

inspdr = _fn
inspect_directory = _fn


def cheatsheet() -> str:
    return "inspdr() -> Inspect all CSV files in a directory."
