# morie.fn — function file (hadesllm/morie)
"""Cliff's delta effect size."""

from morie.fn.cliff import cliffs_delta

cliffs = cliffs_delta


def cheatsheet() -> str:
    return "cliffs() -> Cliff's delta effect size."
