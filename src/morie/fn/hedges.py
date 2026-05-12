# morie.fn -- function file (hadesllm/morie)
"""Hedges' g effect size."""

from morie.fn.g import hedges_g

hedges = hedges_g


def cheatsheet() -> str:
    return "hedges() -> Hedges' g effect size."
