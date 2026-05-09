# moirais.fn — function file (hadesllm/moirais)
"""Hedges' g effect size."""

from moirais.fn.g import hedges_g

hedges = hedges_g


def cheatsheet() -> str:
    return "hedges() -> Hedges' g effect size."
