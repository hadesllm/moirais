# moirais.fn — function file (hadesllm/moirais)
"""Cliff's delta effect size."""

from moirais.fn.cliff import cliffs_delta

cliffs = cliffs_delta


def cheatsheet() -> str:
    return "cliffs() -> Cliff's delta effect size."
