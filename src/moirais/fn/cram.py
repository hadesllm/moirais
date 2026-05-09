# moirais.fn — function file (hadesllm/moirais)
"""Cramer's V effect size."""

from moirais.fn.cramv import cramers_v

cram = cramers_v


def cheatsheet() -> str:
    return "cram() -> Cramer's V effect size."
