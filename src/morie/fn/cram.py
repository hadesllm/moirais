# morie.fn — function file (hadesllm/morie)
"""Cramer's V effect size."""

from morie.fn.cramv import cramers_v

cram = cramers_v


def cheatsheet() -> str:
    return "cram() -> Cramer's V effect size."
