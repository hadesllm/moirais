# morie.fn -- function file (hadesllm/morie)
"""Eta-squared effect size."""

from morie.fn.eta2 import eta_squared

eta = eta_squared


def cheatsheet() -> str:
    return "eta() -> Eta-squared effect size."
