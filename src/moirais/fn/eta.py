# moirais.fn — function file (hadesllm/moirais)
"""Eta-squared effect size."""

from moirais.fn.eta2 import eta_squared

eta = eta_squared


def cheatsheet() -> str:
    return "eta() -> Eta-squared effect size."
