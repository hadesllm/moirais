# moirais.fn — function file (hadesllm/moirais)
"""Omega-squared effect size."""

from moirais.fn.omega2 import omega_squared

omega = omega_squared


def cheatsheet() -> str:
    return "omega() -> Omega-squared effect size."
