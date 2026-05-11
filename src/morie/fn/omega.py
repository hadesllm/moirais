# morie.fn — function file (hadesllm/morie)
"""Omega-squared effect size."""

from morie.fn.omega2 import omega_squared

omega = omega_squared


def cheatsheet() -> str:
    return "omega() -> Omega-squared effect size."
