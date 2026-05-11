# morie.fn — function file (hadesllm/morie)
"""Midpoint competition equilibrium.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hsmpt(data=None, n=50):
    """Midpoint competition equilibrium.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(np.asarray(data, dtype=float))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "hsmpt"
alias = "hsmpt"
quote = "The spice must flow. -- Paul Atreides"
hsmpt = hsmpt


def cheatsheet() -> str:
    return "hsmpt({}) -> Midpoint competition equilibrium."
