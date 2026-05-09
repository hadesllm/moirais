# moirais.fn — function file (hadesllm/moirais)
"""Simulation-based spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pssim(data=None, n=50):
    """Simulation-based spatial.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "pssim"
alias = "pssim"
quote = "The spice must flow. -- Paul Atreides"
pssim = pssim


def cheatsheet() -> str:
    return "pssim({}) -> Simulation-based spatial."
