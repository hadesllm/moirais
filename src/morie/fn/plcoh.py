# morie.fn — function file (hadesllm/morie)
"""Cohesion-based polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plcoh(data=None, n=50):
    """Cohesion-based polarization.

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


short = "plcoh"
alias = "plcoh"
quote = "The spice must flow. -- Paul Atreides"
plcoh = plcoh


def cheatsheet() -> str:
    return "plcoh({}) -> Cohesion-based polarization."
