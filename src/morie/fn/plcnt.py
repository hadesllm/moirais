# morie.fn — function file (hadesllm/morie)
"""Centrist polarization measure.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plcnt(data=None, n=50):
    """Centrist polarization measure.

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


short = "plcnt"
alias = "plcnt"
quote = "The spice must flow. -- Paul Atreides"
plcnt = plcnt


def cheatsheet() -> str:
    return "plcnt({}) -> Centrist polarization measure."
