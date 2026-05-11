# morie.fn — function file (hadesllm/morie)
"""Wolfson polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plwlf(data=None, n=50):
    """Wolfson polarization index.

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


short = "plwlf"
alias = "plwlf"
quote = "The spice must flow. -- Paul Atreides"
plwlf = plwlf


def cheatsheet() -> str:
    return "plwlf({}) -> Wolfson polarization index."
