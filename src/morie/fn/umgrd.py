"""Gradient ascent spatial utility.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umgrd(data=None, n=50):
    """Gradient ascent spatial utility.

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


short = "umgrd"
alias = "umgrd"
quote = "The spice must flow. -- Paul Atreides"
umgrd = umgrd


def cheatsheet() -> str:
    return "umgrd({}) -> Gradient ascent spatial utility."
