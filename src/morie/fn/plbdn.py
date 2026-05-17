# morie.fn -- function file (hadesllm/morie)
"""Benbow-Dunning polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plbdn(data=None, n=50):
    """Benbow-Dunning polarization.

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


short = "plbdn"
alias = "plbdn"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
plbdn = plbdn


def cheatsheet() -> str:
    return "plbdn({}) -> Benbow-Dunning polarization."
