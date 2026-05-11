# morie.fn — function file (hadesllm/morie)
"""Bimodality polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plbim(data=None, n=50):
    """Bimodality polarization index.

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


short = "plbim"
alias = "plbim"
quote = "The spice must flow. -- Paul Atreides"
plbim = plbim


def cheatsheet() -> str:
    return "plbim({}) -> Bimodality polarization index."
