# morie.fn -- function file (hadesllm/morie)
"""Mixture model spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def psmix(data=None, n=50):
    """Mixture model spatial.

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


short = "psmix"
alias = "psmix"
quote = "The spice must flow. -- Paul Atreides"
psmix = psmix


def cheatsheet() -> str:
    return "psmix({}) -> Mixture model spatial."
