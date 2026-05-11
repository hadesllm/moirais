# morie.fn — function file (hadesllm/morie)
"""Condorcet median voter equilibrium.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtcl(data=None, n=50):
    """Condorcet median voter equilibrium.

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


short = "mvtcl"
alias = "mvtcl"
quote = "The spice must flow. -- Paul Atreides"
mvtcl = mvtcl


def cheatsheet() -> str:
    return "mvtcl({}) -> Condorcet median voter equilibrium."
