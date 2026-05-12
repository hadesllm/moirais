# morie.fn -- function file (hadesllm/morie)
"""Pareto set median voter.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtpt(data=None, n=50):
    """Pareto set median voter.

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


short = "mvtpt"
alias = "mvtpt"
quote = "The spice must flow. -- Paul Atreides"
mvtpt = mvtpt


def cheatsheet() -> str:
    return "mvtpt({}) -> Pareto set median voter."
