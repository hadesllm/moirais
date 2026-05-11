# morie.fn — function file (hadesllm/morie)
"""Electoral divergence measure.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def eldiv(data=None, n=50):
    """Electoral divergence measure.

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


short = "eldiv"
alias = "eldiv"
quote = "The spice must flow. -- Paul Atreides"
eldiv = eldiv


def cheatsheet() -> str:
    return "eldiv({}) -> Electoral divergence measure."
