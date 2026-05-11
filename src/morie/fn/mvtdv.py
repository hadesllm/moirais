# morie.fn — function file (hadesllm/morie)
"""Dynamic median voter adjustment.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtdv(data=None, n=50):
    """Dynamic median voter adjustment.

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


short = "mvtdv"
alias = "mvtdv"
quote = "The spice must flow. -- Paul Atreides"
mvtdv = mvtdv


def cheatsheet() -> str:
    return "mvtdv({}) -> Dynamic median voter adjustment."
