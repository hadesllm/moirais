# morie.fn — function file (hadesllm/morie)
"""Plurality electoral model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elplm(data=None, n=50):
    """Plurality electoral model.

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


short = "elplm"
alias = "elplm"
quote = "The spice must flow. -- Paul Atreides"
elplm = elplm


def cheatsheet() -> str:
    return "elplm({}) -> Plurality electoral model."
