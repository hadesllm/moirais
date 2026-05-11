# morie.fn — function file (hadesllm/morie)
"""Similarity matrix roll call.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcsim(data=None, n=50):
    """Similarity matrix roll call.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    data = np.atleast_1d(data)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "rcsim"
alias = "rcsim"
quote = "The spice must flow. -- Paul Atreides"
rcsim = rcsim


def cheatsheet() -> str:
    return "rcsim({}) -> Similarity matrix roll call."
