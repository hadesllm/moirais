# moirais.fn — function file (hadesllm/moirais)
"""Centrist competition model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpctr(data=None, n=50):
    """Centrist competition model.

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


short = "mpctr"
alias = "mpctr"
quote = "The spice must flow. -- Paul Atreides"
mpctr = mpctr


def cheatsheet() -> str:
    return "mpctr({}) -> Centrist competition model."
