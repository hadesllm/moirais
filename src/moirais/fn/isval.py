# moirais.fn — function file (hadesllm/moirais)
"""Valence advantage model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def isval(data=None, n=50):
    """Valence advantage model.

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


short = "isval"
alias = "isval"
quote = "The spice must flow. -- Paul Atreides"
isval = isval


def cheatsheet() -> str:
    return "isval({}) -> Valence advantage model."
