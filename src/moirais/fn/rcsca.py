# moirais.fn — function file (hadesllm/moirais)
"""Scaling roll call votes.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcsca(data=None, n=50):
    """Scaling roll call votes.

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


short = "rcsca"
alias = "rcsca"
quote = "The spice must flow. -- Paul Atreides"
rcsca = rcsca


def cheatsheet() -> str:
    return "rcsca({}) -> Scaling roll call votes."
