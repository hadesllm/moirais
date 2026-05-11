# morie.fn — function file (hadesllm/morie)
"""Pivot model roll call.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def rcpiv(data=None, n=50):
    """Pivot model roll call.

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


short = "rcpiv"
alias = "rcpiv"
quote = "The spice must flow. -- Paul Atreides"
rcpiv = rcpiv


def cheatsheet() -> str:
    return "rcpiv({}) -> Pivot model roll call."
