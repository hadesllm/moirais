# moirais.fn — function file (hadesllm/moirais)
"""Effective number of parties.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mpens(data=None, n=50):
    """Effective number of parties.

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


short = "mpens"
alias = "mpens"
quote = "The spice must flow. -- Paul Atreides"
mpens = mpens


def cheatsheet() -> str:
    return "mpens({}) -> Effective number of parties."
