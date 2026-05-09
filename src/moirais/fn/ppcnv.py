# moirais.fn — function file (hadesllm/moirais)
"""Party convergence dynamics.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppcnv(data=None, n=50):
    """Party convergence dynamics.

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


short = "ppcnv"
alias = "ppcnv"
quote = "The spice must flow. -- Paul Atreides"
ppcnv = ppcnv


def cheatsheet() -> str:
    return "ppcnv({}) -> Party convergence dynamics."
