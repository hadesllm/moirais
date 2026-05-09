# moirais.fn — function file (hadesllm/moirais)
"""Electoral convergence measure.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elcnv(data=None, n=50):
    """Electoral convergence measure.

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


short = "elcnv"
alias = "elcnv"
quote = "The spice must flow. -- Paul Atreides"
elcnv = elcnv


def cheatsheet() -> str:
    return "elcnv({}) -> Electoral convergence measure."
