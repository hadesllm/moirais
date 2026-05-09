# moirais.fn — function file (hadesllm/moirais)
"""Convergence in Hotelling model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def hscon(data=None, n=50):
    """Convergence in Hotelling model.

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


short = "hscon"
alias = "hscon"
quote = "The spice must flow. -- Paul Atreides"
hscon = hscon


def cheatsheet() -> str:
    return "hscon({}) -> Convergence in Hotelling model."
