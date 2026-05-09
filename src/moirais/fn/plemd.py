# moirais.fn — function file (hadesllm/moirais)
"""Earth mover distance polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plemd(data=None, n=50):
    """Earth mover distance polarization.

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


short = "plemd"
alias = "plemd"
quote = "The spice must flow. -- Paul Atreides"
plemd = plemd


def cheatsheet() -> str:
    return "plemd({}) -> Earth mover distance polarization."
