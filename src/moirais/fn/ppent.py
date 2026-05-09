# moirais.fn — function file (hadesllm/moirais)
"""Party entry positioning.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def ppent(data=None, n=50):
    """Party entry positioning.

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


short = "ppent"
alias = "ppent"
quote = "The spice must flow. -- Paul Atreides"
ppent = ppent


def cheatsheet() -> str:
    return "ppent({}) -> Party entry positioning."
